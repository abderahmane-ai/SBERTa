"""
pretrain.py — SBERTa pre-training loop.

Usage
-----
Smoke-test (CPU / single GPU):
    python pretrain.py --config small --total-steps 1000 --batch-size 8

Full run:
    python pretrain.py \\
        --config base \\
        --corpus-dirs corpus \\
        --tokenizer-dir runs/tokenizer \\
        --total-steps 1000000 \\
        --batch-size 32 \\
        --grad-accum 4 \\
        --run-id run-001

Architecture
------------
SBERTa uses ELECTRA-style Replaced Token Detection (RTD) pre-training.

The model handles ALL masking and corruption internally.  The dataset only
needs to return raw (unmasked) token sequences.  Do NOT apply any masking
in the dataset or pass any external switch labels — the model discovers
language boundaries autonomously via the unsupervised L_smooth loss.

Combined loss:
    L = L_gen  +  w_rtd · L_RTD  +  λ_smooth · w_curr · L_smooth  +  λ_div · L_div

    L_gen        : generator MLM on switch-span-masked positions
    L_RTD        : discriminator real/replaced BCE at every real token
                   (6–7× more gradient signal than vanilla MLM)
    L_smooth     : unsupervised temporal stickiness — mean switch magnitude
                   penalised so prototypes self-organise into language blocks
                   (activates linearly after smooth_warmup_steps; no external
                   labels or fastText model required)
    L_div        : prototype diversity — prevents prototype collapse

Checkpointing
-------------
Every `checkpoint_every` steps the trainer writes:
    runs/<run_id>/step-<N>/model.pt
    runs/<run_id>/step-<N>/optimizer.pt
    runs/<run_id>/step-<N>/scheduler.pt
    runs/<run_id>/step-<N>/config.json
    runs/<run_id>/step-<N>/metrics.json
    runs/<run_id>/latest          (plain-text pointer, updated atomically)

Training resumes automatically from the latest checkpoint when found.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Dataset ──────────────────────────────────────────────────────────────────


class StreamingTextDataset(IterableDataset):
    """
    Infinite streaming dataset for SBERTa pre-training.

    Reads plain-text UTF-8 corpus files (one sentence/segment per line),
    tokenises with SBERTaTokenizer, and yields individual samples as dicts.
    Masking and language-boundary detection are handled by the model —
    this class returns raw token ids only (no switch labels required).

    Args:
        corpus_paths:  list of plain-text corpus files.
        tokenizer:     SBERTaTokenizer instance.
        max_length:    maximum sequence length (truncation applied here).
        shuffle_files: shuffle file order at the start of each pass.
    """

    def __init__(
        self,
        corpus_paths: List[Path],
        tokenizer,
        max_length: int = 512,
        shuffle_files: bool = True,
    ) -> None:
        super().__init__()
        self.paths = list(corpus_paths)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_files = shuffle_files

    def _iter_file(self, path: Path) -> Iterator[dict]:
        with open(path, encoding="utf-8", errors="replace") as f_text:
            for raw in f_text:
                ids_content: List[int] = self.tokenizer.encode(
                    raw.strip(),
                    add_sep=False,
                    max_length=self.max_length - 1,   # reserve one slot for SEP
                )
                if not ids_content:
                    continue

                ids: List[int] = ids_content + [self.tokenizer.SEP_ID]
                T = len(ids)
                yield {
                    "input_ids":      torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.ones(T, dtype=torch.long),
                }

    def _get_worker_paths(self) -> List[Path]:
        """
        Return this DataLoader worker's file shard, reshuffled on every call
        (so each pass through the data uses a different file order).

        When num_workers > 0, PyTorch forks N worker processes each of which
        calls __iter__ independently.  Without sharding every worker would
        stream the same files, effectively multiplying the corpus N-fold.
        Interleaved sharding (paths[id::num_workers]) gives each worker a
        disjoint subset with balanced coverage across domains.
        """
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.paths)
        if self.shuffle_files:
            random.shuffle(paths)
        if worker_info is None:
            return paths
        # Interleaved sharding: worker i gets paths[i], paths[i+n], paths[i+2n], …
        return paths[worker_info.id :: worker_info.num_workers]

    def __iter__(self) -> Iterator[dict]:
        # Loop forever so DataLoader never raises StopIteration mid-training.
        while True:
            for path in self._get_worker_paths():
                yield from self._iter_file(path)


class DomainWeightedStreamingDataset(IterableDataset):
    """
    Infinite weighted mixture over named domain datasets.

    At each call to __next__ a domain is sampled proportional to its weight,
    then one sample is drawn from that domain's iterator.  Exhausted iterators
    are silently restarted so the stream is infinite.

    Each DataLoader worker gets its own seeded RNG so workers produce
    independent domain sequences rather than identical correlated ones.

    Args:
        domain_datasets: domain name → StreamingTextDataset.
        domain_weights:  domain name → unnormalised sampling weight.
    """

    def __init__(
        self,
        domain_datasets: Dict[str, "StreamingTextDataset"],
        domain_weights: Dict[str, float],
    ) -> None:
        super().__init__()
        self.domains = list(domain_datasets.keys())
        self.datasets = domain_datasets
        total = sum(domain_weights[d] for d in self.domains)
        self.probs = [domain_weights[d] / total for d in self.domains]

    def __iter__(self) -> Iterator[dict]:
        # Give each worker an independent RNG so domain sequences diverge.
        worker_info = torch.utils.data.get_worker_info()
        seed = int(torch.initial_seed() % (2 ** 32))
        if worker_info is not None:
            seed ^= worker_info.id          # xor with worker id for independence
        rng = random.Random(seed)

        iters = {d: iter(self.datasets[d]) for d in self.domains}
        while True:
            (domain,) = rng.choices(self.domains, weights=self.probs, k=1)
            try:
                yield next(iters[domain])
            except StopIteration:
                iters[domain] = iter(self.datasets[domain])
                yield next(iters[domain])


def collate_fn(batch: List[dict], pad_id: int) -> dict:
    """
    Collate a list of variable-length samples into a padded batch.

    Padding strategy: pad to the length of the longest sequence in the batch
    (not to max_length) to avoid wasting compute on short-sequence batches.
    """
    T = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, T, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n]      = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]

    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ─── Scheduler ────────────────────────────────────────────────────────────────


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to 0."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Corpus utilities ─────────────────────────────────────────────────────────


def resolve_corpus_paths(dirs: List[str]) -> List[Path]:
    """
    Expand a list of directory paths or .txt file paths into a deduplicated
    flat list of .txt files.
    """
    paths: List[Path] = []
    seen: set = set()
    for d in dirs:
        entry = Path(d)
        if entry.is_file() and entry.suffix.lower() == ".txt":
            found = [entry]
        elif entry.is_dir():
            found = sorted(entry.rglob("*.txt"))
        else:
            log.warning("Path not found or not a .txt file — skipping: %s", d)
            continue
        if not found:
            log.warning("No .txt files found in: %s", d)
            continue
        for p in found:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                paths.append(p)

    if not paths:
        raise FileNotFoundError(
            f"No corpus .txt files found in any of: {dirs}\n"
            "Check --corpus-dirs."
        )
    total_mb = sum(p.stat().st_size for p in paths) / 1e6
    log.info("Corpus: %d files, %.1f MB total", len(paths), total_mb)
    return paths


def build_domain_buckets(corpus_paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Bucket corpus files into broad domains based on path components.

    Current mapping:
        wikipedia — any file whose path contains '/wikipedia/'
        darija    — everything else (YouTube comments, native Darija, sentiment…)

    Extend this function if you add more domains.
    """
    buckets: Dict[str, List[Path]] = {"darija": [], "wikipedia": []}
    for p in corpus_paths:
        norm = p.as_posix().lower()
        if "/wikipedia/" in norm:
            buckets["wikipedia"].append(p)
        else:
            buckets["darija"].append(p)
    return {k: v for k, v in buckets.items() if v}


def parse_domain_weights(spec: str) -> Dict[str, float]:
    """
    Parse a comma-separated domain-weight string into a dict.

    Example:
        "darija=0.7,wikipedia=0.3"  →  {"darija": 0.7, "wikipedia": 0.3}
    """
    out: Dict[str, float] = {}
    if not spec or not spec.strip():
        return out
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid domain weight spec '{item}'. Expected key=value."
            )
        key, raw_val = item.split("=", 1)
        domain = key.strip().lower()
        try:
            weight = float(raw_val.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight value '{raw_val}' for domain '{domain}'."
            ) from exc
        if weight <= 0:
            raise ValueError(f"Weight must be > 0; got {weight} for '{domain}'.")
        out[domain] = weight
    return out


# ─── Checkpointing ────────────────────────────────────────────────────────────


def save_checkpoint(
    run_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config,
    metrics: dict,
) -> Path:
    """
    Save a full training checkpoint and update the `latest` pointer file.

    The `latest` file is a plain-text file containing the absolute path of the
    most recent checkpoint directory.  Plain text is used instead of a symlink
    for cross-platform compatibility.
    """
    ckpt_dir = run_dir / f"step-{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(),      ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(),  ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(),  ckpt_dir / "scheduler.pt")
    config.save(ckpt_dir / "config.json")
    (ckpt_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (run_dir / "latest").write_text(str(ckpt_dir), encoding="utf-8")

    log.info("Checkpoint saved → %s", ckpt_dir)
    return ckpt_dir


def load_latest_checkpoint(
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    """
    Load the most recent checkpoint if one exists.

    Returns:
        The global step to resume from, or 0 if no checkpoint is found.
    """
    latest_file = run_dir / "latest"
    if not latest_file.exists():
        return 0

    ckpt_dir = Path(latest_file.read_text(encoding="utf-8").strip())
    if not ckpt_dir.exists():
        log.warning(
            "Latest checkpoint path no longer exists: %s — starting fresh.", ckpt_dir
        )
        return 0

    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt",     map_location=device, weights_only=True)
    )
    optimizer.load_state_dict(
        torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(ckpt_dir / "scheduler.pt", map_location=device, weights_only=True)
    )

    metrics = json.loads((ckpt_dir / "metrics.json").read_text(encoding="utf-8"))
    step = int(metrics.get("step", 0))
    log.info("Resumed from %s (step %d)", ckpt_dir, step)
    return step


# ─── Training Loop ────────────────────────────────────────────────────────────


def train(
    # Model
    model_config_name: str = "base",
    # Data
    corpus_dirs: Optional[List[str]] = None,
    tokenizer_dir: str = "runs/tokenizer",
    domain_weights: Optional[Dict[str, float]] = None,
    # Optimisation
    total_steps: int = 150_000,
    warmup_steps: int = 3_000,
    batch_size: int = 256,
    grad_accum: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    # Sequence
    max_length: int = 512,
    # Logging / checkpointing
    run_id: str = "run-001",
    runs_dir: str = "runs",
    checkpoint_every: int = 5_000,
    log_every: int = 100,
    # DataLoader
    num_workers: int = 16,
) -> None:
    from sberta.config import SBERTaConfig
    from sberta.model import SBERTaForPreTraining
    from sberta.tokenizer import SBERTaTokenizer

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s (%.1f GB VRAM)", props.name, props.total_memory / 1e9)

    # ── Model config ──────────────────────────────────────────────────────
    config_factory = {
        "small":  SBERTaConfig.small,
        "medium": SBERTaConfig.medium,
        "base":   SBERTaConfig.base,
        "large":  SBERTaConfig.large,
    }
    if model_config_name not in config_factory:
        raise ValueError(
            f"Unknown config '{model_config_name}'. "
            f"Valid choices: {list(config_factory)}."
        )
    config = config_factory[model_config_name]()
    log.info(
        "Config: %s | d=%d  L=%d  H=%d  ffn=%d  V=%d  K=%d",
        model_config_name,
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.intermediate_size,
        config.vocab_size,
        config.num_languages,
    )
    log.info(
        "Loss weights: w_rtd=%.1f  λ_smooth=%.3f  λ_div=%.3f",
        config.rtd_weight,
        config.lambda_smooth,
        config.lambda_div,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = SBERTaTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) != "
            f"config.vocab_size ({config.vocab_size}). "
            "Retrain the tokenizer with the correct --vocab_size."
        )

    # ── Dataset ───────────────────────────────────────────────────────────
    if corpus_dirs is None:
        corpus_dirs = ["corpus"]
    corpus_paths = resolve_corpus_paths(corpus_dirs)

    if domain_weights:
        domain_paths = build_domain_buckets(corpus_paths)
        active: Dict[str, float] = {}
        for domain, weight in domain_weights.items():
            if domain not in domain_paths:
                log.warning(
                    "Domain '%s' specified in --domain-weights but no files "
                    "bucketed into it — skipping.",
                    domain,
                )
                continue
            active[domain] = weight
        if not active:
            raise ValueError(
                f"No active weighted domains found. "
                f"Domains with files: {sorted(domain_paths.keys())}."
            )
        log.info("Domain sampling: %s", active)
        domain_datasets = {
            d: StreamingTextDataset(
                domain_paths[d], tokenizer, max_length, shuffle_files=True
            )
            for d in active
        }
        dataset = DomainWeightedStreamingDataset(domain_datasets, active)
    else:
        dataset = StreamingTextDataset(
            corpus_paths, tokenizer, max_length, shuffle_files=True
        )

    # ── DataLoader ────────────────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.PAD_ID),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = SBERTaForPreTraining(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %.1f M", n_params / 1e6)

    # ── Mixed precision ────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    # ── Optimiser ─────────────────────────────────────────────────────────
    # Parameters excluded from weight decay:
    #   · 1-D tensors (LayerNorm weight/bias, all bias vectors)
    #   · prototype vectors L ∈ ℝ^{K×d}  (geometry, not scale)
    #   · switch_embedding  e^{sw}
    #   · log_tau            (temperature; scalar)
    decay_params: List[torch.Tensor] = []
    no_decay_params: List[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "prototypes.prototypes" in name
            or "switch_embedding" in name
            or "log_tau" in name
            or ".compat" in name             # K×K language compatibility matrices
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    log.info(
        "Optimiser groups: %d with decay, %d without",
        len(decay_params), len(no_decay_params),
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Run directory + optional resume ───────────────────────────────────
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    global_step = load_latest_checkpoint(
        run_dir, model, optimizer, scheduler, device
    )

    # ── Training state ────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    # Loss accumulators — reset every log_every optimizer steps.
    LOSS_KEYS = ("loss", "loss_gen", "loss_rtd", "loss_smooth", "loss_div", "loss_sharp")
    acc: Dict[str, float] = {k: 0.0 for k in LOSS_KEYS}
    n_masked_acc: int = 0
    smooth_weight_acc: float = 0.0

    # Prototype usage (token-level dominant prototype counts)
    proto_counts = torch.zeros(config.num_languages, device=device)
    total_tokens: int = 0

    # Switch magnitude stats (reset with loss accumulators)
    sw_sum: float = 0.0
    sw_max: float = 0.0
    sw_n:   int   = 0

    # AMP health tracking
    consecutive_skips: int = 0

    # Last logged metrics (used if a checkpoint fires before first log window)
    last_metrics: dict = {"step": global_step}

    t0 = time.perf_counter()
    log.info("Training steps: %d → %d  (effective batch = %d)",
             global_step, total_steps, batch_size * grad_accum)

    data_iter = iter(loader)

    while global_step < total_steps:

        # ── Gradient-accumulation micro-steps ─────────────────────────────
        # Accumulate gradients over `grad_accum` micro-batches before calling
        # optimizer.step().  Loss components are averaged across micro-batches
        # for each optimizer step, then accumulated across steps for logging.
        step_acc: Dict[str, float] = {k: 0.0 for k in LOSS_KEYS}
        step_n_masked: int = 0

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_step=global_step,
                )
                loss_scaled = out["loss"] / grad_accum

            scaler.scale(loss_scaled).backward()

            step_acc["loss"] += out["loss"].item() / grad_accum
            for k in LOSS_KEYS[1:]:
                step_acc[k] += out[k] / grad_accum
            step_n_masked += out["n_masked"]
            smooth_weight_acc += out["smooth_weight"] / grad_accum

            # Prototype usage and switch stats (no grad needed)
            with torch.no_grad():
                real_mask = attention_mask.bool()

                # Dominant prototype per real token — from context-refined p_ctx
                dominant = out["language_probs"].argmax(-1)[real_mask]  # (N_real,)
                proto_counts += torch.bincount(
                    dominant, minlength=config.num_languages
                ).float()
                total_tokens += dominant.numel()

                # Switch magnitudes from the discriminator forward
                sw = out["switch_magnitudes"][real_mask]  # (N_real,)
                sw_sum += sw.sum().item()
                sw_max  = max(sw_max, sw.max().item())
                sw_n   += sw.numel()

        # Accumulate step-level averages into the logging window
        for k in LOSS_KEYS:
            acc[k] += step_acc[k]
        n_masked_acc += step_n_masked // grad_accum

        # ── Optimiser step ────────────────────────────────────────────────
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Only advance the LR schedule when the optimiser actually stepped
        # (AMP skips the step if NaN/Inf gradients are detected).
        if scaler.get_scale() < scale_before:
            consecutive_skips += 1
            log.warning(
                "AMP skipped optimizer step %d "
                "(NaN/Inf in gradients, scale %.0f → %.0f) "
                "[consecutive skips: %d]",
                global_step, scale_before, scaler.get_scale(), consecutive_skips,
            )
            if consecutive_skips >= 5:
                log.error(
                    "5+ consecutive AMP skips — training is likely diverging. "
                    "Consider reducing --lr or inspecting gradient norms."
                )
        else:
            consecutive_skips = 0
            scheduler.step()

        global_step += 1

        # ── Periodic logging ──────────────────────────────────────────────
        if global_step % log_every == 0:
            elapsed = time.perf_counter() - t0
            steps_per_sec = log_every / elapsed
            tok_per_sec   = steps_per_sec * batch_size * grad_accum * max_length
            lr_now        = scheduler.get_last_lr()[0]
            avg           = {k: acc[k] / log_every for k in LOSS_KEYS}

            # Generator perplexity (cap exponent to avoid overflow)
            ppl = math.exp(min(avg["loss_gen"], 20.0))

            # Learnable temperature τ
            with torch.no_grad():
                tau_val = model.sberta.prototypes.tau.item()

            # Prototype usage distribution
            if total_tokens > 0:
                pct = (proto_counts / total_tokens * 100.0).cpu().tolist()
                proto_str = "  ".join(
                    f"p{i}={pct[i]:.1f}%" for i in range(config.num_languages)
                )
            else:
                proto_str = "N/A"

            # Pairwise prototype cosine similarities — monitors diversity loss
            with torch.no_grad():
                L_n = F.normalize(
                    model.sberta.prototypes.prototypes, dim=-1
                )  # (K, d)
                cos_mat = (L_n @ L_n.T).cpu()
                cos_pairs = [
                    f"({i},{j})={cos_mat[i, j].item():.3f}"
                    for i in range(config.num_languages)
                    for j in range(i + 1, config.num_languages)
                ]
                cos_str = "  ".join(cos_pairs)

            # Switch magnitude stats
            sw_mean_str = f"{sw_sum / sw_n:.4f}" if sw_n > 0 else "N/A"
            sw_max_str  = f"{sw_max:.4f}"         if sw_n > 0 else "N/A"

            # GPU memory
            if device.type == "cuda":
                mem_a = torch.cuda.memory_allocated() / 1e9
                mem_r = torch.cuda.memory_reserved()  / 1e9
                mem_str = f" | mem {mem_a:.1f}/{mem_r:.1f} GB"
            else:
                mem_str = ""

            log.info(
                "step %7d/%d  loss %.4f "
                "[gen=%.4f rtd=%.4f smooth=%.4f(w=%.2f) div=%.4f sharp=%.4f]"
                "  ppl %.1f  τ %.3f  masked/step %.0f"
                "  lr %.2e  ‖g‖ %.3f%s"
                "  %.1f stp/s  %.0f tok/s",
                global_step, total_steps,
                avg["loss"],
                avg["loss_gen"], avg["loss_rtd"],
                avg["loss_smooth"], smooth_weight_acc / log_every,
                avg["loss_div"], avg["loss_sharp"],
                ppl, tau_val,
                n_masked_acc / log_every,
                lr_now, grad_norm, mem_str,
                steps_per_sec, tok_per_sec,
            )
            log.info("  prototypes : %s", proto_str)
            log.info("  cosines    : %s", cos_str)
            log.info("  switch mag : mean=%s  max=%s", sw_mean_str, sw_max_str)

            # Snapshot for checkpoint metadata
            last_metrics = {
                "step":             global_step,
                "loss":             avg["loss"],
                "loss_gen":         avg["loss_gen"],
                "loss_rtd":         avg["loss_rtd"],
                "loss_smooth":      avg["loss_smooth"],
                "smooth_weight":    smooth_weight_acc / log_every,
                "loss_div":         avg["loss_div"],
                "loss_sharp":       avg["loss_sharp"],
                "ppl":              ppl,
                "tau":              tau_val,
                "lr":               lr_now,
            }

            acc          = {k: 0.0 for k in LOSS_KEYS}
            n_masked_acc = 0
            smooth_weight_acc = 0.0
            proto_counts.zero_()
            total_tokens = 0
            sw_sum = sw_max = 0.0
            sw_n   = 0
            t0     = time.perf_counter()

        # ── Checkpoint ────────────────────────────────────────────────────
        if global_step % checkpoint_every == 0 or global_step == total_steps:
            save_checkpoint(
                run_dir, global_step,
                model, optimizer, scheduler,
                config, last_metrics,
            )

    log.info("Training complete at step %d.", global_step)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SBERTa pre-training (ELECTRA-style RTD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default="base",
        choices=["small", "medium", "base", "large"],
        help="Model size preset.",
    )
    p.add_argument(
        "--corpus-dirs", nargs="+", default=["corpus"],
        help="Corpus directories or individual .txt file paths.",
    )
    p.add_argument(
        "--domain-weights", type=str, default="",
        help=(
            "Optional weighted domain sampling, e.g. 'darija=0.7,wikipedia=0.3'. "
            "When omitted all files are sampled uniformly."
        ),
    )
    p.add_argument(
        "--tokenizer-dir", default="runs/tokenizer",
        help="Directory containing sberta.model.",
    )
    p.add_argument("--total-steps",      type=int,   default=1_000_000)
    p.add_argument("--warmup-steps",     type=int,   default=10_000)
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps. Effective batch = batch-size × grad-accum.",
    )
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--max-grad-norm",    type=float, default=1.0)
    p.add_argument("--max-length",       type=int,   default=512)
    p.add_argument("--run-id",           type=str,   default="run-001")
    p.add_argument("--runs-dir",         type=str,   default="runs")
    p.add_argument("--checkpoint-every", type=int,   default=5_000)
    p.add_argument("--log-every",        type=int,   default=100)
    p.add_argument("--num-workers",      type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_config_name=args.config,
        corpus_dirs=args.corpus_dirs,
        tokenizer_dir=args.tokenizer_dir,
        domain_weights=parse_domain_weights(args.domain_weights) or None,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        num_workers=args.num_workers,
    )