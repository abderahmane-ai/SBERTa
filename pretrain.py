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
SBERTa uses a two-phase encoder with ELECTRA-style Replaced Token Detection (RTD).

Phase 1 (n_base_layers standard attention layers) builds contextual representations
with no language signal injected. Language prototypes are assigned from Phase 1
output — contextual and meaningful — before entering Phase 2's language-aware
attention stack. This eliminates the bootstrap paradox present in architectures
that inject language distributions into raw token embeddings.

The model handles ALL masking and corruption internally. The dataset only needs
to return raw (unmasked) token sequences. Do NOT apply any masking externally.

Combined loss:
    L = L_gen  +  w_rtd · L_RTD  +  λ_cluster · L_cluster  +  λ_ortho · L_ortho

    L_gen        : generator MLM on geometrically-masked spans
    L_RTD        : discriminator real/replaced BCE at every real token
                   (6–7× more gradient signal than vanilla MLM)
    L_cluster    : Sinkhorn-Knopp on Phase 1 contextual representations;
                   prototype prior is adaptive EMA, discovering the true corpus
                   language distribution within ~500 steps automatically
    L_ortho      : (L_n L_nᵀ − I)².mean() — directly regularises prototype
                   geometry; prevents prototype vectors from drifting together
                   independently of the assignment dynamics

The model is fully zero-knowledge: no Unicode priors, no script IDs, no
dictionaries. Works on any K-language mixture (Darija, Spanglish, Hinglish, etc.).

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
    this class returns raw token ids only.

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
        buffer_ids: List[int] = []

        with open(path, encoding="utf-8", errors="replace") as f_text:
            for raw in f_text:
                sent_ids = self.tokenizer.encode(
                    raw.strip(),
                    add_sep=True,
                    max_length=self.max_length,
                    sample=True,
                    sample_alpha=0.1,
                )
                if not sent_ids:
                    continue

                if len(buffer_ids) + len(sent_ids) > self.max_length:
                    T = len(buffer_ids)
                    yield {
                        "input_ids":      torch.tensor(buffer_ids,  dtype=torch.long),
                        "attention_mask": torch.ones(T,              dtype=torch.long),
                    }
                    buffer_ids = []

                buffer_ids.extend(sent_ids)

        if buffer_ids:
            T = len(buffer_ids)
            yield {
                "input_ids":      torch.tensor(buffer_ids,  dtype=torch.long),
                "attention_mask": torch.ones(T,              dtype=torch.long),
            }

    def _get_worker_paths(self) -> List[Path]:
        """
        Return this DataLoader worker's file shard, reshuffled on every call.

        Interleaved sharding (paths[id::num_workers]) gives each worker a
        disjoint subset of files with balanced domain coverage.
        """
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.paths)
        if self.shuffle_files:
            random.shuffle(paths)
        if worker_info is None:
            return paths
        return paths[worker_info.id :: worker_info.num_workers]

    def __iter__(self) -> Iterator[dict]:
        while True:
            for path in self._get_worker_paths():
                yield from self._iter_file(path)



def collate_fn(batch: List[dict], pad_id: int) -> dict:
    """Collate variable-length samples into a padded batch."""
    T = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, T, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n]      = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
    }


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
    """Expand directory/file paths into a deduplicated flat list of .txt files."""
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

    The `latest` file is a plain-text file containing the absolute path of
    the most recent checkpoint directory (cross-platform; no symlinks).
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
) -> tuple[int, Optional[Path]]:
    """
    Load the most recent checkpoint if one exists.

    Returns:
        (global_step, ckpt_dir): step to resume from (0 if none),
                                  and the checkpoint directory (None if none).
    """
    latest_file = run_dir / "latest"
    if not latest_file.exists():
        return 0, None

    ckpt_dir = Path(latest_file.read_text(encoding="utf-8").strip())
    if not ckpt_dir.exists():
        log.warning(
            "Latest checkpoint path no longer exists: %s — starting fresh.", ckpt_dir
        )
        return 0, None

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
    return step, ckpt_dir


# ─── Training Loop ────────────────────────────────────────────────────────────


def train(
    # Model
    model_config_name: str = "base",
    # Data
    corpus_dirs: Optional[List[str]] = None,
    tokenizer_dir: str = "runs/tokenizer",
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

    # Prevents fragmentation-induced OOM under PyTorch's default allocator.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
        "Config: %s | d=%d  L=%d (base=%d lang=%d)  H=%d  ffn=%d  V=%d  K=%d",
        model_config_name,
        config.hidden_size,
        config.num_hidden_layers,
        config.n_base_layers,
        config.num_hidden_layers - config.n_base_layers,
        config.num_attention_heads,
        config.intermediate_size,
        config.vocab_size,
        config.num_languages,
    )
    log.info(
        "Loss weights: w_rtd=%.1f  λ_cluster=%.3f  λ_ortho=%.3f",
        config.rtd_weight,
        config.lambda_cluster,
        config.lambda_ortho,
    )
    log.info(
        "Span masking: geo_p=%.2f  min=%d  max=%d  (mean span %.1f tokens)",
        config.span_mask_geo_p,
        config.span_mask_min_len,
        config.span_mask_max_len,
        1.0 / config.span_mask_geo_p,
    )
    log.info(
        "Sinkhorn: epsilon=%.3f  iters=%d  lambda_cluster=%.2f",
        config.sinkhorn_epsilon,
        config.sinkhorn_iters,
        config.lambda_cluster,
    )
    log.info(
        "Prototype prior: adaptive EMA (momentum=%.3f) — initialised uniform (1/K)",
        config.prior_ema_momentum,
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
    scaler  = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    # ── Optimiser ─────────────────────────────────────────────────────────
    # Parameters excluded from weight decay:
    #   · 1-D tensors (LayerNorm weight/bias, all bias vectors)
    #   · prototype vectors L ∈ ℝ^{K×d}  (geometry, not scale)
    #   · log_tau (temperature scalar)
    #   · compat matrices (K×K language compatibility, Phase 2)
    decay_params: List[torch.Tensor] = []
    no_decay_params: List[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "prototypes.prototypes" in name
            or "rho" in name
            or ".compat" in name
            or ".beta" in name
            or ".gamma" in name
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
    global_step, ckpt_dir = load_latest_checkpoint(
        run_dir, model, optimizer, scheduler, device
    )

    # ── Training state ────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    LOSS_KEYS = ("loss", "loss_gen", "loss_rtd", "loss_cluster", "loss_ortho")
    acc: Dict[str, float] = {k: 0.0 for k in LOSS_KEYS}
    rtd_acc_sum: float = 0.0
    n_masked_acc: int  = 0

    # Prototype usage (token-level dominant prototype counts)
    proto_counts = torch.zeros(config.num_languages, device=device)
    total_tokens: int = 0

    # Switch magnitude stats (reset with loss accumulators)
    sw_sum: float = 0.0
    sw_max: float = 0.0
    sw_n:   int   = 0

    # AMP health tracking
    consecutive_skips: int = 0

    last_metrics: dict = {"step": global_step}

    t0 = time.perf_counter()
    log.info(
        "Training steps: %d → %d  (effective batch = %d)",
        global_step, total_steps, batch_size * grad_accum,
    )

    data_iter = iter(loader)

    while global_step < total_steps:

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
                )
                loss_scaled = out["loss"] / grad_accum

            scaler.scale(loss_scaled).backward()

            step_acc["loss"] += out["loss"].item() / grad_accum
            for k in LOSS_KEYS[1:]:
                step_acc[k] += out[k] / grad_accum
            step_n_masked  += out["n_masked"]
            rtd_acc_sum    += out["rtd_acc"] / grad_accum

            with torch.no_grad():
                real_mask = attention_mask.bool()

                dominant = out["language_probs"].argmax(-1)[real_mask]
                proto_counts += torch.bincount(
                    dominant, minlength=config.num_languages
                ).float()
                total_tokens += dominant.numel()

                sw = out["switch_magnitudes"][real_mask]
                sw_sum += sw.sum().item()
                sw_max  = max(sw_max, sw.max().item())
                sw_n   += sw.numel()

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
            elapsed       = time.perf_counter() - t0
            steps_per_sec = log_every / elapsed
            tok_per_sec   = steps_per_sec * batch_size * grad_accum * max_length
            lr_now        = scheduler.get_last_lr()[0]
            avg           = {k: acc[k] / log_every for k in LOSS_KEYS}

            ppl = math.exp(min(avg["loss_gen"], 20.0))

            with torch.no_grad():
                tau_val = model.sberta.prototypes.tau.item()

            # Prototype usage distribution
            if total_tokens > 0:
                pct = (proto_counts / total_tokens * 100.0).cpu().tolist()
                proto_str = "  ".join(
                    f"p{i}={pct[i]:.1f}%" for i in range(config.num_languages)
                )
                usage     = proto_counts / total_tokens
                entropy   = -(usage * (usage + 1e-9).log()).sum().item()
                max_ent   = math.log(config.num_languages)
                entropy_pct = entropy / max_ent * 100.0
                if entropy_pct < 80.0:
                    log.warning(
                        "⚠️  Prototype collapse! Entropy %.1f%% (target >80%%)",
                        entropy_pct,
                    )
            else:
                proto_str   = "N/A"
                entropy_pct = 0.0

            # Pairwise prototype cosine similarities
            with torch.no_grad():
                L_n     = F.normalize(model.sberta.prototypes.prototypes, dim=-1)
                cos_mat = (L_n @ L_n.T).cpu()
                cos_pairs = [
                    f"({i},{j})={cos_mat[i, j].item():.3f}"
                    for i in range(config.num_languages)
                    for j in range(i + 1, config.num_languages)
                ]
                cos_str = "  ".join(cos_pairs)

            # Adaptive EMA prior — shows discovered corpus distribution
            with torch.no_grad():
                prior_str = "  ".join(
                    f"p{i}={model._prototype_prior[i].item():.3f}"
                    for i in range(config.num_languages)
                )

            sw_mean_str = f"{sw_sum / sw_n:.4f}" if sw_n > 0 else "N/A"
            sw_max_str  = f"{sw_max:.4f}"         if sw_n > 0 else "N/A"

            if device.type == "cuda":
                mem_a   = torch.cuda.memory_allocated() / 1e9
                mem_r   = torch.cuda.memory_reserved()  / 1e9
                mem_str = f" | mem {mem_a:.1f}/{mem_r:.1f} GB"
            else:
                mem_str = ""

            log.info(
                "step %7d/%d  loss %.4f "
                "[gen=%.4f rtd=%.4f(acc=%.1f%%) cluster=%.4f ortho=%.4f]"
                "  ppl %.1f  τ %.3f  masked/step %.0f"
                "  lr %.2e  ‖g‖ %.3f%s"
                "  %.1f stp/s  %.0f tok/s",
                global_step, total_steps,
                avg["loss"],
                avg["loss_gen"],
                avg["loss_rtd"], rtd_acc_sum / log_every * 100.0,
                avg["loss_cluster"], avg["loss_ortho"],
                ppl, tau_val,
                n_masked_acc / log_every,
                lr_now, grad_norm, mem_str,
                steps_per_sec, tok_per_sec,
            )
            log.info("  prototypes  : %s  entropy=%.1f%%", proto_str, entropy_pct)
            log.info("  cosines     : %s", cos_str)
            log.info("  EMA prior   : %s", prior_str)
            log.info("  switch mag  : mean=%s  max=%s", sw_mean_str, sw_max_str)

            last_metrics = {
                "step":              global_step,
                "loss":              avg["loss"],
                "loss_gen":          avg["loss_gen"],
                "loss_rtd":          avg["loss_rtd"],
                "rtd_acc":           rtd_acc_sum / log_every,
                "loss_cluster":      avg["loss_cluster"],
                "loss_ortho":        avg["loss_ortho"],
                "ppl":               ppl,
                "tau":               tau_val,
                "lr":                lr_now,
                "proto_entropy_pct": entropy_pct,
            }

            acc          = {k: 0.0 for k in LOSS_KEYS}
            n_masked_acc = 0
            rtd_acc_sum  = 0.0
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
    p.add_argument("--num-workers",      type=int,   default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_config_name=args.config,
        corpus_dirs=args.corpus_dirs,
        tokenizer_dir=args.tokenizer_dir,
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