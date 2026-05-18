"""Darija-first SBERTa pre-training.

Public knobs are intentionally small: preset, data, tokenizer, token budget,
micro-batch, accumulation, and run id. The rest is a fixed recipe.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RUNS_DIR = "runs"
DEFAULT_TOTAL_TOKENS = 250_000_000


class StreamingTextDataset(IterableDataset):
    """Streams packed token sequences from UTF-8 text files."""

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
                if len(buffer_ids) + len(sent_ids) > self.max_length and buffer_ids:
                    yield self._sample(buffer_ids)
                    buffer_ids = []
                buffer_ids.extend(sent_ids)
        if buffer_ids:
            yield self._sample(buffer_ids)

    @staticmethod
    def _sample(ids: List[int]) -> dict:
        T = len(ids)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(T, dtype=torch.long),
        }

    def _get_worker_paths(self) -> List[Path]:
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
    T = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, T, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_corpus_paths(dirs: List[str]) -> List[Path]:
    paths: List[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        entry = Path(d)
        if entry.is_file() and entry.suffix.lower() == ".txt":
            found = [entry]
        elif entry.is_dir():
            found = sorted(entry.rglob("*.txt"))
        else:
            log.warning("Skipping missing/non-text path: %s", d)
            continue
        for p in found:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No .txt corpus files found in: {dirs}")
    total_mb = sum(p.stat().st_size for p in paths) / 1e6
    log.info("Corpus: %d files, %.1f MB", len(paths), total_mb)
    return paths


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def ramp(progress: float, start: float, width: float) -> float:
    return min(1.0, max(0.0, (progress - start) / max(width, 1e-8)))


def structure_scales(config, progress: float, entropy_frac: Optional[float]) -> dict:
    cluster = ramp(progress, config.cluster_start_ratio, config.cluster_ramp_ratio)
    ortho = cluster
    lang = ramp(progress, config.lang_start_ratio, config.lang_ramp_ratio)
    if entropy_frac is None or entropy_frac < config.lang_min_entropy:
        lang = 0.0
    return {"cluster_scale": cluster, "ortho_scale": ortho, "lang_bias_scale": lang}


def save_checkpoint(
    run_dir: Path,
    step: int,
    tokens_seen: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config,
    metrics: dict,
) -> Path:
    ckpt_dir = run_dir / f"step-{step:07d}"
    tmp_dir = run_dir / f".step-{step:07d}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    torch.save(model.state_dict(), tmp_dir / "model.pt")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
    config.save(tmp_dir / "config.json")
    metrics = {**metrics, "step": step, "tokens_seen": tokens_seen}
    (tmp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    os.replace(tmp_dir, ckpt_dir)

    latest_tmp = run_dir / "latest.tmp"
    latest_tmp.write_text(str(ckpt_dir.resolve()), encoding="utf-8")
    os.replace(latest_tmp, run_dir / "latest")
    log.info("Checkpoint saved: %s", ckpt_dir)
    return ckpt_dir


def load_latest_checkpoint(
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, int, Optional[Path]]:
    latest_file = run_dir / "latest"
    if not latest_file.exists():
        return 0, 0, None
    ckpt_dir = Path(latest_file.read_text(encoding="utf-8").strip())
    if not ckpt_dir.exists():
        log.warning("Latest checkpoint is missing: %s", ckpt_dir)
        return 0, 0, None
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True))
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=True))
    scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", map_location=device, weights_only=True))
    metrics = json.loads((ckpt_dir / "metrics.json").read_text(encoding="utf-8"))
    step = int(metrics.get("step", 0))
    tokens_seen = int(metrics.get("tokens_seen", 0))
    log.info("Resumed from %s at step %d, %d tokens", ckpt_dir, step, tokens_seen)
    return step, tokens_seen, ckpt_dir


def prototype_entropy(language_probs: torch.Tensor, real_mask: torch.Tensor, k: int) -> float:
    with torch.no_grad():
        dominant = language_probs.argmax(-1)[real_mask]
        if dominant.numel() == 0:
            return 0.0
        counts = torch.bincount(dominant, minlength=k).float()
        usage = counts / counts.sum().clamp(min=1.0)
        entropy = -(usage * (usage + 1e-9).log()).sum()
        return (entropy / math.log(k)).item()


def train(
    preset: str = "darija-medium",
    corpus_dirs: Optional[List[str]] = None,
    tokenizer_dir: str = "runs/tokenizer",
    total_tokens: int = DEFAULT_TOTAL_TOKENS,
    micro_batch_size: int = 16,
    grad_accum: int = 4,
    run_id: str = "darija-run-001",
) -> None:
    from sberta.config import SBERTaConfig
    from sberta.model import SBERTaForPreTraining
    from sberta.tokenizer import SBERTaTokenizer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(1337)

    factories = {
        "darija-small": SBERTaConfig.darija_small,
        "darija-medium": SBERTaConfig.darija_medium,
        "darija-base": SBERTaConfig.darija_base,
    }
    if preset not in factories:
        raise ValueError(f"Unknown preset {preset}. Choose one of {list(factories)}.")
    config = factories[preset]()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s (%.1f GB)", props.name, props.total_memory / 1e9)

    tokenizer = SBERTaTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size}; expected {config.vocab_size}."
        )

    corpus_paths = resolve_corpus_paths(corpus_dirs or ["corpus"])
    dataset = StreamingTextDataset(corpus_paths, tokenizer, config.max_length)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(collate_fn, pad_id=tokenizer.PAD_ID),
        prefetch_factor=4 if config.num_workers > 0 else None,
        persistent_workers=(config.num_workers > 0),
    )

    model = SBERTaForPreTraining(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Preset %s | %.1fM params | K=%d | V=%d",
        preset, n_params / 1e6, config.num_languages, config.vocab_size,
    )

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

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    est_tokens_per_step = micro_batch_size * grad_accum * config.max_length
    total_steps = max(1, math.ceil(total_tokens / est_tokens_per_step))
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    run_dir = Path(RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    global_step, tokens_seen, _ = load_latest_checkpoint(
        run_dir, model, optimizer, scheduler, device
    )

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device=device.type, enabled=(amp_enabled and amp_dtype is torch.float16))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    last_entropy: Optional[float] = None
    last_metrics: dict = {"step": global_step, "tokens_seen": tokens_seen}
    next_checkpoint = ((tokens_seen // config.checkpoint_every_tokens) + 1) * config.checkpoint_every_tokens

    loss_keys = ("loss", "loss_gen", "loss_rtd", "loss_cluster", "loss_ortho")
    acc: Dict[str, float] = {k: 0.0 for k in loss_keys}
    metric_sums: Dict[str, float] = {
        "rtd_acc": 0.0,
        "rtd_precision": 0.0,
        "rtd_recall": 0.0,
        "rtd_f1": 0.0,
        "replaced_rate": 0.0,
        "cluster_scale": 0.0,
        "lang_bias_scale": 0.0,
    }
    window_tokens = 0
    n_masked_acc = 0
    sw_sum = 0.0
    sw_max = 0.0
    sw_n = 0
    consecutive_skips = 0
    t0 = time.perf_counter()

    log.info(
        "Training to %d tokens (estimated %d steps, effective batch=%d)",
        total_tokens, total_steps, micro_batch_size * grad_accum,
    )

    while tokens_seen < total_tokens:
        step_acc = {k: 0.0 for k in loss_keys}
        step_metric = {k: 0.0 for k in metric_sums}
        step_tokens = 0
        step_masked = 0
        entropy_values: List[float] = []

        progress = min(1.0, tokens_seen / max(total_tokens, 1))
        scales = structure_scales(config, progress, last_entropy)

        for _ in range(grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            real_tokens = int(attention_mask.sum().item())

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(input_ids=input_ids, attention_mask=attention_mask, **scales)
                loss_scaled = out["loss"] / grad_accum

            scaler.scale(loss_scaled).backward()
            for k in loss_keys:
                val = out[k].item() if hasattr(out[k], "item") else out[k]
                step_acc[k] += val / grad_accum
            for k in step_metric:
                step_metric[k] += float(out[k]) / grad_accum
            step_masked += int(out["n_masked"])
            step_tokens += real_tokens

            real_mask = attention_mask.bool()
            entropy_values.append(
                prototype_entropy(out["language_probs"], real_mask, config.num_languages)
            )
            sw = out["switch_magnitudes"][real_mask]
            sw_sum += sw.sum().item()
            sw_max = max(sw_max, sw.max().item() if sw.numel() else 0.0)
            sw_n += sw.numel()

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if scaler.get_scale() < scale_before:
            consecutive_skips += 1
            log.warning("AMP skipped step %d [consecutive=%d]", global_step, consecutive_skips)
        else:
            consecutive_skips = 0
            scheduler.step()

        global_step += 1
        tokens_seen += step_tokens
        window_tokens += step_tokens
        last_entropy = sum(entropy_values) / max(len(entropy_values), 1)

        for k in loss_keys:
            acc[k] += step_acc[k]
        for k in metric_sums:
            metric_sums[k] += step_metric[k]
        n_masked_acc += step_masked

        should_log = global_step % config.log_every == 0 or tokens_seen >= total_tokens
        if should_log:
            elapsed = max(time.perf_counter() - t0, 1e-9)
            denom = config.log_every if global_step % config.log_every == 0 else max(1, global_step % config.log_every)
            avg = {k: acc[k] / denom for k in loss_keys}
            mavg = {k: metric_sums[k] / denom for k in metric_sums}
            lr_now = scheduler.get_last_lr()[0]
            ppl = math.exp(min(avg["loss_gen"], 20.0))
            prior = " ".join(f"p{i}={v:.3f}" for i, v in enumerate(model._prototype_prior.tolist()))
            sw_mean = sw_sum / sw_n if sw_n else 0.0
            log.info(
                "step %d | %.1f%% | loss %.4f [gen %.4f rtd %.4f cl %.4f ort %.4f] "
                "ppl %.1f lr %.2e grad %.3f tok/s %.0f",
                global_step,
                100.0 * tokens_seen / max(total_tokens, 1),
                avg["loss"], avg["loss_gen"], avg["loss_rtd"],
                avg["loss_cluster"], avg["loss_ortho"],
                ppl, lr_now, grad_norm, window_tokens / elapsed,
            )
            log.info(
                "  rtd acc %.1f p %.1f r %.1f f1 %.1f repl %.1f | proto entropy %.1f%% | scales c=%.2f l=%.2f",
                mavg["rtd_acc"] * 100.0,
                mavg["rtd_precision"] * 100.0,
                mavg["rtd_recall"] * 100.0,
                mavg["rtd_f1"] * 100.0,
                mavg["replaced_rate"] * 100.0,
                (last_entropy or 0.0) * 100.0,
                mavg["cluster_scale"],
                mavg["lang_bias_scale"],
            )
            log.info("  prior %s | switch mean %.4f max %.4f | masked/step %.0f", prior, sw_mean, sw_max, n_masked_acc / denom)
            last_metrics = {
                "loss": avg["loss"],
                "loss_gen": avg["loss_gen"],
                "loss_rtd": avg["loss_rtd"],
                "loss_cluster": avg["loss_cluster"],
                "loss_ortho": avg["loss_ortho"],
                "rtd_acc": mavg["rtd_acc"],
                "rtd_precision": mavg["rtd_precision"],
                "rtd_recall": mavg["rtd_recall"],
                "rtd_f1": mavg["rtd_f1"],
                "replaced_rate": mavg["replaced_rate"],
                "proto_entropy": last_entropy,
                "lr": lr_now,
            }
            acc = {k: 0.0 for k in loss_keys}
            metric_sums = {k: 0.0 for k in metric_sums}
            window_tokens = 0
            n_masked_acc = 0
            sw_sum = sw_max = 0.0
            sw_n = 0
            t0 = time.perf_counter()

        if tokens_seen >= next_checkpoint or tokens_seen >= total_tokens:
            save_checkpoint(
                run_dir,
                global_step,
                tokens_seen,
                model,
                optimizer,
                scheduler,
                config,
                last_metrics,
            )
            next_checkpoint += config.checkpoint_every_tokens

    log.info("Training complete: %d steps, %d tokens.", global_step, tokens_seen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SBERTa Darija pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset", default="darija-medium", choices=["darija-small", "darija-medium", "darija-base"])
    p.add_argument("--corpus-dirs", nargs="+", default=["corpus"])
    p.add_argument("--tokenizer-dir", default="runs/tokenizer")
    p.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--run-id", default="darija-run-001")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        preset=args.preset,
        corpus_dirs=args.corpus_dirs,
        tokenizer_dir=args.tokenizer_dir,
        total_tokens=args.total_tokens,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        run_id=args.run_id,
    )
