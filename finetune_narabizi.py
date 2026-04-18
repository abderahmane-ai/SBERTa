"""
finetune_narabizi.py — Fine-tune SBERTa on the NArabizi dataset.

Tasks
-----
sentiment : 3-way classification — NEG / NEU / POS
topic     : 4-way classification — Religion / Societal / Sport / NONE

Dataset
-------
Touileb & Barnes, ACL 2021 Findings.
GitHub: https://github.com/SamiaTouileb/NArabizi

~1,500 sentences of Algerian Arabizi (Latin-script code-switched Arabic).
Pre-defined train / dev / test splits (inherited from NArabizi treebank).

Baseline to beat
----------------
DziriBERT sentiment accuracy: 80.5%  (Abdaoui et al., 2021)
mBERT sentiment accuracy:     ~72%

Usage
-----
python finetune_narabizi.py \
    --task sentiment \
    --data-dir NArabizi/data/Narabizi \
    --pretrained-dir runs/sberta-base-100k/step-0036900 \
    --tokenizer-dir runs/tokenizer \
    --run-id narabizi-sentiment-001

python finetune_narabizi.py \
    --task topic \
    --data-dir NArabizi/data/Narabizi \
    --pretrained-dir runs/sberta-base-100k/step-0036900 \
    --tokenizer-dir runs/tokenizer \
    --run-id narabizi-topic-001

Data preparation
----------------
Clone https://github.com/SamiaTouileb/NArabizi into --data-dir.

Expected structure:
    data/narabizi/
        pos/
            train_NArabizi.conllu
            dev_NArabizi.conllu
            test_NArabizi.conllu
        sentiment/
            train_Narabizi_sentiment.txt
            dev_Narabizi_sentiment.txt
            test_Narabizi_sentiment.txt
        topic/
            train_Narabizi_topic.txt
            dev_Narabizi_topic.txt
            test_Narabizi_topic.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Label maps ───────────────────────────────────────────────────────────────

LABEL_MAPS: Dict[str, Dict[str, int]] = {
    "sentiment": {"NEG": 0, "NEU": 1, "POS": 2, "MIX": 3},
    "topic":     {"Religion": 0, "Societal": 1, "Sport": 2, "NONE": 3},
}


# ─── Dataset ──────────────────────────────────────────────────────────────────

def _read_conllu_sentences(conllu_path: Path) -> Dict[str, str]:
    """Parse a CoNLL-U file and return {sent_id: sentence_text} mapping.
    
    NArabizi CoNLL-U files use # sent_id = <id> and # text = <text> comments.
    Falls back to joining token forms if no text comment is present.
    """
    sentences: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_text: Optional[str] = None
    current_tokens: List[str] = []
    
    with open(conllu_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            
            if line.startswith("# sent_id"):
                current_id = line.split("=", 1)[1].strip()
                current_text = None
                current_tokens = []
            elif line.startswith("# text"):
                current_text = line.split("=", 1)[1].strip()
            elif line and not line.startswith("#"):
                parts = line.split("\t")
                # Skip multi-word tokens (e.g. "1-2") and empty nodes
                if parts[0].isdigit():
                    current_tokens.append(parts[1])
            elif line == "" and current_id is not None:
                text = current_text or " ".join(current_tokens)
                sentences[current_id] = text
                current_id = None
    
    # Flush last sentence if file doesn't end with blank line
    if current_id is not None:
        sentences[current_id] = current_text or " ".join(current_tokens)
    
    return sentences


def _read_annotation_tsv(tsv_path: Path) -> Dict[str, str]:
    """Parse a NArabizi annotation file.
    
    Format: # sent_id = <id> \\t <label>  (with optional header line)
    The sent_id prefix "# sent_id = " is stripped to match CoNLL-U keys.
    """
    annotations: Dict[str, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            label = parts[1].strip()
            # Skip header line
            if key in ("ID", "SENT"):
                continue
            # Strip "# sent_id = " prefix if present
            if key.startswith("# sent_id ="):
                key = key.split("=", 1)[1].strip()
            annotations[key] = label
    return annotations


class NArabiziDataset(Dataset):
    """NArabizi sentence-level classification dataset.
    
    Reads CoNLL-U sentence texts and matches them to annotation labels.
    Returns tokenized input_ids + attention_mask + integer label.
    """
    
    def __init__(
        self,
        conllu_path: Path,
        annotation_path: Path,
        tokenizer,
        label_map: Dict[str, int],
        max_length: int = 128,
        augment: bool = False,
        aug_prob: float = 0.1,
    ) -> None:
        sentences = _read_conllu_sentences(conllu_path)
        annotations = _read_annotation_tsv(annotation_path)
        
        self.samples: List[Tuple[str, int]] = []
        skipped = 0
        
        for sent_id, label_str in annotations.items():
            if sent_id not in sentences:
                skipped += 1
                continue
            if label_str not in label_map:
                skipped += 1
                continue
            self.samples.append((sentences[sent_id], label_map[label_str]))
        
        if skipped:
            log.warning(
                "%s: skipped %d samples (missing sentence or unknown label)",
                annotation_path.name, skipped,
            )
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.aug_prob = aug_prob
        
        log.info(
            "%s: %d samples, labels=%s",
            annotation_path.name, len(self.samples),
            {v: k for k, v in label_map.items()},
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        text, label = self.samples[idx]
        
        # Tokenize
        ids = self.tokenizer.encode(
            text,
            add_sep=True,
            max_length=self.max_length,
        )
        
        return {
            "input_ids":      torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def _collate(batch: List[dict], pad_id: int) -> dict:
    T = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)
    
    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, T, dtype=torch.long)
    labels         = torch.stack([item["label"] for item in batch])
    
    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n]      = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
    
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


# ─── Model ────────────────────────────────────────────────────────────────────

class SBERTaForSequenceClassification(nn.Module):
    """SBERTa fine-tuning head for sentence-level classification.
    
    Pooling strategy: masked mean-pool over all real token positions.
    SBERTa has no [CLS] token by design — mean pooling is the canonical
    approach and matches how sentence embeddings are extracted at inference.
    
    Architecture:
        encoder → mean_pool → dropout → linear(d, num_labels) → logits
    
    The dropout rate is slightly higher than pre-training (0.1 → 0.1–0.3)
    to compensate for the small NArabizi dataset (~1k training sentences).
    """
    
    def __init__(self, config, num_labels: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        from sberta.model import SBERTaModel
        
        self.encoder: SBERTaModel = SBERTaModel(config)
        self.dropout: nn.Dropout = nn.Dropout(dropout_prob)
        self.classifier: nn.Linear = nn.Linear(config.hidden_size, num_labels)
        
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids:      (B, T)
            attention_mask: (B, T) binary
            labels:         (B,) integer class indices — optional
        
        Returns:
            dict with 'logits' (B, num_labels) and optionally 'loss' (scalar)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        H, _p0, _s = self.encoder(input_ids, attention_mask)  # (B, T, d)
        
        # Masked mean pool over real tokens
        mask_f = attention_mask.float().unsqueeze(-1)          # (B, T, 1)
        pooled = (H * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # (B, d)
        
        logits = self.classifier(self.dropout(pooled))         # (B, num_labels)
        
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        
        return out
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_dir: str | Path,
        num_labels: int,
        dropout_prob: float = 0.1,
    ) -> "SBERTaForSequenceClassification":
        """Load pre-trained SBERTa encoder weights into the classification model.
        
        Only encoder weights are loaded — the classification head is randomly
        initialised (as it was not pre-trained).
        """
        from sberta.config import SBERTaConfig
        
        pretrained_dir = Path(pretrained_dir)
        config = SBERTaConfig.load(pretrained_dir / "config.json")
        model = cls(config, num_labels=num_labels, dropout_prob=dropout_prob)
        
        # Load pre-training checkpoint — extract encoder weights only
        pt_state = torch.load(
            pretrained_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        
        # Pre-training state_dict keys are under "sberta.*"
        # Map them to "encoder.*" for this model
        encoder_state: dict = {}
        for k, v in pt_state.items():
            if k.startswith("sberta."):
                new_key = k.replace("sberta.", "encoder.", 1)
                encoder_state[new_key] = v
        
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        
        # Only the classifier head should be missing — everything else is an error
        classifier_keys = {"classifier.weight", "classifier.bias", "dropout.p"}
        truly_missing = [k for k in missing if k not in classifier_keys]
        
        if truly_missing:
            log.warning("Unexpected missing keys: %s", truly_missing)
        if unexpected:
            log.warning("Unexpected keys in checkpoint: %s", unexpected)
        
        log.info(
            "Loaded pre-trained encoder from %s (%d params transferred)",
            pretrained_dir,
            sum(v.numel() for k, v in encoder_state.items()),
        )
        
        return model


# ─── Training utilities ───────────────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Accuracy and per-class F1 (macro)."""
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    
    n_classes = logits.size(-1)
    tp = torch.zeros(n_classes)
    fp = torch.zeros(n_classes)
    fn = torch.zeros(n_classes)
    
    for c in range(n_classes):
        tp[c] = ((preds == c) & (labels == c)).sum().float()
        fp[c] = ((preds == c) & (labels != c)).sum().float()
        fn[c] = ((preds != c) & (labels == c)).sum().float()
    
    precision = tp / (tp + fp).clamp(min=1e-9)
    recall    = tp / (tp + fn).clamp(min=1e-9)
    f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    macro_f1  = f1.mean().item()
    
    return {"acc": acc, "macro_f1": macro_f1, "per_class_f1": f1.tolist()}


@torch.no_grad()
def evaluate(
    model: SBERTaForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]
        
        out = model(input_ids, attention_mask)
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels)
    
    model.train()
    
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss   = F.cross_entropy(logits, labels).item()
    
    metrics = compute_metrics(logits, labels)
    metrics["loss"] = loss
    
    return metrics


# ─── Fine-tuning loop ─────────────────────────────────────────────────────────

def finetune(
    task: str,
    data_dir: str,
    pretrained_dir: str,
    tokenizer_dir: str,
    run_id: str = "narabizi-001",
    runs_dir: str = "runs",
    # Optimisation
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    dropout_prob: float = 0.2,
    # Data
    max_length: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> None:
    from sberta.tokenizer import SBERTaTokenizer
    
    torch.manual_seed(seed)
    random.seed(seed)
    
    assert task in LABEL_MAPS, f"Unknown task '{task}'. Choose from: {list(LABEL_MAPS)}"
    label_map  = LABEL_MAPS[task]
    num_labels = len(label_map)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    
    data_dir = Path(data_dir)
    run_dir  = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = SBERTaTokenizer.from_pretrained(tokenizer_dir)
    
    # ── Datasets ──────────────────────────────────────────────────────────
    # NArabizi structure: pos/*.conllu, sentiment/*.txt, topic/*.txt
    pos_dir = data_dir / "pos"
    task_dir = data_dir / task
    
    train_ds = NArabiziDataset(
        pos_dir / "train_NArabizi.conllu",
        task_dir / f"train_Narabizi_{task}.txt",
        tokenizer, label_map, max_length, augment=False,
    )
    dev_ds = NArabiziDataset(
        pos_dir / "dev_NArabizi.conllu",
        task_dir / f"dev_Narabizi_{task}.txt",
        tokenizer, label_map, max_length, augment=False,
    )
    test_ds = NArabiziDataset(
        pos_dir / "test_NArabizi.conllu",
        task_dir / f"test_Narabizi_{task}.txt",
        tokenizer, label_map, max_length, augment=False,
    )
    
    collate = lambda b: _collate(b, pad_id=tokenizer.PAD_ID)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate,
    )
    dev_loader  = DataLoader(
        dev_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )
    
    # ── Model ─────────────────────────────────────────────────────────────
    model = SBERTaForSequenceClassification.from_pretrained(
        pretrained_dir, num_labels=num_labels, dropout_prob=dropout_prob,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %.1f M", n_params / 1e6)
    log.info(
        "Task: %s | Labels: %s | Train: %d | Dev: %d | Test: %d",
        task, label_map, len(train_ds), len(dev_ds), len(test_ds),
    )
    
    # ── Optimiser — frozen encoder, head only ─────────────────────────────
    # Encoder is frozen; only the classification head is trained.
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    
    head_params = list(model.classifier.parameters())
    
    optimizer = torch.optim.AdamW(
        head_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ── Training ──────────────────────────────────────────────────────────
    best_dev_acc  = 0.0
    best_dev_f1   = 0.0
    best_step     = 0
    global_step   = 0
    history: List[dict] = []
    
    log.info(
        "Fine-tuning: %d epochs × %d batches = %d steps  "
        "(warmup=%d, LR=%.1e, encoder=frozen)",
        epochs, len(train_loader), total_steps,
        warmup_steps, learning_rate,
    )
    
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        epoch_loss = 0.0
        model.train()
        
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            
            out  = model(input_ids, attention_mask, labels=labels)
            loss = out["loss"]
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss  += loss.item()
            global_step += 1
        
        epoch_loss /= len(train_loader)
        dev_metrics = evaluate(model, dev_loader, device)
        elapsed     = time.perf_counter() - t0
        
        log.info(
            "Epoch %2d/%d  train_loss=%.4f  "
            "dev_loss=%.4f  dev_acc=%.1f%%  dev_F1=%.4f  (%.1fs)",
            epoch, epochs, epoch_loss,
            dev_metrics["loss"],
            dev_metrics["acc"] * 100.0,
            dev_metrics["macro_f1"],
            elapsed,
        )
        
        history.append({
            "epoch":      epoch,
            "train_loss": epoch_loss,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        })
        
        # Save best checkpoint by accuracy (primary) then F1 (tiebreak)
        is_best = (
            dev_metrics["acc"] > best_dev_acc
            or (dev_metrics["acc"] == best_dev_acc and dev_metrics["macro_f1"] > best_dev_f1)
        )
        
        if is_best:
            best_dev_acc  = dev_metrics["acc"]
            best_dev_f1   = dev_metrics["macro_f1"]
            best_step     = global_step
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            log.info(
                "  ✓ New best — saved (acc=%.1f%%  F1=%.4f)",
                best_dev_acc * 100.0, best_dev_f1
            )
    
    # ── Test evaluation — load best checkpoint ────────────────────────────
    model.load_state_dict(
        torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, device)
    
    log.info("=" * 60)
    log.info("TEST RESULTS (best dev checkpoint, step %d)", best_step)
    log.info("  Accuracy  : %.2f%%", test_metrics["acc"] * 100.0)
    log.info("  Macro F1  : %.4f",   test_metrics["macro_f1"])
    log.info("  Per-class F1: %s", {
        k: f"{v:.4f}" for k, v in
        zip(LABEL_MAPS[task].keys(), test_metrics["per_class_f1"])
    })
    log.info("  DziriBERT baseline (sentiment acc): 80.5%%")
    log.info("=" * 60)
    
    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "task":          task,
        "run_id":        run_id,
        "pretrained_dir": str(pretrained_dir),
        "best_dev_acc":  best_dev_acc,
        "best_dev_f1":   best_dev_f1,
        "best_step":     best_step,
        "test":          test_metrics,
        "history":       history,
        "hparams": {
            "epochs":        epochs,
            "batch_size":    batch_size,
            "learning_rate": learning_rate,
            "dropout_prob":  dropout_prob,
            "max_length":    max_length,
            "seed":          seed,
        },
    }
    
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Results saved → %s", run_dir / "results.json")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune SBERTa on NArabizi sentiment/topic classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    p.add_argument("--task",           required=True, choices=["sentiment", "topic"])
    p.add_argument("--data-dir",       required=True, help="Path to NArabizi repo clone")
    p.add_argument("--pretrained-dir", required=True, help="SBERTa checkpoint directory")
    p.add_argument("--tokenizer-dir",  default="runs/tokenizer")
    p.add_argument("--run-id",         default="narabizi-001")
    p.add_argument("--runs-dir",       default="runs")
    
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--batch-size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--weight-decay",   type=float, default=0.01)
    p.add_argument("--warmup-ratio",   type=float, default=0.1)
    p.add_argument("--max-grad-norm",  type=float, default=1.0)
    p.add_argument("--dropout",        type=float, default=0.2)
    p.add_argument("--max-length",     type=int,   default=128)
    p.add_argument("--num-workers",    type=int,   default=0)
    p.add_argument("--seed",           type=int,   default=42)
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune(
        task=args.task,
        data_dir=args.data_dir,
        pretrained_dir=args.pretrained_dir,
        tokenizer_dir=args.tokenizer_dir,
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        dropout_prob=args.dropout,
        max_length=args.max_length,
        num_workers=args.num_workers,
        seed=args.seed,
    )
