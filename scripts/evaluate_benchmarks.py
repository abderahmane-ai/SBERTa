"""Fine-tune DziriBERT or SBERTa on a Darija classification benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def label_map(rows: list[dict[str, str]], label_col: str) -> dict[str, int]:
    return {label: i for i, label in enumerate(sorted({str(r[label_col]) for r in rows}))}


def macro_f1(preds: list[int], labels: list[int], n_labels: int) -> float:
    scores = []
    for label in range(n_labels):
        tp = sum(p == label and y == label for p, y in zip(preds, labels))
        fp = sum(p == label and y != label for p, y in zip(preds, labels))
        fn = sum(p != label and y == label for p, y in zip(preds, labels))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-8))
    return sum(scores) / max(len(scores), 1)


class EncodedRows(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        text_col: str,
        label_col: str,
        labels: dict[str, int],
        encode,
    ) -> None:
        self.items = []
        for row in rows:
            encoded = encode(row[text_col])
            self.items.append((encoded["input_ids"], encoded["attention_mask"], labels[str(row[label_col])]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ids, mask, label = self.items[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    T = max(x["input_ids"].numel() for x in batch)
    B = len(batch)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.long)
    labels = torch.stack([x["labels"] for x in batch])
    for i, item in enumerate(batch):
        n = item["input_ids"].numel()
        ids[i, :n] = item["input_ids"]
        mask[i, :n] = item["attention_mask"]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class SBERTaClassifier(nn.Module):
    def __init__(self, checkpoint: Path, n_labels: int) -> None:
        super().__init__()
        from sberta.config import SBERTaConfig
        from sberta.model import SBERTaForPreTraining

        if checkpoint.is_file() and checkpoint.name == "latest":
            checkpoint = Path(checkpoint.read_text(encoding="utf-8").strip())
        config = SBERTaConfig.load(checkpoint / "config.json")
        pretrain = SBERTaForPreTraining(config)
        state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
        pretrain.load_state_dict(state, strict=False)
        self.encoder = pretrain.get_encoder()
        self.head = nn.Linear(config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        h, _, _, _ = self.encoder(input_ids, attention_mask, lang_bias_scale=1.0)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        logits = self.head(pooled)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def build_model(args: argparse.Namespace, n_labels: int):
    if args.model == "dziribert":
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained("alger-ia/dziribert", num_labels=n_labels)
    return SBERTaClassifier(Path(args.sberta_checkpoint), n_labels)


def build_encoder(args: argparse.Namespace):
    if args.model == "dziribert":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("alger-ia/dziribert")
        return tok.pad_token_id, lambda text: tok(text, truncation=True, max_length=512)

    from sberta.tokenizer import SBERTaTokenizer
    tok = SBERTaTokenizer.from_pretrained(args.tokenizer_dir)

    def encode(text: str) -> dict[str, Any]:
        ids = tok.encode(text, max_length=512)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    return tok.PAD_ID, encode


def evaluate(model, loader, device, n_labels: int) -> dict[str, float]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds.extend(out["logits"].argmax(-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    acc = sum(p == y for p, y in zip(preds, labels)) / max(len(labels), 1)
    return {"accuracy": acc, "macro_f1": macro_f1(preds, labels, n_labels)}


def main() -> None:
    args = parse_args()
    train_rows = read_rows(Path(args.train))
    dev_rows = read_rows(Path(args.dev))
    test_rows = read_rows(Path(args.test))
    labels = label_map(train_rows + dev_rows + test_rows, args.label_col)
    pad_id, encode = build_encoder(args)
    train_data = EncodedRows(train_rows, args.text_col, args.label_col, labels, encode)
    dev_data = EncodedRows(dev_rows, args.text_col, args.label_col, labels, encode)
    test_data = EncodedRows(test_rows, args.text_col, args.label_col, labels, encode)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad_id))
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, collate_fn=lambda b: collate(b, pad_id))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=lambda b: collate(b, pad_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_state = None
    best_f1 = -1.0

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        dev_metrics = evaluate(model, dev_loader, device, len(labels))
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    result = {
        "task": args.task,
        "model": args.model,
        "labels": labels,
        "dev": evaluate(model, dev_loader, device, len(labels)),
        "test": evaluate(model, test_loader, device, len(labels)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a DziriBERT/SBERTa benchmark.")
    p.add_argument("--task", required=True)
    p.add_argument("--model", choices=["dziribert", "sberta"], required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--tokenizer-dir", default="runs/tokenizer")
    p.add_argument("--sberta-checkpoint", default="runs/darija-run-001/latest")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output", default="runs/benchmarks/result.json")
    return p.parse_args()


if __name__ == "__main__":
    main()
