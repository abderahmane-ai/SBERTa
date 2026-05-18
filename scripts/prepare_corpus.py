"""Prepare Darija pretraining text and a JSONL manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from sberta.tokenizer import normalise

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIZI_RE = re.compile(r"[A-Za-z]*[23456789][A-Za-z0-9]*|[A-Za-z0-9]*[23456789][A-Za-z]*")
LINGUISTIC_DIGITS = set("23456789")


def iter_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend(sorted(p.rglob("*.txt")))
        else:
            paths.extend(sorted(Path().glob(item)))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        rp = path.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(path)
    return unique


def spam_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    valid = sum(1 for c in chars if c.isalpha() or c in LINGUISTIC_DIGITS)
    return 1.0 - valid / len(chars)


def script_stats(text: str) -> dict:
    chars = max(1, sum(1 for c in text if not c.isspace()))
    arabic = len(ARABIC_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    arabizi = len(ARABIZI_RE.findall(text))
    return {
        "arabic_ratio": arabic / chars,
        "latin_ratio": latin / chars,
        "arabizi_token_count": arabizi,
    }


def dedup_key(text: str) -> str:
    folded = re.sub(r"[^\w\s]", "", text.lower())
    folded = " ".join(folded.split())
    return hashlib.sha1(folded.encode("utf-8")).hexdigest()


def prepare(
    inputs: list[str],
    output: Path,
    manifest: Path,
    source_name: str,
    license_status: str,
    usage: str,
    min_tokens: int = 3,
    max_spam_ratio: float = 0.35,
) -> None:
    paths = iter_paths(inputs)
    if not paths:
        raise FileNotFoundError("No input .txt files found.")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = kept = duplicates = filtered = 0
    arabic_sum = latin_sum = arabizi_tokens = 0.0

    with output.open("w", encoding="utf-8") as out:
        for path in paths:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    total += 1
                    line = normalise(raw.strip())
                    if len(line.split()) < min_tokens or spam_ratio(line) > max_spam_ratio:
                        filtered += 1
                        continue
                    key = dedup_key(line)
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    stats = script_stats(line)
                    arabic_sum += stats["arabic_ratio"]
                    latin_sum += stats["latin_ratio"]
                    arabizi_tokens += stats["arabizi_token_count"]
                    kept += 1
                    out.write(line + "\n")

    output_bytes = output.stat().st_size
    record = {
        "source": source_name,
        "paths": [str(p) for p in paths],
        "output": str(output),
        "bytes": output_bytes,
        "license_status": license_status,
        "usage": usage,
        "total_lines": total,
        "kept_lines": kept,
        "filtered_lines": filtered,
        "duplicate_lines": duplicates,
        "arabic_ratio_mean": arabic_sum / max(kept, 1),
        "latin_ratio_mean": latin_sum / max(kept, 1),
        "arabizi_token_count": int(arabizi_tokens),
    }
    with manifest.open("a", encoding="utf-8") as mf:
        mf.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Darija corpus text with provenance.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", default="corpus/darija_pretrain.txt")
    p.add_argument("--manifest", default="data/manifest.jsonl")
    p.add_argument("--source-name", required=True)
    p.add_argument("--license-status", required=True)
    p.add_argument(
        "--usage",
        choices=["pretraining", "evaluation", "research-only", "restricted"],
        default="pretraining",
    )
    p.add_argument("--min-tokens", type=int, default=3)
    p.add_argument("--max-spam-ratio", type=float, default=0.35)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(
        inputs=args.input,
        output=Path(args.output),
        manifest=Path(args.manifest),
        source_name=args.source_name,
        license_status=args.license_status,
        usage=args.usage,
        min_tokens=args.min_tokens,
        max_spam_ratio=args.max_spam_ratio,
    )
