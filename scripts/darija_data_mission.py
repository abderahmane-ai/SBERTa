"""Autonomous Algerian Darija corpus acquisition, cleaning, and reporting."""
from __future__ import annotations

import argparse
import collections
import csv
import html
import io
import json
import math
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
import tiktoken
from datasets import Dataset, DatasetDict, load_dataset

from sberta.tokenizer import normalise

TARGET_TOKENS = 25_000_000
ROOT = Path(".")
RAW_DIR = ROOT / "raw_data"
CLEAN_DIR = ROOT / "cleaned_data"
FINAL_DIR = ROOT / "FINAL_DARIJA_CORPUS"
REPORT_DIR = ROOT / "reports"
LOG_PATH = ROOT / "mission_log.txt"
MANIFEST_PATH = ROOT / "data" / "manifest.jsonl"
SPOT_CHECK_PATH = REPORT_DIR / "spot_check_samples.md"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCT_RUN_RE = re.compile(r"([!?؟.,،؛:])\1{2,}")
SPACE_RE = re.compile(r"\s+")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIZI_RE = re.compile(
    r"\b(?:[A-Za-z]{2,}[235679][A-Za-z0-9]*|[A-Za-z0-9]*[235679][A-Za-z]{2,})\b"
)

ALGERIAN_MARKERS = {
    "واش", "راك", "راكي", "راني", "راه", "راهو", "راهي", "حنا", "نتوما",
    "بزاف", "صحا", "خويا", "اختي", "كاين", "ماكانش", "علابالي", "علاش",
    "شكون", "برك", "هكا", "كيما", "تاع", "نتاع", "بصح", "معليش", "ولاه",
    "wach", "wesh", "rak", "raki", "rani", "rah", "raho", "rahi", "hna",
    "bezaf", "bezzaf", "saha", "khouya", "khoya", "khti", "kayn", "kayen",
    "makach", "3labali", "3lach", "chkon", "chkoun", "bark", "haka",
    "kima", "ta3", "nta3", "besah", "bseh", "ma3lich", "wlh", "sah",
}

FORBIDDEN_MARKERS = {
    "عايز", "اوي", "دلوقتي", "ازاي", "كده", "هلق", "كتير", "بدي", "ليش",
    "شلون", "وايد", "مش", "اوى", "جدا", "شو", "مو", "بدك", "هيدا",
    "عاوز", "النهاردة", "هاي", "اكو", "چ", "gulf", "egyptian", "levantine",
}

TEMPLATE_PATTERNS = [
    re.compile(r"\b(subscribe|click here|follow me|like and subscribe)\b", re.I),
    re.compile(r"\b(abonnez|cliquez ici|subscribe to)\b", re.I),
    re.compile(r"(.)\1{8,}"),
]

ENTITY_TEMPLATE_PATTERNS = [
    re.compile(r"\blist of\b", re.I),
    re.compile(r"\bambassadors? of\b", re.I),
    re.compile(r"\bproclamation\s+\d+\b", re.I),
    re.compile(r"\bwikidata\b", re.I),
]

ENGLISH_STOPWORDS = {
    "the", "of", "and", "to", "in", "for", "on", "with", "at", "from",
    "by", "about", "into", "over", "after", "before", "between",
}


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def ensure_dirs() -> None:
    for path in (RAW_DIR, CLEAN_DIR, FINAL_DIR, REPORT_DIR, MANIFEST_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


def load_sources(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def active_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [source for source in sources if not source.get("disabled", False)]


def retry(fn, label: str, tries: int = 4):
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == tries:
                log(f"FAILED {label}: {exc}")
                raise
            log(f"retry {attempt}/{tries} for {label}: {exc}")
            time.sleep(delay)
            delay *= 2


def pick_text(row: dict[str, Any], fields: list[str] | None) -> str | None:
    candidates = fields or []
    candidates += [
        "text", "Text", "sentence", "content", "comment", "comments",
        "utterance", "transcription", "processed_transcription",
        "original_transcription", "tokens",
    ]
    for key in candidates:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            return " ".join(str(x) for x in value)
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) >= 5:
            return value
    return None


def row_passes_filter(row: dict[str, Any], filt: dict[str, list[str]] | None) -> bool:
    if not filt:
        return True
    for key, allowed in filt.items():
        value = str(row.get(key, "")).strip()
        if value and value in allowed:
            continue
        if value and value.lower() in {x.lower() for x in allowed}:
            continue
        return False
    return True


def dataset_to_iter(ds: Dataset | DatasetDict, split: str | None) -> Iterable[dict[str, Any]]:
    if isinstance(ds, DatasetDict):
        if split and split in ds:
            use = ds[split]
        else:
            first = next(iter(ds.keys()))
            use = ds[first]
    else:
        use = ds
    for row in use:
        yield dict(row)


def download_hf_source(source: dict[str, Any], max_rows: int | None, force: bool) -> Path:
    name = source["name"]
    out_dir = RAW_DIR / name
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "raw.jsonl"
    out_txt = out_dir / "raw.txt"
    if out_jsonl.exists() and out_txt.exists():
        log(f"raw exists, skipping download: {name}")
        return out_txt

    dataset_id = source["dataset"]
    dataset_config = source.get("config")
    split = source.get("split")
    log(f"downloading HF dataset {dataset_id} config={dataset_config} split={split}")

    def _load():
        try:
            if dataset_config:
                return load_dataset(dataset_id, dataset_config, split=split)
            return load_dataset(dataset_id, split=split)
        except Exception:
            if dataset_config:
                return load_dataset(dataset_id, dataset_config)
            return load_dataset(dataset_id)

    ds = retry(_load, f"load_dataset {dataset_id}")
    n = 0
    kept = 0
    with out_jsonl.open("w", encoding="utf-8") as jf, out_txt.open("w", encoding="utf-8") as tf:
        for row in dataset_to_iter(ds, split):
            n += 1
            if max_rows is not None and n > max_rows:
                break
            if not row_passes_filter(row, source.get("filter")):
                continue
            text = pick_text(row, source.get("text_fields"))
            if not text:
                continue
            rec = {"source": name, "text": text, "raw": row}
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tf.write(text.replace("\n", " ") + "\n")
            kept += 1
    log(f"downloaded {name}: rows_seen={n} text_rows={kept}")
    return out_txt


def request_download(url: str, path: Path) -> None:
    def _download() -> None:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

    retry(_download, f"download {url}")


def records_from_frame(frame: pd.DataFrame, source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    fields = source.get("text_fields")
    for row in frame.fillna("").to_dict(orient="records"):
        text = pick_text(row, fields)
        if text:
            yield {"source": source["name"], "text": text, "raw": row}


def records_from_json_bytes(data: bytes, source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    payload = json.loads(data.decode("utf-8"))
    rows: Iterable[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("data", "rows", "examples", "records", "items"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        else:
            rows = payload.values()
    else:
        rows = []
    fields = source.get("text_fields")
    for row in rows:
        if isinstance(row, dict):
            text = pick_text(row, fields)
            raw = row
        else:
            text = str(row)
            raw = {"value": row}
        if text and len(text.strip()) >= 5:
            yield {"source": source["name"], "text": text, "raw": raw}


def records_from_text_bytes(data: bytes, source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for line in data.decode("utf-8", errors="replace").splitlines():
        text = line.strip()
        if len(text) >= 5:
            yield {"source": source["name"], "text": text, "raw": {"text": text}}


def records_from_zip(path: Path, source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    wanted = re.compile(source.get("zip_member_pattern", r"(transcript|pseudo|label|text|csv|json)"), re.I)
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = info.filename
            suffix = Path(name).suffix.lower()
            if info.is_dir() or "__MACOSX" in name or not wanted.search(name):
                continue
            if suffix not in {".csv", ".tsv", ".json", ".jsonl", ".txt"}:
                continue
            data = zf.read(info)
            if suffix in {".csv", ".tsv"}:
                sep = "\t" if suffix == ".tsv" else None
                frame = pd.read_csv(io.BytesIO(data), sep=sep, engine="python")
                yield from records_from_frame(frame, source)
            elif suffix == ".json":
                yield from records_from_json_bytes(data, source)
            elif suffix == ".jsonl":
                for line in data.decode("utf-8", errors="replace").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = pick_text(row, source.get("text_fields")) if isinstance(row, dict) else str(row)
                    if text:
                        yield {"source": source["name"], "text": text, "raw": row}
            else:
                yield from records_from_text_bytes(data, source)


def download_url_source(source: dict[str, Any], max_rows: int | None, force: bool) -> Path:
    name = source["name"]
    out_dir = RAW_DIR / name
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "raw.jsonl"
    out_txt = out_dir / "raw.txt"
    if out_jsonl.exists() and out_txt.exists():
        log(f"raw exists, skipping download: {name}")
        return out_txt

    file_name = source.get("file_name") or Path(source["url"].split("?")[0]).name or "download"
    download_path = out_dir / file_name
    log(f"downloading URL source {name}: {source['url']}")
    request_download(source["url"], download_path)

    fmt = source.get("format") or download_path.suffix.lower().lstrip(".")
    records: Iterable[dict[str, Any]]
    if fmt in {"xlsx", "xls"}:
        records = records_from_frame(pd.read_excel(download_path), source)
    elif fmt in {"csv", "tsv"}:
        sep = "\t" if fmt == "tsv" else None
        records = records_from_frame(pd.read_csv(download_path, sep=sep, engine="python"), source)
    elif fmt == "json":
        records = records_from_json_bytes(download_path.read_bytes(), source)
    elif fmt == "jsonl":
        records = (
            {"source": name, "text": pick_text(json.loads(line), source.get("text_fields")), "raw": json.loads(line)}
            for line in download_path.read_text(encoding="utf-8", errors="replace").splitlines()
        )
    elif fmt == "zip":
        records = records_from_zip(download_path, source)
    else:
        records = records_from_text_bytes(download_path.read_bytes(), source)

    kept = 0
    with out_jsonl.open("w", encoding="utf-8") as jf, out_txt.open("w", encoding="utf-8") as tf:
        for rec in records:
            text = rec.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tf.write(text.replace("\n", " ") + "\n")
            kept += 1
            if max_rows is not None and kept >= max_rows:
                break
    log(f"downloaded {name}: text_rows={kept}")
    return out_txt


def discover_and_download(sources: list[dict[str, Any]], max_rows: int | None, force: bool) -> list[dict[str, Any]]:
    raw_sources: list[dict[str, Any]] = []
    for source in sources:
        source_type = source.get("type")
        if source_type == "huggingface":
            try:
                raw_sources.append({"path": download_hf_source(source, max_rows, force), "source": source})
            except Exception:
                continue
        elif source_type == "url":
            try:
                raw_sources.append({"path": download_url_source(source, max_rows, force), "source": source})
            except Exception as exc:
                log(f"FAILED url source {source['name']}: {exc}")
                continue
        else:
            log(f"manual source registered: {source['name']} -> {source.get('url', 'no url')}")
    return raw_sources


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PUNCT_RUN_RE.sub(r"\1\1", text)
    text = "".join(
        ch if (ch.isalpha() or ch.isdigit() or ch.isspace() or ch in "!?؟.,،؛:'\"-_/") else " "
        for ch in text
    )
    text = SPACE_RE.sub(" ", text).strip()
    return normalise(text)


def script_tag(text: str) -> str:
    arabic = len(ARABIC_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    arabizi = len(ARABIZI_RE.findall(text))
    if latin > arabic and (arabizi > 0 or marker_score(text, ALGERIAN_MARKERS) > 0):
        return "arabizi"
    if arabic > 0:
        return "arabic"
    return "other"


def marker_score(text: str, markers: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for marker in markers if marker in lowered)


def non_linguistic_ratio(text: str, tag: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    valid = 0
    for c in chars:
        if c.isalpha() or c in "23456789":
            valid += 1
        elif tag == "arabizi" and c.isdigit():
            valid += 1
    return 1.0 - valid / len(chars)


def is_template_or_robotic(text: str) -> bool:
    return any(pattern.search(text) for pattern in TEMPLATE_PATTERNS + ENTITY_TEMPLATE_PATTERNS)


def is_english_heavy(text: str, tag: str) -> bool:
    if tag != "arabizi":
        return False
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    if len(words) < 8:
        return False
    stopword_hits = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    return stopword_hits / len(words) > 0.18 and marker_score(text, ALGERIAN_MARKERS) < 2


def is_probably_algerian(text: str, tag: str, trusted_source: bool) -> bool:
    if marker_score(text, FORBIDDEN_MARKERS) > 0:
        return False
    score = marker_score(text, ALGERIAN_MARKERS)
    if score > 0:
        return True
    if tag == "arabizi":
        return len(ARABIZI_RE.findall(text)) > 0
    return trusted_source and len(text.split()) >= 7


def exact_key(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def simhash(text: str) -> int:
    words = exact_key(text).split()
    shingles = [" ".join(words[i:i + 4]) for i in range(max(1, len(words) - 3))]
    vec = [0] * 64
    for shingle in shingles:
        h = hash(shingle) & ((1 << 64) - 1)
        for i in range(64):
            vec[i] += 1 if h & (1 << i) else -1
    out = 0
    for i, value in enumerate(vec):
        if value > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class Deduper:
    def __init__(self, max_hamming: int = 6) -> None:
        self.exact: set[str] = set()
        self.buckets: dict[int, list[int]] = collections.defaultdict(list)
        self.max_hamming = max_hamming

    def seen(self, text: str) -> bool:
        key = exact_key(text)
        if key in self.exact:
            return True
        h = simhash(text)
        bucket = h >> 48
        for other in self.buckets[bucket]:
            if hamming(h, other) <= self.max_hamming:
                return True
        self.exact.add(key)
        self.buckets[bucket].append(h)
        return False


def collect_candidate_lines(raw_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in raw_sources:
        path = item["path"]
        source = item["source"]
        source_name = source["name"]
        trusted = bool(source.get("trusted_source", True))
        with path.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                text = clean_text(raw)
                tag = script_tag(text)
                if tag == "other":
                    continue
                if len(text.split()) < 5:
                    continue
                if non_linguistic_ratio(text, tag) > 0.30:
                    continue
                if is_template_or_robotic(text):
                    continue
                if is_english_heavy(text, tag):
                    continue
                if not is_probably_algerian(text, tag, trusted):
                    continue
                candidates.append({
                    "source": source_name,
                    "script": tag,
                    "text": text,
                    "quality_tier": source.get("quality_tier", "standard"),
                    "license_status": source.get("license_status", "unknown"),
                    "usage": source.get("usage", "pretraining"),
                    "max_source_tokens": source.get("max_source_tokens"),
                })
    return candidates


def train_char_lm(lines: list[str]) -> dict[str, Any]:
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    vocab: set[str] = set()
    for line in lines:
        padded = "^^" + line + "$"
        vocab.update(padded)
        for i in range(2, len(padded)):
            counts[padded[i - 2:i]][padded[i]] += 1
    return {"counts": counts, "vocab": vocab}


def char_perplexity(model: dict[str, Any], line: str) -> float:
    counts = model["counts"]
    vocab_size = max(len(model["vocab"]), 1)
    padded = "^^" + line + "$"
    loss = 0.0
    n = 0
    for i in range(2, len(padded)):
        ctx = padded[i - 2:i]
        ch = padded[i]
        total = sum(counts[ctx].values())
        prob = (counts[ctx][ch] + 1) / (total + vocab_size)
        loss -= math.log(prob)
        n += 1
    return math.exp(loss / max(n, 1))


def clean_and_dedup(raw_sources: list[dict[str, Any]], max_keep: int | None) -> Path:
    log("cleaning and deduplicating raw data")
    candidates = collect_candidate_lines(raw_sources)
    log(f"candidate lines after filters: {len(candidates)}")
    if not candidates:
        raise RuntimeError("No candidate lines survived cleaning.")

    lm_sample = [row["text"] for row in candidates[: min(len(candidates), 200_000)]]
    char_lm = train_char_lm(lm_sample)
    scored: list[tuple[float, dict[str, str]]] = [
        (char_perplexity(char_lm, row["text"]), row) for row in candidates
    ]
    ppls = sorted(score for score, _ in scored)
    cutoff = ppls[min(len(ppls) - 1, int(0.98 * len(ppls)))]

    deduper = Deduper()
    enc = tiktoken.get_encoding("cl100k_base")
    source_token_caps = {
        item["source"]["name"]: item["source"].get("max_source_tokens")
        for item in raw_sources
        if item["source"].get("max_source_tokens")
    }
    source_token_counts: collections.Counter[str] = collections.Counter()
    out_path = CLEAN_DIR / "darija_cleaned.jsonl"
    txt_path = CLEAN_DIR / "darija_cleaned.txt"
    kept = 0
    with out_path.open("w", encoding="utf-8") as jf, txt_path.open("w", encoding="utf-8") as tf:
        for ppl, row in scored:
            if ppl > cutoff:
                continue
            if deduper.seen(row["text"]):
                continue
            tokens = len(enc.encode(row["text"]))
            cap = source_token_caps.get(row["source"])
            if cap is not None and source_token_counts[row["source"]] + tokens > int(cap):
                continue
            source_token_counts[row["source"]] += tokens
            record = {**row, "char_ppl": ppl}
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            tf.write(row["text"] + "\n")
            kept += 1
            if max_keep is not None and kept >= max_keep:
                break
    log(f"cleaned corpus written: {out_path} lines={kept} ppl_cutoff={cutoff:.2f}")
    return out_path


def count_and_report(cleaned_jsonl: Path) -> dict[str, Any]:
    enc = tiktoken.get_encoding("cl100k_base")
    source_tokens: collections.Counter[str] = collections.Counter()
    source_lines: collections.Counter[str] = collections.Counter()
    script_tokens: collections.Counter[str] = collections.Counter()
    word_counts: collections.Counter[str] = collections.Counter()
    quality_tokens: collections.Counter[str] = collections.Counter()
    license_status: dict[str, str] = {}
    ppl_values: list[float] = []
    samples: dict[str, list[str]] = {"arabic": [], "arabizi": []}
    total_tokens = 0
    total_lines = 0

    with cleaned_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = row["text"]
            tokens = len(enc.encode(text))
            total_tokens += tokens
            total_lines += 1
            source_tokens[row["source"]] += tokens
            source_lines[row["source"]] += 1
            script_tokens[row["script"]] += tokens
            quality_tokens[row.get("quality_tier", "standard")] += tokens
            license_status[row["source"]] = row.get("license_status", "unknown")
            ppl_values.append(float(row.get("char_ppl", 0.0)))
            word_counts.update(text.lower().split())
            if row["script"] in samples and len(samples[row["script"]]) < 20:
                samples[row["script"]].append(text)

    def hist(vals: list[float], bins: int = 10) -> list[dict[str, float]]:
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if lo == hi:
            return [{"min": lo, "max": hi, "count": len(vals)}]
        width = (hi - lo) / bins
        counts = [0] * bins
        for val in vals:
            idx = min(bins - 1, int((val - lo) / width))
            counts[idx] += 1
        return [
            {"min": lo + i * width, "max": lo + (i + 1) * width, "count": c}
            for i, c in enumerate(counts)
        ]

    report = {
        "status": "MISSION COMPLETE" if total_tokens >= TARGET_TOKENS else "IN_PROGRESS",
        "target_tokens": TARGET_TOKENS,
        "total_tokens_cl100k_base": total_tokens,
        "tokens_remaining": max(0, TARGET_TOKENS - total_tokens),
        "total_lines": total_lines,
        "script_distribution_tokens": dict(script_tokens),
        "source_breakdown_tokens": dict(source_tokens),
        "source_breakdown_lines": dict(source_lines),
        "quality_tier_tokens": dict(quality_tokens),
        "source_license_status": license_status,
        "perplexity_histogram": hist(ppl_values),
        "top_100_words": word_counts.most_common(100),
        "spot_check_samples": samples,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "darija_corpus_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_spot_check(report)
    log(f"report written: {report_path} tokens={total_tokens}")
    return report


def write_spot_check(report: dict[str, Any]) -> None:
    lines = [
        "# Darija Corpus Spot Check",
        "",
        f"Status: {report['status']}",
        f"Tokens: {report['total_tokens_cl100k_base']:,} / {report['target_tokens']:,}",
        "",
    ]
    for script, samples in report["spot_check_samples"].items():
        lines.append(f"## {script}")
        lines.append("")
        for i, sample in enumerate(samples, 1):
            lines.append(f"{i}. {sample}")
        lines.append("")
    SPOT_CHECK_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_manifest(report: dict[str, Any], cleaned_jsonl: Path) -> None:
    record = {
        "source": "darija_data_mission_combined",
        "output": str(cleaned_jsonl),
        "bytes": cleaned_jsonl.stat().st_size,
        "license_status": "mixed; inspect raw source records before redistribution",
        "usage": "pretraining",
        "total_lines": report["total_lines"],
        "tokens_cl100k_base": report["total_tokens_cl100k_base"],
        "source_breakdown_tokens": report["source_breakdown_tokens"],
        "script_distribution_tokens": report["script_distribution_tokens"],
        "quality_tier_tokens": report["quality_tier_tokens"],
        "source_license_status": report["source_license_status"],
    }
    with MANIFEST_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def finalize_if_complete(cleaned_jsonl: Path, report: dict[str, Any]) -> None:
    if report["total_tokens_cl100k_base"] < TARGET_TOKENS:
        return
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cleaned_jsonl, FINAL_DIR / "darija_corpus.jsonl")
    shutil.copy2(CLEAN_DIR / "darija_cleaned.txt", FINAL_DIR / "darija_corpus.txt")
    shutil.copy2(REPORT_DIR / "darija_corpus_report.json", FINAL_DIR / "report.json")
    log("MISSION COMPLETE: final corpus copied to FINAL_DARIJA_CORPUS/")


def run_mission(args: argparse.Namespace) -> None:
    ensure_dirs()
    log("MISSION START: Algerian Darija corpus acquisition")
    sources = active_sources(load_sources(Path(args.sources)))
    raw_paths = discover_and_download(sources, args.max_rows_per_source, args.force_redownload)
    if not raw_paths:
        raise RuntimeError("No downloadable sources completed.")
    cleaned = clean_and_dedup(raw_paths, args.max_clean_lines)
    report = count_and_report(cleaned)
    update_manifest(report, cleaned)
    finalize_if_complete(cleaned, report)
    if report["status"] != "MISSION COMPLETE":
        log(
            f"MISSION CONTINUES: need {report['tokens_remaining']} more cl100k_base tokens. "
            "Add more source exports or API-backed scrapers and rerun."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Acquire and clean Algerian Darija corpus data.")
    p.add_argument("--sources", default="configs/darija_sources.json")
    p.add_argument("--max-rows-per-source", type=int, default=None)
    p.add_argument("--max-clean-lines", type=int, default=None)
    p.add_argument("--force-redownload", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    try:
        run_mission(parse_args())
    except KeyboardInterrupt:
        log("MISSION INTERRUPTED")
        sys.exit(130)
