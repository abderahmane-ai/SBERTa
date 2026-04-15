"""
clean_corpus.py — Strip noise from scraped corpus files in-place.

Passes applied (in order):
  1. URLs                   http://... / www....
  2. HTML tags              <br>, <div>, etc.
  3. Mentions               @username
  4. Hashtags               #DZ, #Algerie
  5. Long number strings    phone numbers, prices (5+ consecutive digits)
  6. Emoji                  all standard Unicode emoji ranges
  7. Repeated characters    loool → lool, sahhhh → sahh
  8. Whitespace collapse

Lines are then dropped if:
  - They contain zero Arabic/Latin letters (pure punctuation / table rows)
  - They have fewer than MIN_TOKENS whitespace-split tokens
  - They are exact duplicates of a previously kept line (per-file)

The script rewrites each file atomically: it writes to a .tmp sibling first,
then renames — so a crash mid-pass will never corrupt existing data.

Usage
-----
    python clean_corpus.py --corpus-dir corpus/youtube
    python clean_corpus.py --corpus-dir corpus/youtube --min-tokens 4 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Emoji regex ──────────────────────────────────────────────────────────────

# Covers all standard emoji Unicode ranges:
#   Emoticons / Misc Symbols                U+1F300–U+1F9FF
#   Transport / Map Symbols                 U+1F680–U+1F6FF  (subrange of above)
#   Supplemental Symbols and Pictographs    U+1FA00–U+1FA6F, U+1FA70–U+1FAFF
#   Dingbats                                U+2702–U+27B0
#   Enclosed Alphanumerics / Symbols        U+2460–U+24FF, U+25A0–U+26FF
#   Variation selectors                     U+FE00–U+FE0F
#   Zero-width joiner                       U+200D  (used in compound emoji)
#   Regional indicator symbols              U+1F1E0–U+1F1FF
#   Combining enclosing keycap              U+20E3
_EMOJI: re.Pattern = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # Misc Symbols, Emoticons, supplemental
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U00002460-\U000024FF"
    "\U000025A0-\U000026FF"
    "\U0001F1E0-\U0001F1FF"   # Regional indicators (flags)
    "\U0000FE00-\U0000FE0F"   # Variation selectors
    "\U0000200D"              # Zero-width joiner
    "\U000020E3"              # Combining enclosing keycap
    "]+",
    flags=re.UNICODE,
)

_URL: re.Pattern = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG: re.Pattern = re.compile(r"<[^>]+>")
_MENTION: re.Pattern = re.compile(r"@\w+")
_HASHTAG: re.Pattern = re.compile(r"#\w+")
_LONG_NUMBER: re.Pattern = re.compile(r"\b\d{5,}\b")
_REP_CHARS: re.Pattern = re.compile(r"(.)\1{2,}")
_MULTI_SPACE: re.Pattern = re.compile(r"\s{2,}")

# Matches any line that contains at least one Arabic or Latin letter.
# Lines with zero letters are pure punctuation / table separators and
# carry no linguistic signal useful for language model pre-training.
_HAS_LETTER: re.Pattern = re.compile(
    r"[\u0600-\u06FF\u0750-\u077Fa-zA-Z]"
)


def clean_text(text: str) -> str:
    """Remove URLs, HTML, mentions, hashtags, long numbers, emoji,
    excessive character repetitions, and collapse whitespace."""
    text = _URL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(" ", text)
    text = _LONG_NUMBER.sub(" ", text)
    text = _EMOJI.sub(" ", text)
    text = _REP_CHARS.sub(r"\1\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ─── Per-file cleaning ────────────────────────────────────────────────────────


def clean_file(
    path: Path,
    min_tokens: int = 4,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Strip emojis from every line of a corpus file.

    Args:
        path:       Path to a UTF-8 corpus file (one sentence per line).
        min_tokens: Lines with fewer whitespace-split tokens than this after
                    cleaning are dropped.
        dry_run:    If True, report what would change but do not write.

    Returns:
        (lines_in, lines_kept, lines_dropped) counts.
    """
    lines_in = 0
    kept: list[str] = []
    seen: set[str] = set()

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            lines_in += 1
            cleaned = clean_text(raw.rstrip("\n"))
            if not _HAS_LETTER.search(cleaned):
                continue  # pure punctuation / table row
            if len(cleaned.split()) < min_tokens:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            kept.append(cleaned)

    lines_dropped = lines_in - len(kept)

    if not dry_run:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as out:
            for line in kept:
                out.write(line + "\n")
        tmp.replace(path)   # atomic rename — crash-safe

    return lines_in, len(kept), lines_dropped


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strip emojis and noise from SBERTa corpus files."
    )
    p.add_argument(
        "--corpus-dir",
        required=True,
        type=Path,
        help="Root directory containing *.txt corpus files to clean in-place.",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=4,
        help="Minimum whitespace tokens per line after emoji removal (default: 4).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report statistics without modifying files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir: Path = args.corpus_dir

    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {corpus_dir}")

    txt_files = sorted(corpus_dir.rglob("*.txt"))
    if not txt_files:
        log.warning("No .txt files found in %s", corpus_dir)
        return

    total_in = total_kept = total_dropped = 0

    for path in txt_files:
        if path.name == "checkpoint.json":
            continue
        try:
            lines_in, lines_kept, lines_dropped = clean_file(
                path, min_tokens=args.min_tokens, dry_run=args.dry_run
            )
        except PermissionError:
            log.warning("Skipping (permission denied): %s", path)
            continue
        total_in += lines_in
        total_kept += lines_kept
        total_dropped += lines_dropped
        pct = 100.0 * lines_dropped / lines_in if lines_in else 0
        action = "would drop" if args.dry_run else "dropped"
        log.info(
            "%-30s  %6d → %6d lines  (%s %d, %.1f%%)",
            path.name,
            lines_in,
            lines_kept,
            action,
            lines_dropped,
            pct,
        )

    log.info("─" * 60)
    log.info(
        "Total  %d lines in → %d kept  (%d dropped, %.1f%%)",
        total_in,
        total_kept,
        total_dropped,
        100.0 * total_dropped / total_in if total_in else 0,
    )
    if args.dry_run:
        log.info("DRY RUN — no files were modified.")
    else:
        log.info("All files cleaned in-place.")


if __name__ == "__main__":
    main()
