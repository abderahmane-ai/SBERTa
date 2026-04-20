"""
SBERTa tokenizer training script.

Trains a 50 k SentencePiece Unigram model on a Darija corpus.

Usage
-----
    python train_tokenizer.py \
        --input  data/raw/*.txt      \
        --output runs/tokenizer/     \
        --vocab_size 50265           \
        --num_threads 8

The script:
  1. Streams corpus files through the normaliser and writes a cleaned temp file.
  2. Trains SentencePiece with the settings described below.
  3. Writes sberta.model and sberta.vocab to the output directory.

SentencePiece training settings
---------------------------------
model_type         : unigram — probabilistic EM segmentation; handles script
                     imbalance and morphological richness better than BPE.

character_coverage : 0.9999 — retains essentially all character types across
                     Arabic, Latin, and Arabizi scripts. Combined with byte
                     fallback, coverage is effectively 1.0.

byte_fallback      : True — any character outside the learnt vocabulary maps to
                     its UTF-8 byte tokens rather than [UNK]. This guarantees
                     zero information loss at inference and is essential for a
                     language with active script innovation (Arabizi).

max_sentencepiece_length: 16 — caps subword piece length. Prevents the model
                     from learning entire words as single pieces in high-
                     frequency Arabic morphological clusters, which would make
                     switch magnitude s_t less informative.

split_digits       : False — Arabizi uses digits as phonemes (3=ع, 7=ح, 9=ق).
                     Splitting on digit boundaries would fragment Arabizi words.

split_by_unicode_script: False — same reason: Arabizi mixes Latin chars and
                     digit-phonemes in a single word (3ndek, m7el). Script
                     boundaries inside Arabizi words must not become token
                     boundaries.

Special token IDs (baked into the SP model):
    [PAD]  = 0    pad_id  / pad_piece
    [UNK]  = 1    unk_id  / unk_piece
    [MASK] = 2    user_defined_symbols[0]
    [SEP]  = 3    user_defined_symbols[1]
    BOS/EOS disabled (SBERTa does not use them).

"""

from __future__ import annotations

import argparse
import glob
import io
import logging
import os
import tempfile
import re
from pathlib import Path
from typing import Iterator, List, Optional

import sentencepiece as spm

from sberta.tokenizer import normalise

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log: logging.Logger = logging.getLogger(__name__)


# ─── Corpus streaming ────────────────────────────────────────────────────────


def iter_lines(paths: List[Path], min_chars: int = 5) -> Iterator[str]:
    """
    Stream normalised lines from a list of text files.
    
    Includes principled upsampling of Arabic script lines to prevent Unigram 
    from starving the Arabic vocabulary in Arabizi-dominated corpora.

    Args:
        paths:     list of plain-text corpus files (UTF-8, one sentence per line).
        min_chars: skip lines shorter than this after normalisation.
    Yields:
        Normalised non-empty lines (with Arabic-heavy lines yielded 3x).
    """
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line: str = normalise(raw.strip())
                if len(line) >= min_chars:
                    yield line
                    # Upsample if line is >30% Arabic script
                    if len(arabic_pattern.findall(line)) > len(line) * 0.3:
                        yield line
                        yield line


def write_normalised_corpus(
    paths: List[Path],
    dest: Path,
    min_chars: int = 5,
    max_lines: Optional[int] = None,
) -> int:
    """
    Write a normalised corpus to a single temp file for SentencePiece.

    SentencePiece training reads from a single file; this function collapses
    all input files into one after normalisation.

    Args:
        paths:     input corpus files.
        dest:      output file path.
        min_chars: minimum line length after normalisation.
        max_lines: cap total lines (useful for fast tokenizer iterations).
    Returns:
        Number of lines written.
    """
    n: int = 0
    with open(dest, "w", encoding="utf-8") as out:
        for line in iter_lines(paths, min_chars):
            out.write(line + "\n")
            n += 1
            if max_lines is not None and n >= max_lines:
                break
            if n % 500_000 == 0:
                log.info("Normalised %d lines …", n)
    return n


# ─── SentencePiece training ──────────────────────────────────────────────────


def train(
    corpus_path: Path,
    output_prefix: Path,
    vocab_size: int = 50_265,
    character_coverage: float = 0.9999,
    num_threads: int = 4,
    input_sentence_size: int = 10_000_000,
    shuffle_input_sentence: bool = True,
) -> None:
    """
    Train a SentencePiece Unigram model.

    Args:
        corpus_path:            path to the normalised corpus file.
        output_prefix:          prefix for .model and .vocab outputs.
        vocab_size:             target vocabulary size (should match SBERTaConfig).
        character_coverage:     proportion of characters to cover.
        num_threads:            parallelism for training.
        input_sentence_size:    maximum sentences read for training
                                (set < corpus size to subsample).
        shuffle_input_sentence: shuffle before subsampling.
    """
    # Special tokens are baked in with fixed IDs so SBERTaTokenizer never
    # needs to shift IDs manually.
    params: str = " ".join([
        f"--input={corpus_path}",
        f"--model_prefix={output_prefix}",
        f"--vocab_size={vocab_size}",
        f"--model_type=unigram",
        f"--character_coverage={character_coverage}",
        f"--num_threads={num_threads}",
        f"--input_sentence_size={input_sentence_size}",
        f"--shuffle_input_sentence={'true' if shuffle_input_sentence else 'false'}",
        # ── Special tokens ────────────────────────────────────────────────
        "--pad_id=0",
        "--pad_piece=[PAD]",
        "--unk_id=1",
        "--unk_piece=[UNK]",
        "--bos_id=-1",          # disabled — SBERTa has no BOS
        "--eos_id=-1",          # disabled — SBERTa has no EOS
        "--user_defined_symbols=[MASK],[SEP]",   # → ids 2, 3
        # ── Script and digit handling ─────────────────────────────────────
        "--split_digits=false",             # preserve Arabizi digit-phonemes
        "--split_by_number=false",          # MUST BE FALSE: allows numbers/letters to mix in one word (e.g. 3ndek)
        "--split_by_unicode_script=false",  # do not split inside Arabizi words
        "--byte_fallback=true",             # no true UNK at inference
        # ── Piece length ──────────────────────────────────────────────────
        "--max_sentencepiece_length=16",
        # ── Training stability ────────────────────────────────────────────
        "--seed_sentencepiece_size=1000000",
        "--hard_vocab_limit=false",         # graceful degradation if corpus is small
        "--train_extremely_large_corpus=false",
    ])
    log.info("Starting SentencePiece training …")
    spm.SentencePieceTrainer.Train(params)
    log.info("Training complete: %s.model, %s.vocab", output_prefix, output_prefix)


# ─── Verification ────────────────────────────────────────────────────────────


def verify(model_path: Path) -> None:
    """
    Smoke-test the trained model on a small Darija sentence set covering
    all scripts present in the training corpus.
    """
    sp: spm.SentencePieceProcessor = spm.SentencePieceProcessor()
    sp.Load(str(model_path))

    test_sentences: List[str] = [
        "wach rak labas?",              # Arabizi (Latin)
        "أنا بخير والحمد لله",           # Arabic script
        "je suis en bonne santé",       # French
        "3ndek chi haja?",              # Arabizi with digit-phoneme
        "tafawt d taqbaylit",           # Berber/Tamazight
        "wach t3awdli l'histoire?",     # code-switch: Arabizi + French
        "download-it men google",       # English word in Darija context
        "m9abel ça va",                 # Arabizi + French switch
    ]

    log.info("─── Verification ───────────────────────────────────────")
    for sent in test_sentences:
        normed: str = normalise(sent)
        pieces: List[str] = sp.EncodeAsPieces(normed)
        ids: List[int] = sp.EncodeAsIds(normed)
        decoded: str = sp.DecodeIds(ids)
        log.info("Input   : %s", sent)
        log.info("Normed  : %s", normed)
        log.info("Pieces  : %s", pieces)
        log.info("Roundtrip: %s", decoded)
        log.info("")

    # Confirm special token IDs
    assert sp.PieceToId("[PAD]") == 0,   "[PAD] must be id 0"
    assert sp.PieceToId("[UNK]") == 1,   "[UNK] must be id 1"
    assert sp.PieceToId("[MASK]") == 2,  "[MASK] must be id 2"
    assert sp.PieceToId("[SEP]") == 3,   "[SEP] must be id 3"
    log.info("Special token IDs verified: [PAD]=0, [UNK]=1, [MASK]=2, [SEP]=3")
    log.info("Vocabulary size: %d", sp.GetPieceSize())


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SBERTa SentencePiece Unigram tokenizer."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help=(
            "Corpus files or glob patterns (e.g. data/raw/*.txt). "
            "Plain-text, UTF-8, one sentence/segment per line."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will contain sberta.model and sberta.vocab.",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=50_265,
        help="Vocabulary size (default: 50265, must match SBERTaConfig.vocab_size).",
    )
    parser.add_argument(
        "--character_coverage", type=float, default=0.9999,
        help="Fraction of characters to cover (default: 0.9999).",
    )
    parser.add_argument(
        "--num_threads", type=int, default=4,
        help="Threads for SP training (default: 4).",
    )
    parser.add_argument(
        "--max_lines", type=int, default=None,
        help="Cap input lines for fast iteration (default: no cap).",
    )
    parser.add_argument(
        "--min_chars", type=int, default=5,
        help="Minimum characters per line after normalisation (default: 5).",
    )
    parser.add_argument(
        "--input_sentence_size", type=int, default=10_000_000,
        help="Max sentences sampled by SP trainer (default: 10M).",
    )
    parser.add_argument(
        "--no_verify", action="store_true",
        help="Skip post-training verification.",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    # Resolve glob patterns
    input_paths: List[Path] = []
    for pattern in args.input:
        matched: List[str] = glob.glob(pattern, recursive=True)
        if not matched:
            log.warning("No files matched pattern: %s", pattern)
        input_paths.extend(Path(p) for p in matched)

    if not input_paths:
        raise FileNotFoundError("No input files found. Check --input patterns.")

    log.info("Input files: %d", len(input_paths))

    output_dir: Path = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix: Path = output_dir / "sberta"

    # Normalise corpus into a single temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path: Path = Path(tmp.name)

    try:
        n_lines: int = write_normalised_corpus(
            input_paths, tmp_path, min_chars=args.min_chars, max_lines=args.max_lines
        )
        log.info("Normalised corpus: %d lines → %s", n_lines, tmp_path)

        train(
            corpus_path=tmp_path,
            output_prefix=output_prefix,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            num_threads=args.num_threads,
            input_sentence_size=args.input_sentence_size,
        )
    finally:
        os.unlink(tmp_path)

    if not args.no_verify:
        verify(output_prefix.with_suffix(".model"))


if __name__ == "__main__":
    main()
