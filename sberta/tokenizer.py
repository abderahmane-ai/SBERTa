"""
SBERTa tokenizer — SentencePiece Unigram wrapper for Algerian Darija.

Design
------
Algorithm      : SentencePiece Unigram LM — probabilistic segmentation via EM,
                 optimal for morphologically rich and low-resource languages.
                 Crucially, it supports subword regularisation: sampling from
                 multiple valid segmentations at training time, acting as data
                 augmentation for under-represented scripts (Berber/Tamazight).

Vocab size     : 50,265 (matches SBERTaConfig default).

Normalisation  : NFC → strip Arabic harakat → remove tatweel → Arabic-Indic
                 digits to ASCII → lowercase non-Arabic uppercase → collapse
                 whitespace. Arabizi digit-letters (3, 7, 9, …) are untouched:
                 they are phonemic units and the SP model learns them as such.

Script policy  : No script-boundary pre-tokenisation. Splitting on Arabic↔Latin
                 transitions would guarantee clean switch boundaries but would
                 contradict SBERTa's philosophy of deriving language identity
                 endogenously from token embeddings. The prototype mechanism
                 handles ambiguity; the tokeniser must not pre-empt it.

Special tokens : [PAD]=0, [UNK]=1, [MASK]=2, [SEP]=3. No [CLS] — SBERTa uses
                 mean pooling over H^(L) for sequence-level tasks.

Byte fallback  : enabled — any unseen character maps to its UTF-8 byte tokens
                 rather than [UNK], preserving all information at inference.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sentencepiece as spm
import torch


# ─── Normaliser ───────────────────────────────────────────────────────────────

# Arabic harakat (diacritics) — never written in native Darija; appear only in
# pasted MSA or religious text. Unicode ranges: 0610–061A, 064B–065F, 0670,
# 06D6–06DC, 06DF–06E4, 06E7, 06E8, 06EA–06ED.
_HARAKAT: re.Pattern = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670"
    r"\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Tatweel / kashida — decorative elongation glyph (ـ). Semantically empty.
_TATWEEL: re.Pattern = re.compile(r"\u0640+")

# Arabic-Indic digit → ASCII digit mapping
_ARABIC_INDIC: Dict[int, int] = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Non-Arabic uppercase (ASCII only) — for efficient lowercasing
_NON_ARABIC_UPPER: re.Pattern = re.compile(r'[A-Z]')

# Arabic Unicode block (U+0600–U+06FF) — used to preserve case of Arabic chars
_ARABIC_BLOCK_LO: int = 0x0600
_ARABIC_BLOCK_HI: int = 0x06FF


def normalise(text: str) -> str:
    """
    Normalise a Darija string for tokenisation.

    Steps (order matters):
      1. NFC — canonical decomposition + recomposition; collapses compatibility
         variants (e.g. ﻻ → لا).
      2. Strip harakat — diacritics are absent in Darija and add vocabulary
         noise when copied from MSA sources.
      3. Remove tatweel — purely decorative; 'كييييف' and 'كيف' are the same word.
      4. Arabic-Indic digits → ASCII — prevents the same Arabizi digit (e.g. 3)
         from appearing as two token types depending on the user's keyboard.
      5. Lowercase non-Arabic uppercase — halves the effective Latin alphabet
         and handles inconsistent social-media casing. Arabic has no case.
      6. Collapse whitespace.

    Arabizi digit-letters (2 ء, 3 ع, 5 خ, 6 ط, 7 ح, 9 ق) survive steps 1–6
    unchanged. The SP model trained on Darija data learns them as part of
    coherent lexical units (3ndek, 7atta, m9abel, …).

    Args:
        text: raw Darija string.
    Returns:
        Normalised string.
    """
    text = unicodedata.normalize("NFC", text)
    text = _HARAKAT.sub("", text)
    text = _TATWEEL.sub("", text)
    text = text.translate(_ARABIC_INDIC)
    # Lowercase only ASCII uppercase (Arabic has no case) - vectorized with regex
    text = _NON_ARABIC_UPPER.sub(lambda m: m.group(0).lower(), text)
    text = " ".join(text.split())
    return text


# ─── Tokenizer ────────────────────────────────────────────────────────────────


class SBERTaTokenizer:
    """
    SentencePiece Unigram tokenizer for SBERTa.

    Special token IDs are baked into the SP model at training time
    (see train_tokenizer.py); no manual ID shifting is required here.

    Special tokens:
        [PAD]  : id 0  — padding
        [UNK]  : id 1  — unknown (byte fallback enabled: rarely fires)
        [MASK] : id 2  — used by the MLM objective
        [SEP]  : id 3  — sequence separator; used for sentence-pair fine-tuning

    Usage:
        tok = SBERTaTokenizer("sberta.model")
        ids = tok.encode("wach rak?")
        batch = tok.batch_encode(["wach rak?", "أنا labas"], max_length=128)
    """

    PAD_TOKEN: str = "[PAD]"
    UNK_TOKEN: str = "[UNK]"
    MASK_TOKEN: str = "[MASK]"
    SEP_TOKEN: str = "[SEP]"

    PAD_ID: int = 0
    UNK_ID: int = 1
    MASK_ID: int = 2
    SEP_ID: int = 3

    def __init__(self, model_path: Union[str, Path]) -> None:
        """
        Args:
            model_path: path to the .model file produced by train_tokenizer.py.
        """
        self._sp: spm.SentencePieceProcessor = spm.SentencePieceProcessor()
        self._sp.Load(str(model_path))
        self._vocab_size: int = self._sp.GetPieceSize()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ── Encoding ──────────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        """
        Normalise and segment text into subword pieces.

        Args:
            text: raw input string.
        Returns:
            List of subword piece strings (▁ prefix marks space-initial tokens).
        """
        return self._sp.EncodeAsPieces(normalise(text))

    def encode(
        self,
        text: str,
        add_sep: bool = True,
        max_length: Optional[int] = None,
        sample: bool = False,
        sample_alpha: float = 0.1,
    ) -> List[int]:
        """
        Normalise and encode text to token IDs.

        Args:
            text:         raw input string.
            add_sep:      append [SEP] at the end (True for pre-training).
            max_length:   truncate to this many tokens (including [SEP]).
            sample:       if True, sample a segmentation from the Unigram
                          distribution instead of taking the Viterbi path.
                          Enable during pre-training for subword regularisation.
            sample_alpha: smoothing coefficient for sampling (default 0.1).
        Returns:
            List of integer token IDs.
        """
        text = normalise(text)
        if sample:
            ids: List[int] = self._sp.SampleEncodeAsIds(
                text, nbest_size=-1, alpha=sample_alpha
            )
        else:
            ids = self._sp.EncodeAsIds(text)
        if add_sep:
            ids = ids + [self.SEP_ID]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def encode_pair(
        self,
        text_a: str,
        text_b: str,
        max_length: Optional[int] = None,
        sample: bool = False,
        sample_alpha: float = 0.1,
    ) -> Tuple[List[int], List[int]]:
        """
        Encode a sentence pair for fine-tuning tasks.

        Returns:
            ids:             text_a [SEP] text_b [SEP]
            token_type_ids:  0…0 1…1 (segment A vs B)

        Args:
            text_a, text_b: raw input strings.
            max_length:      total length cap (applied after concatenation).
            sample:          subword regularisation sampling.
            sample_alpha:    smoothing coefficient.
        """
        ids_a: List[int] = self.encode(
            text_a, add_sep=True, sample=sample, sample_alpha=sample_alpha
        )
        ids_b: List[int] = self.encode(
            text_b, add_sep=True, sample=sample, sample_alpha=sample_alpha
        )
        ids: List[int] = ids_a + ids_b
        token_type_ids: List[int] = [0] * len(ids_a) + [1] * len(ids_b)
        if max_length is not None:
            ids = ids[:max_length]
            token_type_ids = token_type_ids[:max_length]
        return ids, token_type_ids

    # ── Batch encoding ────────────────────────────────────────────────────────

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        pad: bool = True,
        add_sep: bool = True,
        sample: bool = False,
        sample_alpha: float = 0.1,
        return_tensors: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of strings with padding.

        Args:
            texts:        list of raw input strings.
            max_length:   maximum sequence length (truncation + padding target).
            pad:          if True, pad all sequences to the same length.
            add_sep:      append [SEP] to each sequence.
            sample:       subword regularisation sampling.
            sample_alpha: smoothing coefficient.
            return_tensors: if True, return torch.Tensors; else return lists.
        Returns:
            dict with:
                input_ids:      (B, T) int64
                attention_mask: (B, T) int64 — 1 for real tokens, 0 for padding
        """
        all_ids: List[List[int]] = [
            self.encode(
                t,
                add_sep=add_sep,
                max_length=max_length,
                sample=sample,
                sample_alpha=sample_alpha,
            )
            for t in texts
        ]

        if not pad:
            if return_tensors:
                raise ValueError(
                    "return_tensors=True requires pad=True for batched encoding."
                )
            return {"input_ids": all_ids}

        # Pad to the length of the longest sequence in the batch, capped at
        # max_length. This avoids unnecessary padding when the batch is short.
        T: int = min(max(len(ids) for ids in all_ids), max_length)
        
        # Vectorized padding: pre-allocate tensors
        B = len(all_ids)
        padded_ids = torch.full((B, T), self.PAD_ID, dtype=torch.long)
        masks = torch.zeros((B, T), dtype=torch.long)
        
        for i, ids in enumerate(all_ids):
            length = min(len(ids), T)
            padded_ids[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
            masks[i, :length] = 1

        if not return_tensors:
            return {"input_ids": padded_ids.tolist(), "attention_mask": masks.tolist()}

        return {
            "input_ids": padded_ids,
            "attention_mask": masks,
        }

    # ── Decoding ──────────────────────────────────────────────────────────────

    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to a string.

        Args:
            ids:                  list or 1-D tensor of token IDs.
            skip_special_tokens:  if True, remove [PAD], [UNK], [MASK], [SEP].
        Returns:
            Decoded string.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        special: set = (
            {self.PAD_ID, self.UNK_ID, self.MASK_ID, self.SEP_ID}
            if skip_special_tokens
            else set()
        )
        ids = [i for i in ids if i not in special]
        return self._sp.DecodeIds(ids)

    def batch_decode(
        self,
        ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            ids: (B, T) tensor or list of lists.
            skip_special_tokens: strip special tokens before decoding.
        Returns:
            List of decoded strings.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [self.decode(row, skip_special_tokens) for row in ids]

    # ── Vocabulary utilities ──────────────────────────────────────────────────

    def id_to_piece(self, token_id: int) -> str:
        return self._sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        return self._sp.PieceToId(piece)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._sp.IdToPiece(i) for i in ids]

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, directory: Union[str, Path]) -> "SBERTaTokenizer":
        """Load from a directory containing sberta.model."""
        return cls(Path(directory) / "sberta.model")

    def save(self, directory: Union[str, Path]) -> None:
        """Copy the SP model file into directory."""
        import shutil
        dest: Path = Path(directory)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(self._sp.model_file(), str(dest / "sberta.model"))