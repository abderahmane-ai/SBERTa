# Tokenizer Reference

> Part of the SBERTa project. For model architecture and training, see [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

SBERTa uses a custom **SentencePiece Unigram** tokenizer for **Algerian Darija**, a North African Arabic variety with pervasive code-switching across Arabic script, Arabizi, French, Tamazight-influenced forms, and informal social-media spellings.

The tokenizer is Darija-specific. It is not just a generic multilingual tokenizer with a Darija corpus behind it.

---

## Contents

- [Algorithm Choice](#algorithm-choice)
- [Normalisation Pipeline](#normalisation-pipeline)
- [Arabizi Handling](#arabizi-handling)
- [Technical Configuration](#technical-configuration)
- [Special Tokens](#special-tokens)
- [File Reference](#file-reference)
- [Usage](#usage)
- [Training The Tokenizer](#training-the-tokenizer)

---

## Algorithm Choice

SBERTa uses the **Unigram Language Model** algorithm from SentencePiece.

| Property | BPE | Unigram LM |
| :--- | :--- | :--- |
| Strategy | Greedy merge rules | Probabilistic EM optimisation |
| Segmentation | One deterministic path | Best path plus sampling |
| Morphology | Can over-merge frequent forms | Learns piece probabilities |
| Data augmentation | No native sampling | Supports subword regularisation |

### Subword Regularisation

Darija words often have many spellings: Arabic script variation, Arabizi digit-phonemes, French loanwords, and informal elongation. During pre-training, the tokenizer can sample from valid segmentations:

```python
tok.encode("wach rak labas?", sample=True, sample_alpha=0.1)
```

This acts as lightweight data augmentation without changing the corpus.

---

## Normalisation Pipeline

All text passes through `sberta/tokenizer.py -> normalise()`.

```text
Raw text
   |
   +-- 1. NFC Unicode normalisation
   |
   +-- 2. Strip Arabic harakat
   |
   +-- 3. Remove tatweel
   |
   +-- 4. Arabic-Indic digits -> ASCII
   |
   +-- 5. Fold French accents
   |
   +-- 6. Lowercase ASCII uppercase
   |
   +-- 7. Fold Arabic orthographic variants
   |
   +-- 8. Cap character elongations at two repeats
   |
   +-- 9. Collapse whitespace
```

### Arabic Normalisation

| Input pattern | Normalised form | Reason |
| :--- | :--- | :--- |
| `أ إ آ` | `ا` | Reduces sparse Alef variants |
| `ة` | `ه` | Common dialectal spelling fold |
| `ى` | `ي` | Reduces orthographic duplicates |
| Harakat | removed | Rare in native Darija text |
| Tatweel | removed | Decorative, not lexical |

### Latin / French Normalisation

French accents are folded to base Latin characters, and ASCII uppercase is lowercased. This keeps `ça`, `ca`, `École`, and `ecole` closer in the vocabulary when they appear inside Darija contexts.

### Elongation

Character elongations are capped at two repeats:

```text
haaaaaay -> haay
كييييف -> كييف
```

This keeps expressive spelling without letting extreme elongation dominate the vocabulary.

---

## Arabizi Handling

Arabizi represents Arabic phonemes with Latin letters and digits:

| Digit | Approximate phoneme | Example |
| :---: | :--- | :--- |
| `2` | hamza | `2ana` |
| `3` | ayn | `3ndek` |
| `5` | kha | `5oya` |
| `7` | ha | `7atta` |
| `9` | qaf/gaf-like forms | `m9abel` |

These digits are linguistic characters in Arabizi. The tokenizer preserves them by training SentencePiece with:

```text
--split_digits=false
--split_by_number=false
--split_by_unicode_script=false
```

This prevents words such as `3ndek`, `m7el`, and `m9abel` from being fragmented at digit or script boundaries.

---

## Technical Configuration

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| `vocab_size` | 50,000 | Comparable to DziriBERT and large enough for mixed scripts |
| `model_type` | `unigram` | Probabilistic segmentation and sampling |
| `character_coverage` | 0.9999 | Keeps rare script characters |
| `byte_fallback` | true | Avoids true unknown-character loss |
| `max_sentencepiece_length` | 16 | Avoids over-long memorised chunks |
| `split_digits` | false | Preserves Arabizi digit-phonemes |
| `split_by_number` | false | Allows mixed digit/letter words |
| `split_by_unicode_script` | false | Keeps mixed-script tokens intact |
| BOS / EOS | disabled | SBERTa uses `[SEP]`, not BOS/EOS |

---

## Special Tokens

Special token IDs are baked into the SentencePiece model.

| Token | ID | Role |
| :--- | :---: | :--- |
| `[PAD]` | 0 | Padding |
| `[UNK]` | 1 | Unknown fallback, rarely used with byte fallback |
| `[MASK]` | 2 | Generator MLM masking |
| `[SEP]` | 3 | Sequence separator |

During pre-training, `[PAD]`, `[UNK]`, `[MASK]`, and `[SEP]` are excluded from generator replacement sampling.

---

## File Reference

| File | Purpose |
| :--- | :--- |
| `sberta/tokenizer.py` | Tokenizer wrapper and normaliser |
| `train_tokenizer.py` | SentencePiece training CLI |
| `scripts/prepare_corpus.py` | Corpus cleaning, deduplication, manifest writing |
| `DATA.md` | Data source policy and manifest schema |

---

## Usage

### Basic Encoding

```python
from sberta.tokenizer import SBERTaTokenizer

tok = SBERTaTokenizer("runs/tokenizer/sberta.model")
ids = tok.encode("wach rak labas?")
pieces = tok.convert_ids_to_tokens(ids)
```

### Batch Encoding

```python
batch = tok.batch_encode(
    ["wach rak?", "أنا labas"],
    max_length=128,
    return_tensors=True,
)
```

### Sentence Pairs

```python
ids, token_type_ids = tok.encode_pair("wach rak?", "labas, hamdullah")
```

### Decoding

```python
text = tok.decode(ids, skip_special_tokens=True)
```

---

## Training The Tokenizer

```bash
python train_tokenizer.py \
  --input corpus/darija_pretrain.txt \
  --output runs/tokenizer \
  --vocab_size 50000 \
  --num_threads 8
```

The script:

1. Streams input files through `normalise()`.
2. Upsamples Arabic-heavy lines to protect Arabic-script vocabulary when the corpus is Arabizi-heavy.
3. Trains SentencePiece Unigram with byte fallback.
4. Verifies special token IDs and roundtrip behavior.
5. Writes `sberta.model` and `sberta.vocab`.

Key options:

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--vocab_size` | 50000 | Must match `SBERTaConfig.vocab_size` |
| `--num_threads` | 4 | SentencePiece trainer parallelism |
| `--min_chars` | 5 | Minimum normalised characters |
| `--max_lines` | none | Cap lines for fast iteration |
| `--input_sentence_size` | 10M | SentencePiece sampling cap |
| `--no_verify` | false | Skip verification |
