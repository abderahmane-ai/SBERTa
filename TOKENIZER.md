# Tokenizer Reference

> **Part of the SBERTa project.** For model architecture and training, see [README.md](README.md).

The SBERTa tokenizer is a custom **SentencePiece Unigram** model built for **Algerian Darija**, a North African Arabic variety characterised by pervasive code-switching across Arabic, French, Berber (Tamazight), and Arabizi scripts.

---

## Contents

- [Algorithm Choice](#algorithm-choice)
- [Normalisation Pipeline](#normalisation-pipeline)
- [Arabizi Handling](#arabizi-handling)
- [Technical Configuration](#technical-configuration)
- [Special Tokens](#special-tokens)
- [File Reference](#file-reference)
- [Usage](#usage)
- [Training the Tokenizer](#training-the-tokenizer)

---

## Algorithm Choice

SBERTa uses the **Unigram Language Model** algorithm from SentencePiece. This differs fundamentally from Byte-Pair Encoding (BPE):

| Property | BPE | Unigram LM |
| :--- | :--- | :--- |
| **Strategy** | Greedy merge rules | Probabilistic EM optimisation |
| **Segmentation** | Single deterministic path | Best path + sampling |
| **Morphology** | Struggles with agglutinative/mixed scripts | Robust — models piece probability directly |
| **Data augmentation** | No | Yes — subword regularisation |

### Subword Regularisation

During pre-training, the tokenizer samples from the distribution of valid segmentations instead of always taking the single best path. For a language like Darija — where the same word can be spelled dozens of ways — this provides **built-in data augmentation** and improves robustness to orthographic variance.

Controlled by `encode(..., sample=True, sample_alpha=0.1)`.

---

## Normalisation Pipeline

Before tokenisation, all text passes through a deterministic normalisation function (`sberta/tokenizer.py → normalise()`). **Order is critical.**

```
Raw text
   │
   ├─ 1. NFC Unicode normalisation
   │      Collapses compatibility variants (e.g. ﻻ → لا)
   │
   ├─ 2. Strip Arabic harakat (diacritics)
   │      Strips U+0610–U+061A, U+064B–U+065F, U+0670, U+06D6–U+06ED
   │      These are absent in native Darija; they appear only in copy-pasted MSA or religious text.
   │
   ├─ 3. Remove tatweel (kashida)
   │      Strips U+0640 — a decorative elongation glyph.
   │      "كييييف" and "كيف" become the same token.
   │
   ├─ 4. Arabic-Indic digits → ASCII
   │      ٠١٢٣٤٥٦٧٨٩  →  0123456789
   │      Prevents the same Arabizi digit-phoneme (e.g. 3) appearing as two token types.
   │
   ├─ 5. Lowercase non-Arabic characters
   │      Latin and other non-Arabic uppercase → lowercase.
   │      Arabic has no case concept; the Arabic Unicode block is left untouched.
   │
   └─ 6. Collapse whitespace
          Normalise runs of whitespace to single spaces.
```

---

## Arabizi Handling

**Arabizi** is an informal writing system that represents Arabic phonemes using Latin letters and digit substitutes:

| Digit | Arabic phoneme | Example |
| :---: | :---: | :--- |
| `2` | ء (hamza) | `2ana` = أنا |
| `3` | ع (ayn) | `3ndek` = عندك |
| `5` | خ (kha) | `5oya` = خويا |
| `7` | ح (ha) | `7atta` = حتى |
| `9` | ق (qaf) | `m9abel` = مقابل |

These digit-phonemes are **phonemic units** in Arabizi. Splitting on digit boundaries would fragment words like `3ndek` and `m9abel` into meaningless pieces.

### How the tokenizer preserves Arabizi

The SentencePiece model is trained with:

```
--split_digits=false              # 3ndek stays together
--split_by_number=false           # digits/letters can mix in one word
--split_by_unicode_script=false   # no forced splits at Arabic↔Latin boundaries
```

Digit-phonemes survive the normalisation pipeline unchanged (no digit stripping). The Unigram model learns them as coherent lexical units because they appear consistently in Darija text.

### No script-boundary pre-tokenisation

Some tokenizers force token breaks whenever the script changes (e.g. Arabic → Latin). SBERTa deliberately **does not do this**. The prototype mechanism learns language identity endogenously from token embeddings; pre-empting it with hard script boundaries would undermine the model's ability to represent code-switching as a gradient phenomenon.

---

## Technical Configuration

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| `vocab_size` | 50,265 | Matches `SBERTaConfig.vocab_size`; same as RoBERTa for comparability |
| `model_type` | `unigram` | Probabilistic EM; handles script imbalance better than BPE |
| `character_coverage` | 0.9999 | Retains essentially all character types across all scripts |
| `byte_fallback` | `true` | Unseen characters → UTF-8 byte tokens; zero information loss |
| `max_sentencepiece_length` | 16 | Prevents multi-morpheme clusters from being learned as one piece |
| `split_digits` | `false` | Preserves Arabizi digit-phonemes |
| `split_by_unicode_script` | `false` | Preserves mixed-script Arabizi words |
| `BOS / EOS` | disabled | SBERTa uses mean-pooling; no sentinel tokens needed |

---

## Special Tokens

Special token IDs are **baked into the SentencePiece model** at training time. The wrapper never shifts IDs manually.

| Token | ID | Role |
| :--- | :---: | :--- |
| `[PAD]` | **0** | Padding to fill batches to a fixed length. |
| `[UNK]` | **1** | Unknown token. Rarely fires — byte fallback is always active. |
| `[MASK]` | **2** | Masked position for the MLM pre-training objective. |
| `[SEP]` | **3** | Sequence separator; appended to every encoded sequence. Used for sentence-pair fine-tuning. |

> **No `[CLS]`** — SBERTa obtains sequence representations via **mean-pooling** over all final hidden states, not a special prepended token.

---

## File Reference

| File | Purpose |
| :--- | :--- |
| [`sberta/tokenizer.py`](sberta/tokenizer.py) | `SBERTaTokenizer` class + `normalise()` function |
| [`train_tokenizer.py`](train_tokenizer.py) | CLI to train a new SP model from a raw corpus |
| [`scripts/clean_corpus.py`](scripts/clean_corpus.py) | Corpus pre-processor: strips URLs, HTML, mentions, hashtags, long numbers, emoji, and duplicate lines |

---

## Usage

### Basic encoding

```python
from sberta.tokenizer import SBERTaTokenizer

tok = SBERTaTokenizer("runs/tokenizer/sberta.model")

# Single string → list of token IDs (includes [SEP])
ids = tok.encode("wach rak labas?")
print(ids)
# → [4312, 7891, 3201, 914, 23, 3]

# View piece strings
print(tok.convert_ids_to_tokens(ids))
# → ['▁wach', '▁rak', '▁la', 'bas', '?', '[SEP]']
```

### Subword regularisation (for training)

```python
# Sample a random valid segmentation instead of the Viterbi path
ids = tok.encode("wach rak labas?", sample=True, sample_alpha=0.1)
```

### Batch encoding

```python
batch = tok.batch_encode(
    ["wach rak?", "أنا بخير"],
    max_length=128,
    return_tensors=True,   # returns torch.Tensors
)
# batch["input_ids"]      → Tensor(2, 128)  dtype=int64
# batch["attention_mask"] → Tensor(2, 128)  dtype=int64  (0 = padding)
```

### Sentence pairs (fine-tuning)

```python
ids, token_type_ids = tok.encode_pair("wach rak?", "labas, Hamdullah")
# ids            → A [SEP] B [SEP]  (concatenated)
# token_type_ids → 0…0 1…1         (segment A vs B)
```

### Decoding

```python
text = tok.decode(ids, skip_special_tokens=True)

texts = tok.batch_decode(batch["input_ids"], skip_special_tokens=True)
```

### Persistence

```python
# Save the underlying .model file to a directory
tok.save("my_tokenizer/")

# Reload from that directory
tok = SBERTaTokenizer.from_pretrained("my_tokenizer/")
```

---

## Training the Tokenizer

Train a new SentencePiece model from scratch on your corpus:

```bash
python train_tokenizer.py \
    --input  corpus/*.txt corpus/wikipedia/*.txt corpus/youtube/*.txt \
    --output runs/tokenizer/ \
    --vocab_size  50265 \
    --num_threads 8 \
    --min_chars   5
```

The script will:
1. Stream all input files through `normalise()` into a single temporary file.
2. Train the SentencePiece Unigram model with the parameters described above.
3. Run a built-in verification suite to confirm special token IDs and roundtrip fidelity.
4. Write `sberta.model` and `sberta.vocab` to the output directory.

Key CLI options:

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--vocab_size` | 50265 | Must match `SBERTaConfig.vocab_size` |
| `--num_threads` | 4 | Parallelism for SP training |
| `--min_chars` | 5 | Minimum characters per line after normalisation |
| `--max_lines` | — | Cap input lines (useful for fast iteration) |
| `--no_verify` | false | Skip post-training verification |
