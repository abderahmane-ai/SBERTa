# SBERTa

**Switching BERT architecture for Algerian Darija code-switching.**

SBERTa is a transformer pre-training project for Algerian Darija: Arabic script, Arabizi, French code-switching, Tamazight-influenced forms, and noisy social-media spelling. It treats code-switching as a structural signal rather than a nuisance, but keeps the training recipe deliberately stable and low-knob.

The immediate goal is to build a Darija encoder that can beat **DziriBERT** on the same Algerian benchmarks, especially Roman-script / Arabizi tasks where DziriBERT is strongest. DziriBERT is a BERT-base style model with a 50k vocabulary, trained on roughly 1M Algerian tweets / 150 MB, and evaluated on Twifil and NArabizi ([paper](https://arxiv.org/abs/2109.12346), [GitHub](https://github.com/alger-ia/dziribert), [Hugging Face](https://huggingface.co/alger-ia/dziribert)).

---

## What Makes It Different

Most multilingual models see Algerian code-switching as noise: script alternation, French insertions, Arabizi digit-phonemes, and inconsistent spelling all get flattened into generic multilingual capacity. SBERTa makes the structure explicit while still learning it from data.

**Two-Phase Encoder** — Phase 1 is standard self-attention and builds contextual representations without language bias. Language prototypes are assigned after this context exists. Phase 2 may use language-aware attention, but only after the training schedule allows it.

**Darija Prototype States** — The default recipe uses `K=4` soft prototype states, intended for the dominant modes in Algerian text: Arabic-script Darija/MSA-like text, Arabizi/Roman script, French, and other borrowed/local forms. These are soft latent states, not supervised labels.

**Sinkhorn-Knopp with Adaptive Prior** — Sinkhorn prevents prototype collapse, while the EMA prior tracks raw model probabilities instead of forcing a rigid uniform split. This keeps the clustering pressure useful without letting the prior simply echo Sinkhorn’s own constrained output.

**Language-Aware Attention** — Phase 2 layers contain per-head compatibility matrices and pairwise divergence terms. Their contribution is ramped in during training, so the model starts as a stable transformer and only later uses code-switching structure.

**GDES + ELECTRA-Style Pre-Training** — A generator proposes replacements and a discriminator detects them. Discriminator gradients do not update token embeddings, following the DeBERTaV3/ELECTRA stability lesson that RTD can otherwise distort shared embeddings.

---

## Architecture

Full mathematical and training specification: [ARCHITECTURE.md](ARCHITECTURE.md).

Short version:

- **Embeddings:** token + position embeddings only; no script IDs or dictionaries.
- **Phase 1:** standard Pre-LN self-attention.
- **Pivot:** contextual states are mapped to soft prototype distributions.
- **Phase 2:** language-aware attention can use prototype compatibility and pairwise language divergence.
- **Pre-training:** generator MLM + discriminator RTD + scheduled clustering and orthogonality losses.
- **Pooling:** no `[CLS]`; sentence representations use mean pooling over real tokens.

---

## Stable Training Recipe

The public CLI intentionally exposes only high-ROI controls:

```bash
python pretrain.py \
  --preset darija-medium \
  --corpus-dirs corpus \
  --tokenizer-dir runs/tokenizer \
  --total-tokens 250000000 \
  --micro-batch-size 16 \
  --grad-accum 4 \
  --run-id darija-medium-v1
```

Everything else is fixed inside the preset: optimizer, warmup, dropout, span masking, Sinkhorn settings, prototype temperature, loss weights, clipping, logging, and checkpoint cadence.

Training is staged:

1. **Phase A:** generator MLM + RTD only.
2. **Phase B:** prototype clustering and orthogonality ramp in.
3. **Phase C:** language-aware attention ramps in once prototype entropy is healthy.

The trainer logs RTD precision/recall/F1, replacement rate, prototype entropy, EMA prior, switch magnitude, gradient norm, AMP skips, and loss components.

---

## Presets

| Preset | Hidden | Layers | Heads | FFN | K | Use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `darija-small` | 256 | 4 | 4 | 1024 | 4 | Smoke tests and quick experiments |
| `darija-medium` | 512 | 8 | 8 | 2048 | 4 | Recommended single-GPU run |
| `darija-base` | 768 | 12 | 12 | 3072 | 4 | DziriBERT-scale comparison |

---

## Data And Tokenizer

Tokenizer reference: [TOKENIZER.md](TOKENIZER.md).  
Data policy and manifest format: [DATA.md](DATA.md).

Train the tokenizer:

```bash
python train_tokenizer.py \
  --input corpus/darija_pretrain.txt \
  --output runs/tokenizer \
  --vocab_size 50000
```

Prepare a corpus source and append manifest provenance:

```bash
python scripts/prepare_corpus.py \
  --input data/raw/youtube \
  --output corpus/darija_pretrain.txt \
  --source-name youtube_algerian_channels \
  --license-status "YouTube API, local research use" \
  --usage pretraining
```

Target for v1: at least 500 MB of cleaned Algerian-centric text, ideally 1-2 GB, with Roman-script / Arabizi text kept above 25%.

---

## Benchmark Gate

Benchmark protocol: [BENCHMARKS.md](BENCHMARKS.md).

SBERTa v1 is successful only if it:

- beats DziriBERT on NArabizi sentiment/topic,
- is not worse on Twifil sentiment/emotion,
- reports both accuracy and macro-F1,
- uses identical splits and seeds for DziriBERT and SBERTa.

Example:

```bash
python scripts/evaluate_benchmarks.py \
  --task narabizi_sentiment \
  --model dziribert \
  --train data/narabizi/train.csv \
  --dev data/narabizi/dev.csv \
  --test data/narabizi/test.csv \
  --output runs/benchmarks/dziribert_narabizi_sentiment.json
```

Use `--model sberta --sberta-checkpoint runs/darija-medium-v1/latest` for SBERTa.

---

## Project Structure

```text
sberta/
    model.py        architecture and pre-training wrapper
    config.py       Darija presets and fixed recipe constants
    tokenizer.py    SentencePiece wrapper and normaliser
pretrain.py         stable Darija pre-training loop
train_tokenizer.py  SentencePiece Unigram training
test.py             synthetic and stability tests
scripts/
    prepare_corpus.py
    evaluate_benchmarks.py
DATA.md             data sources, manifest, usage policy
BENCHMARKS.md       DziriBERT comparison protocol
ARCHITECTURE.md     architecture specification
TOKENIZER.md        tokenizer reference
```

---

## References

- Clark et al. (2020) — ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
- He et al. (2021) — DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing
- Caron et al. (2020) — SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
- Kudo & Richardson (2018) — SentencePiece: A simple and language independent subword tokenizer
- Abdaoui et al. (2021) — DziriBERT: a pre-trained language model for the Algerian dialect

---

## License

MIT
