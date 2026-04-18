# SBERTa

**Switching BERT architecture for code-switched text.**

SBERTa is a transformer pre-trained specifically for code-switching — text that mixes languages within a sentence or document. It makes language identity and language boundaries explicit architectural components, learned end-to-end without any external labels, language detection tools, dictionaries, or Unicode script hints.

While originally designed for Algerian Darija (Arabic script, Latin script, French loanwords, dialect Arabic), the architecture is now a **universal, zero-knowledge standard** capable of unsupervised language boundary discovery in any code-switched mixture (e.g., Spanglish, Hinglish, Franglais).

---

## What makes it different

Standard multilingual models treat code-switching as noise. SBERTa treats it as a structural signal.

**Language Prototypes & Sinkhorn-Knopp** — The model learns K prototype vectors representing language directions in embedding space. To prevent the prototypes from collapsing into a single language without relying on hardcoded rules, SBERTa uses **Optimal Transport (Sinkhorn-Knopp)**. This mathematically guarantees that the model discovers exactly K distinct linguistic clusters from the data distribution itself.

**Zero-Knowledge Clustering** — SBERTa requires no Unicode priors or dictionaries. Language structure emerges purely from the Masked Language Modeling (MLM) objective, stabilized by the Sinkhorn clustering loss. It works on multi-script and mono-script mixtures equally well.

**Switch Magnitudes** — A continuous scalar per token measuring how much the language distribution changed from the previous token. Zero means same language, one means complete switch. Fully differentiable.

**Language-Aware Attention** — Each attention head has a K×K compatibility matrix that learns asymmetric language affinities. An Arabic query attending to a Latin key gets a different bias than the reverse. The switch magnitude adds a second bias toward boundary positions.

**GDES & ELECTRA-style Pre-training** — A small generator proposes token replacements via geometric span masking; the full discriminator detects them at every real token position. SBERTa solves the notorious ELECTRA instability via **Gradient-Disentangled Embedding Sharing (GDES)**. The discriminator's gradients do not flow into the token embeddings, ending the gradient tug-of-war and allowing the generator to build clean semantic clusters for the Sinkhorn algorithm to discover.

**Temporal Stickiness** — An unsupervised loss penalizes high switch magnitudes at within-sequence boundaries, acting as a Markov prior that pushes prototypes toward long, linguistically coherent spans.

---

## Architecture

Full mathematical specification in [ARCHITECTURE.md](ARCHITECTURE.md).

The short version:

- **Embeddings:** GDES. Token embeddings are shaped purely by the Generator.
- **Generator Masking:** Standard geometric span masking independent of language predictions.
- **Language Discovery:** p⁽⁰⁾ is computed from tok+pos embeddings. Sinkhorn-Knopp balances the assignments, and a clustering loss aligns p⁽⁰⁾ with the balanced targets.
- **Attention:** Language-aware attention routes information based on the discovered clusters.
- **Pooling:** No [CLS] token. Sentence representations use mean pooling over real token positions.

---

## Configurations

| Config | Hidden | Layers | Heads | FFN  | Params |
|--------|--------|--------|-------|------|--------|
| Small  | 256    | 4      | 4     | 1024 | ~16M   |
| Medium | 512    | 8      | 8     | 2048 | ~51M   |
| Base   | 768    | 12     | 12    | 3072 | ~124M  |
| Large  | 1024   | 24     | 16    | 4096 | ~355M  |

---

## Fine-tuning

`finetune_narabizi.py` fine-tunes a pre-trained SBERTa encoder on the [NArabizi dataset](https://github.com/SamiaTouileb/NArabizi) (Touileb & Barnes, ACL 2021 Findings) for sentiment classification (NEG/NEU/POS) and topic classification (Religion/Societal/Sport/NONE).

Baseline to beat: DziriBERT sentiment accuracy 80.5%.

---

## Project structure

```
sberta/
    model.py        — full architecture
    config.py       — configuration dataclass
    tokenizer.py    — SentencePiece wrapper
corpus/             — training data
finetune_narabizi.py
ARCHITECTURE.md     — mathematical specification
```

---

## References

- Clark et al. (2020) — ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
- He et al. (2021) — DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing
- Caron et al. (2020) — SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (Sinkhorn-Knopp)
- Kudo & Richardson (2018) — SentencePiece: A simple and language independent subword tokenizer

---

## License

MIT
