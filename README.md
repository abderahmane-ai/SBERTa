# SBERTa

**Switching BERT architecture for code-switched text.**

SBERTa is a transformer pre-trained specifically for code-switching — text that mixes languages within a sentence or document. It makes language identity and language boundaries explicit architectural components, learned end-to-end without any external labels, language detection tools, dictionaries, or Unicode script hints.

While originally designed for Algerian Darija (Arabic script, Latin script, French loanwords, dialect Arabic), the architecture is now a **universal, zero-knowledge standard** capable of unsupervised language boundary discovery in any code-switched mixture (e.g., Spanglish, Hinglish, Franglais).

---

## What makes it different

Standard multilingual models treat code-switching as noise. SBERTa treats it as a structural signal.

**Two-Phase Encoder** — SBERTa avoids the "bootstrap paradox" of early clustering. Phase 1 uses standard attention to build semantic context. Language prototypes are assigned from these contextual outputs at a pivot layer, before Phase 2 applies language-aware attention.

**Language Prototypes & Sinkhorn-Knopp** — The model learns K prototype vectors. To prevent collapse, SBERTa uses **Optimal Transport (Sinkhorn-Knopp)**. Crucially, it uses an **Adaptive EMA Prior** to track the batch marginals, allowing the model to dynamically discover the true language distribution of the corpus without forcing a 50/50 split.

**Zero-Knowledge Clustering** — SBERTa requires no Unicode priors or dictionaries. Language structure emerges purely from the Masked Language Modeling (MLM) objective, stabilized by the Sinkhorn clustering loss.

**Language-Aware Attention** — Each attention head in Phase 2 has a K×K compatibility matrix that learns asymmetric language affinities. An Arabic query attending to a Latin key gets a different bias than the reverse.

**GDES & ELECTRA-style Pre-training** — A small generator proposes token replacements via geometric span masking; the discriminator detects them. SBERTa solves ELECTRA instability via **Gradient-Disentangled Embedding Sharing (GDES)**. The discriminator's gradients do not flow into the token embeddings, allowing the generator to build clean semantic clusters for the Sinkhorn algorithm to discover.

---

## Architecture

Full mathematical specification in [ARCHITECTURE.md](ARCHITECTURE.md).

The short version:

- **Two-Phase Encoder:** Phase 1 builds language-agnostic contextual representations. The language pivot assigns prototypes based on context. Phase 2 applies language-aware attention.
- **Embeddings:** GDES. Token embeddings are shaped purely by the Generator.
- **Generator Masking:** Standard geometric span masking independent of language predictions.
- **Language Discovery:** p is computed from Phase 1 contextual outputs. Sinkhorn-Knopp clusters them using an adaptive EMA prior that discovers the true corpus language distribution online.
- **Attention:** Phase 2 routes information based on the discovered clusters via per-head K×K compatibility matrices.
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