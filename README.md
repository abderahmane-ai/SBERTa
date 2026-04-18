# SBERTa

**Switching BERT architecture for code-switched text.**

SBERTa is a transformer pre-trained specifically for code-switching — text that mixes languages within a sentence or document. It makes language identity and language boundaries explicit architectural components, learned end-to-end without any external labels or language detection tools.

The primary target is Algerian Darija: Arabic script, Latin script (Arabizi), French loanwords, and dialect Arabic, all mixed freely in the same sentence.

---

## What makes it different

Standard multilingual models treat code-switching as noise. SBERTa treats it as signal.

**Language prototypes** — the model learns K prototype vectors representing language directions in embedding space. Every token gets a soft probability distribution over these prototypes, computed before any contextual processing. A Unicode script prior (Arabic = prototype 0, Latin = prototype 1) grounds these distributions in objective orthographic evidence from step one, so the model doesn't start from random noise.

**Switch magnitudes** — a continuous scalar per token measuring how much the language distribution changed from the previous token. Zero means same language, one means complete switch. Fully differentiable, no thresholds.

**Language-aware attention** — each attention head has a K×K compatibility matrix that learns asymmetric language affinities. An Arabic query attending to a Latin key gets a different bias than the reverse. The switch magnitude adds a second bias toward boundary positions.

**ELECTRA-style pre-training** — a small generator proposes token replacements; the full discriminator detects them at every real token position. This gives 6–7× more gradient signal than vanilla MLM for the same data budget, which matters for low-resource languages. Masking targets entire language-homogeneous spans rather than random tokens, forcing the generator to reconstruct full language segments from cross-language context.

**Temporal stickiness** — an unsupervised loss penalises high switch magnitudes at within-script boundaries, pushing prototypes toward long linguistically coherent spans. Cross-script boundaries (Arabic → Latin) are excluded from this penalty because they are genuine switches, not ambiguity.

---

## Architecture

Full mathematical specification in [ARCHITECTURE.md](ARCHITECTURE.md).

The short version:

- Pre-contextual language distributions computed once from raw (tok + pos) embeddings, blended with the Unicode script prior, then used everywhere: span masking, embedding augmentation, attention biases. No circular dependency, no separate refinement stage.
- Post-LayerNorm encoder stack. The self-attention layers resolve all remaining ambiguity — loanwords, numerals, punctuation, Arabizi — through full T×T attention.
- No [CLS] token. Sentence representations use mean pooling over real token positions.
- K=2 for the Arabic/Latin use case. The architecture supports arbitrary K.

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
- Kudo & Richardson (2018) — SentencePiece: A simple and language independent subword tokenizer
- Touileb & Barnes (2021) — NArabizi: A Treebank for Algerian Arabizi

---

## License

MIT
