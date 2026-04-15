# SBERTa: Code-Switching BERT with ELECTRA-Style Pre-training

A general-purpose transformer architecture for code-switched text, featuring explicit language modeling, contextual language refinement, and efficient ELECTRA-style replaced token detection.

---

## Overview

SBERTa makes language identity and language boundaries explicit architectural components rather than leaving them implicit in representations. The model learns to:

- Assign soft language distributions to each token via learnable prototypes
- Detect language boundaries through continuous switch magnitudes
- Apply language-aware attention biases using per-head compatibility matrices
- Pre-train efficiently with ELECTRA-style replaced token detection (6-7× more signal than vanilla MLM)

**Key innovation:** Fully self-supervised and language-agnostic. Works for any code-switching scenario without hardcoded language labels.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train Tokenizer

```bash
python train_tokenizer.py \
  --input corpus/*.txt \
  --output runs/tokenizer \
  --vocab-size 50265
```

### Pre-train Model

```bash
python pretrain.py \
  --config base \
  --corpus-dirs corpus \
  --domain-weights "darija=0.7,wikipedia=0.3" \
  --tokenizer-dir runs/tokenizer \
  --total-steps 150000 \
  --batch-size 256 \
  --grad-accum 2 \
  --num-workers 16 \
  --run-id sberta-rtx6000-v1
```

### Fine-tune for Classification

```python
from sberta.model import SBERTaModel
from sberta.config import SBERTaConfig
import torch.nn as nn

# Load pre-trained encoder
config = SBERTaConfig.load("runs/sberta-base/step-0100000/config.json")
encoder = SBERTaModel(config)
encoder.load_state_dict(torch.load("runs/sberta-base/step-0100000/model.pt", weights_only=True))

# Add classification head with mean pooling
class Classifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        H, p_ctx, s = self.encoder(input_ids, attention_mask)
        
        # Mean pooling over real tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (H * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1.0)
        
        return self.classifier(pooled)
```

---

## Architecture Highlights

### Two-Stage Language Distributions

- **Pre-contextual** $p^{(0)}_t$: Fast, computed from raw embeddings, used for augmentation
- **Context-refined** $p^{(\text{ctx})}_t$: Accurate, uses windowed attention, used for attention biases
- Fixes Latin-script ambiguity (e.g., "chat" = French or English?)

### Language-Aware Attention

$$\text{score}_h(i, j) = \frac{Q_i K_j^\top}{\sqrt{d_h}} + p_i^\top C_h p_j + \gamma s_j$$

- $C_h \in \mathbb{R}^{K \times K}$: Per-head language compatibility matrix (learns asymmetric affinities)
- $\gamma s_j$: Global switch-position bias (attends to boundaries)

### ELECTRA-Style Pre-training

- **Generator** (1/4 size): Proposes token replacements via MLM
- **Discriminator** (full SBERTa): Detects real vs replaced tokens
- **Switch-span masking**: Masks entire language segments, not random tokens
- **Result**: 6-7× more training signal than vanilla MLM

### Training Objectives

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + w_{\text{rtd}} \mathcal{L}_{\text{RTD}} + \lambda_{\text{smooth}} \cdot w_{\text{curr}} \cdot \mathcal{L}_{\text{smooth}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}$$

- $\mathcal{L}_{\text{gen}}$: Generator MLM on masked spans
- $\mathcal{L}_{\text{RTD}}$: Discriminator replaced token detection (all tokens, 6-7× signal)
- $\mathcal{L}_{\text{smooth}}$: Unsupervised temporal stickiness — no external labels or fastText required
- $\mathcal{L}_{\text{div}}$: Prototype diversity (prevents collapse)

---

## Model Configurations

| Config | Hidden | Layers | Heads | FFN  | Params |
|--------|--------|--------|-------|------|--------|
| Small  | 256    | 4      | 4     | 1024 | ~16M   |
| Medium | 512    | 8      | 8     | 2048 | ~51M   |
| Base   | 768    | 12     | 12    | 3072 | ~124M  |
| Large  | 1024   | 24     | 16    | 4096 | ~355M  |

---

## Use Cases

SBERTa is designed for **any code-switching scenario**:

- **Arabic/French/Arabizi** (Algerian, Moroccan, Tunisian dialects)
- **Spanish/English** (Spanglish)
- **Hindi/English** (Hinglish)
- **Multilingual social media** (mixed-language posts)

No hardcoded language labels. Prototypes discover languages from data.

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Complete mathematical specification
- **[TOKENIZER.md](TOKENIZER.md)** — Tokenizer design and training

---

## Project Structure

```
SBERTa/
├── sberta/
│   ├── model.py           # Full architecture (ELECTRA-style)
│   ├── config.py          # Configuration dataclass
│   ├── tokenizer.py       # SentencePiece wrapper
│   └── __init__.py
├── scripts/
│   └── clean_corpus.py    # Corpus cleaning (noise, dedup, hashtags)
├── corpus/                # Training data
├── pretrain.py            # Pre-training script
├── train_tokenizer.py     # Tokenizer training
├── finetune_narabizi.py   # Example fine-tuning (NArabizi sentiment)
├── ARCHITECTURE.md        # Mathematical specification
├── TOKENIZER.md           # Tokenizer documentation
└── requirements.txt
```

---

## Citation

```bibtex
@misc{sberta2026,
  title={SBERTa: Code-Switching BERT with ELECTRA-Style Pre-training},
  author={Ainouche Abderahmane},
  year={2026},
  note={General-purpose transformer for code-switched text}
}
```

---

## License

MIT License
