# SBERTa: Switch-aware BERT for Code-Switched Text

A transformer architecture for code-switched text, featuring explicit language modeling through learnable prototypes, language-aware attention, and efficient ELECTRA-style pre-training.

---

## Overview

SBERTa makes language identity and code-switching boundaries explicit architectural components. The model learns to:

- Assign soft language distributions to each token via **K learnable prototype vectors**
- Detect language boundaries through **continuous switch magnitudes**
- Apply **language-aware attention biases** using per-head compatibility matrices
- Pre-train efficiently with **ELECTRA-style replaced token detection** (6-7× more signal than vanilla MLM)

**Key innovation:** Fully self-supervised and language-agnostic. No hardcoded language labels, no external tools (fastText, langid). Prototypes discover languages from data.

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
  --tokenizer-dir runs/tokenizer \
  --total-steps 150000 \
  --warmup-steps 10000 \
  --batch-size 32 \
  --grad-accum 4 \
  --lr 1e-4 \
  --checkpoint-every 5000 \
  --log-every 100 \
  --num-workers 16 \
  --run-id sberta-base-150k
```

**Training time:** ~24-30 hours on RTX 6000 Ada (effective batch size 128)

### Fine-tune for Classification

```python
from sberta.model import SBERTaModel
from sberta.config import SBERTaConfig
from sberta.tokenizer import SBERTaTokenizer
import torch
import torch.nn as nn

# Load pre-trained encoder
config = SBERTaConfig.load("runs/sberta-base-150k/step-0150000/config.json")
encoder = SBERTaModel(config)
encoder.load_state_dict(
    torch.load("runs/sberta-base-150k/step-0150000/model.pt", 
               map_location="cpu", weights_only=True),
    strict=False  # Skip pre-training-only components
)

# Add classification head with mean pooling
class SBERTaClassifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        H, p0, s = self.encoder(input_ids, attention_mask)
        
        # Mean pooling over real tokens (no [CLS] token in SBERTa)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (H * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1.0)
        
        return self.classifier(pooled)

# Initialize classifier
model = SBERTaClassifier(encoder, num_labels=3)

# Fine-tune with standard cross-entropy loss
# See finetune_narabizi.py for complete example
```

---

## Architecture Highlights

### Language Prototypes

**K learnable prototype vectors** $L \in \mathbb{R}^{K \times d}$ with learnable temperature $\tau$:

$$p^{(0)}_t = \text{softmax}\left(\frac{h_{\text{base},t} \cdot L^\top}{\tau}\right)$$

- Computed from raw embeddings (tok + pos, before augmentation)
- Used for embedding augmentation and attention biases
- Prototypes learn language clusters endogenously from data
- No external labels or language detection tools required

### Switch Magnitudes

Continuous measure of language boundary strength:

$$s_t = 1 - p^{(0)}_t \cdot p^{(0)}_{t-1}, \quad s_1 = 0$$

- $s_t \in [0, 1]$: 0 = same language, 1 = complete switch
- Fully differentiable, learned end-to-end
- Used in attention biases and temporal smoothness loss

### Language-Aware Attention

$$\text{score}_h(i, j) = \frac{Q_i K_j^\top}{\sqrt{d_h}} + p_i^\top C_h p_j + \gamma \cdot s_j$$

- $C_h \in \mathbb{R}^{K \times K}$: Per-head language compatibility matrix
  - Learns asymmetric affinities (e.g., Arabic↔Arabizi high, French↔Arabic medium)
  - Initialized to identity + small noise for head specialization
- $\gamma \cdot s_j$: Global switch-position bias (attends to boundaries)

### ELECTRA-Style Pre-training

- **Generator** (hidden_size / 2): Proposes token replacements via MLM
- **Discriminator** (full SBERTa): Detects real vs replaced tokens at **all positions**
- **Switch-span masking**: Masks entire language-homogeneous segments
- **Result**: 6-7× more training signal than vanilla 15%-masked MLM

### Training Objectives

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + w_{\text{rtd}} \mathcal{L}_{\text{RTD}} + \lambda_{\text{smooth}} \cdot w_{\text{curr}} \cdot \mathcal{L}_{\text{smooth}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_{\text{balance}} \mathcal{L}_{\text{balance}}$$

- $\mathcal{L}_{\text{gen}}$: Generator MLM on masked spans (normalized by n_masked)
- $\mathcal{L}_{\text{RTD}}$: Discriminator BCE at all real token positions
- $\mathcal{L}_{\text{smooth}}$: Unsupervised temporal stickiness (mean switch magnitude)
  - Two-phase curriculum: burn-in (5% steps, L_div only) → warmup (15% steps, ramp 0.05→1.0)
  - No external labels required
- $\mathcal{L}_{\text{div}}$: Exponential prototype repulsion (prevents collapse)
- $\mathcal{L}_{\text{balance}}$: Soft minimum-usage with EMA tracking (rescues dying prototypes)

**Default hyperparameters:**
- $w_{\text{rtd}} = 50.0$, $\lambda_{\text{smooth}} = 5.0$, $\lambda_{\text{div}} = 5.0$, $\lambda_{\text{balance}} = 30.0$
- Burn-in: 5% of total steps, Warmup: 15% of total steps

---

## Model Configurations

| Config | Hidden | Layers | Heads | FFN  | Params | Use Case |
|--------|--------|--------|-------|------|--------|----------|
| Small  | 256    | 4      | 4     | 1024 | ~16M   | Quick experiments |
| Medium | 512    | 8      | 8     | 2048 | ~51M   | Low-resource |
| Base   | 768    | 12     | 12    | 3072 | ~124M  | Standard (recommended) |
| Large  | 1024   | 24     | 16    | 4096 | ~355M  | High-resource |

---

## Use Cases

SBERTa is designed for **any code-switching scenario**:

- **Arabic/French/Arabizi** (Algerian, Moroccan, Tunisian dialects)
- **Spanish/English** (Spanglish)
- **Hindi/English** (Hinglish)
- **Multilingual social media** (mixed-language posts)
- **Any language pair** with sufficient training data

No hardcoded language labels. Set `num_languages=K` based on your corpus (typically K=3-5).

---

## Training Tips

### Corpus Requirements

- **Size**: Minimum 50M tokens, recommended 500M+ tokens
- **Format**: Plain text, UTF-8, one sentence/segment per line
- **Quality**: Remove noise, duplicates, and non-linguistic content
- **Balance**: Natural distribution is fine; EMA balance loss handles heterogeneity

### Hyperparameter Tuning

**If prototypes collapse (entropy < 80%):**
- Increase burn-in: `burnin_ratio = 0.10` (10% of steps)
- Reduce smooth loss: `lambda_smooth = 3.0` (for high code-switching corpora)
- Increase diversity: `lambda_div = 10.0`

**If training is unstable:**
- Reduce learning rate: `lr = 5e-5`
- Increase warmup: `warmup_steps = 15000`
- Reduce batch size: `batch_size = 16`

**For code-switched corpora (>30% mixed sentences):**
- Use `lambda_smooth = 3.0-5.0` (lower than monolingual default)
- Increase warmup: `smooth_warmup_ratio = 0.20`

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
│   ├── clean_and_merge_corpus.py  # Corpus cleaning
│   └── extract_text_from_json.py  # JSON extraction
├── corpus/                # Training data
│   └── darija_corpus_clean.txt
├── pretrain.py            # Pre-training script
├── train_tokenizer.py     # Tokenizer training
├── finetune_narabizi.py   # Example fine-tuning (NArabizi sentiment)
├── notebooks/
│   └── training.ipynb     # Kaggle training notebook
├── ARCHITECTURE.md        # Mathematical specification
├── TOKENIZER.md           # Tokenizer documentation
└── requirements.txt
```

---

## Citation

```bibtex
@misc{sberta2025,
  title={SBERTa: Switch-aware BERT for Code-Switched Text},
  author={Ainouche, Abderahmane},
  year={2025},
  note={Transformer architecture with explicit language modeling for code-switching}
}
```

---

## License

MIT License

---

## Acknowledgments

- ELECTRA architecture (Clark et al., 2020) for efficient pre-training
- SentencePiece (Kudo & Richardson, 2018) for subword tokenization
- Algerian Darija community for corpus contributions
