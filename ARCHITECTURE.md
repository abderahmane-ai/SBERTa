# SBERTa: Architecture Specification

**SBERTa** (**S**witching **B**idirectional **E**ncoder **R**epresentations from **T**ransformers **a**rchitecture) is a transformer architecture for code-switched text featuring explicit unsupervised language modeling and ELECTRA-style adversarial pre-training.

---

## 1. Core Design Philosophy

SBERTa makes three linguistic phenomena explicit architectural components:

1. **Language identity** — soft distributions over $K$ language prototypes
2. **Language boundaries** — continuous switch magnitudes between consecutive tokens
3. **Language interactions** — per-head compatibility matrices governing cross-language attention

All components are learned end-to-end without hardcoded language labels, making the architecture applicable to any code-switching scenario.

---

## 2. Language Prototypes

### 2.1 Prototype Vectors

The model learns $K$ prototype vectors $\mathbf{L} = [\boldsymbol{\ell}_1, \ldots, \boldsymbol{\ell}_K] \in \mathbb{R}^{K \times d}$ representing language directions in embedding space.

**Initialization:** Orthogonal initialization scaled by 0.5 to avoid softmax saturation:
$$\mathbf{L} \sim \text{Orthogonal}(\mathbb{R}^{K \times d}), \quad \mathbf{L} \leftarrow 0.5 \cdot \mathbf{L}$$

### 2.2 Pre-Contextual Language Distributions

For each token $t$, compute a soft distribution over prototypes from raw base embeddings $\mathbf{h}_{\text{base},t} = \mathbf{E}_{\text{tok}}(x_t) + \mathbf{E}_{\text{pos}}(t)$:

$$p^{(0)}_t = \text{softmax}\left(\frac{\mathbf{h}_{\text{base},t} \mathbf{L}^\top}{\tau}\right) \in \Delta^K$$

where $\tau$ is a learnable temperature parameter stored as $\log \tau$ for unconstrained optimization:
$$\tau = \exp(\log \tau), \quad \log \tau \in \mathbb{R}$$

**Properties:**
- Computed before contextual processing (no circular dependency)
- Fast but potentially ambiguous for Latin-script tokens
- Used for embedding augmentation

### 2.3 Switch Magnitudes

Continuous measure of language change between consecutive tokens:

$$s_t = 1 - p^{(0)}_t{}^\top p^{(0)}_{t-1}, \quad s_1 = 0$$

Since $p^{(0)}_t \in \Delta^K$, the inner product $p^{(0)}_t{}^\top p^{(0)}_{t-1} \in [0, 1]$, thus $s_t \in [0, 1]$.

**Interpretation:**
- $s_t \approx 0$: Same language as previous token
- $s_t \approx 1$: Complete language switch
- $s_t \in (0, 1)$: Partial or gradual transition

---

## 3. Contextual Language Refinement

### 3.1 Motivation

Pre-contextual distributions $p^{(0)}_t$ fail for Latin-script ambiguity:
- "chat" — French (*cat*) or English (*to chat*)?
- "la" — French article or Arabic لا?
- "est" — French or Spanish copula?

Context is required to disambiguate.

### 3.2 Windowed Self-Attention

A lightweight module computes context-refined distributions $p^{(\text{ctx})}_t$ via windowed self-attention over $\mathbf{h}_{\text{base}}$:

$$\mathbf{Q} = \mathbf{W}_Q \mathbf{h}_{\text{base}}, \quad \mathbf{K} = \mathbf{W}_K \mathbf{h}_{\text{base}} \quad \text{where } \mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{(d/4) \times d}$$

$$\text{scores}_{ij} = \frac{\mathbf{Q}_i \mathbf{K}_j^\top}{\sqrt{d/4}} + \text{mask}_{\text{window}}(i, j)$$

where $\text{mask}_{\text{window}}(i, j) = 0$ if $|i - j| \leq w$ (default $w=3$), else $-\infty$.

$$\mathbf{ctx}_t = \sum_{j} \text{softmax}(\text{scores}_{t,:})_j \, \mathbf{h}_{\text{base},j}$$

$$p^{(\text{ctx})}_t = \text{softmax}\left(\frac{\mathbf{W}_{\text{proj}} \mathbf{ctx}_t}{\tau}\right) \in \Delta^K$$

**Properties:**
- Small projection ($d/4$) keeps computational cost minimal
- Window size $w=3$ provides local context
- Shares temperature $\tau$ with pre-contextual distributions

### 3.3 Two-Stage Usage

- **Stage 1 (augmentation):** Use $p^{(0)}_t$ to avoid circular dependency
- **Stage 2 (attention):** Use $p^{(\text{ctx})}_t$ for accurate language-aware biases

---

## 4. Language-Augmented Embeddings

### 4.1 Base Embeddings

$$\mathbf{h}_{\text{base},t} = \mathbf{E}_{\text{tok}}(x_t) + \mathbf{E}_{\text{pos}}(t)$$

No normalization or augmentation yet. Used to compute $p^{(0)}_t$ and $s_t$.

### 4.2 Augmentation

$$\mathbf{h}^{(0)}_t = \text{LayerNorm}\left(\mathbf{h}_{\text{base},t} + \sum_{k=1}^K p^{(0)}_{t,k} \, \mathbf{e}_k + s_t \, \mathbf{e}_{\text{sw}}\right)$$

where:
- $\mathbf{e}_k \in \mathbb{R}^d$ are learnable language embeddings (one per prototype)
- $\mathbf{e}_{\text{sw}} \in \mathbb{R}^d$ is a learnable switch embedding

**Interpretation:**
- $\sum_k p^{(0)}_{t,k} \mathbf{e}_k$: Soft language signal (weighted by distribution)
- $s_t \mathbf{e}_{\text{sw}}$: Boundary signal (scaled by switch magnitude)

---

## 5. Language-Aware Attention

### 5.1 Attention Score Computation

Standard transformer attention with two additive language-aware biases:

$$\text{score}_h(i, j) = \frac{\mathbf{Q}_{h,i} \mathbf{K}_{h,j}^\top}{\sqrt{d_h}} + p^{(\text{ctx})}_i{}^\top \mathbf{C}_h \, p^{(\text{ctx})}_j + \gamma \, s_j$$

where:
- $h \in \{1, \ldots, H\}$ indexes attention heads
- $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ is a per-head language compatibility matrix
- $\gamma \in \mathbb{R}$ is a global switch-position bias (shared across heads)

### 5.2 Language Compatibility Bias

$$p^{(\text{ctx})}_i{}^\top \mathbf{C}_h \, p^{(\text{ctx})}_j = \sum_{k=1}^K \sum_{\ell=1}^K p^{(\text{ctx})}_{i,k} \, [\mathbf{C}_h]_{k\ell} \, p^{(\text{ctx})}_{j,\ell}$$

**Properties:**
- Asymmetric: $[\mathbf{C}_h]_{k\ell} \neq [\mathbf{C}_h]_{\ell k}$ in general
- Learns language-specific affinities (e.g., Arabic $\leftrightarrow$ Arabizi high, French $\leftrightarrow$ Arabic medium)
- Initialized as identity: $\mathbf{C}_h = \mathbf{I}_K$ (equivalent to scalar $\beta_h = 1$)
- Only $K^2 H$ parameters (192 for base config with $K=4$, $H=12$)

### 5.3 Switch Position Bias

$$\gamma \, s_j$$

**Properties:**
- Encourages attention toward language boundaries when $\gamma > 0$
- Initialized to zero (model learns if useful)
- Shared across all heads

### 5.4 Fixed Language Distributions

Both $p^{(\text{ctx})}_t$ and $s_t$ are computed once and remain fixed across all encoder layers. They are threaded through the encoder but not updated layer-by-layer.

---

## 6. Encoder Architecture

### 6.1 Layer Structure

Each encoder layer $\ell \in \{1, \ldots, L\}$ applies:

$$\mathbf{H}_{\text{attn}}^{(\ell)} = \text{MultiHeadAttention}(\mathbf{H}^{(\ell-1)}, p^{(\text{ctx})}, s)$$

$$\mathbf{H}_{\text{mid}}^{(\ell)} = \text{LayerNorm}(\mathbf{H}^{(\ell-1)} + \text{Dropout}(\mathbf{H}_{\text{attn}}^{(\ell)}))$$

$$\mathbf{H}^{(\ell)} = \text{LayerNorm}(\mathbf{H}_{\text{mid}}^{(\ell)} + \text{Dropout}(\text{FFN}(\mathbf{H}_{\text{mid}}^{(\ell)})))$$

where $\mathbf{H}^{(0)} = [\mathbf{h}^{(0)}_1, \ldots, \mathbf{h}^{(0)}_T]$ are the augmented embeddings.

### 6.2 Post-LayerNorm

SBERTa uses post-LayerNorm (LayerNorm after residual) for stability during ELECTRA-style training.

---

## 7. ELECTRA-Style Pre-training

### 7.1 Architecture Overview

**Generator** (small):
- Hidden size: $d_{\text{gen}} = d / 4$
- Attention heads: $H_{\text{gen}} = H / 4$
- Layers: $L_{\text{gen}} = L / 4$
- Shares token embeddings with discriminator

**Discriminator** (full SBERTa):
- Full architecture ($d$, $H$, $L$)
- Binary classification head: $\mathbb{R}^d \to \mathbb{R}$

### 7.2 Switch-Span Masking

Instead of random 15% token masking, SBERTa masks entire language-homogeneous spans:

1. Compute dominant language per token: $\ell_t = \arg\max_k p^{(0)}_{t,k}$
2. Identify span boundaries where $\ell_t \neq \ell_{t-1}$
3. Randomly select whole spans until $\approx 15\%$ coverage

**Rationale:** Forces generator to reconstruct language segments from cross-language context, directly targeting code-switching.

### 7.3 Training Flow

1. Compute $p^{(0)}_t$ from original (unmasked) input
2. Apply switch-span masking using $p^{(0)}_t$
3. Generator proposes replacements for masked positions
4. Discriminator receives corrupted sequence (original + replacements)
5. Discriminator classifies each token as real or replaced

---

## 8. Training Objectives

### 8.1 Generator Loss

Masked language modeling on masked spans only:

$$\mathcal{L}_{\text{gen}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P_{\text{gen}}(x_t \mid \mathbf{x}_{\setminus \mathcal{M}})$$

where $\mathcal{M}$ is the set of masked positions.

### 8.2 Discriminator Loss (RTD)

Binary cross-entropy at every real (non-padding) token:

$$\mathcal{L}_{\text{RTD}} = -\frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \left[ y_t \log \sigma(f_{\text{disc}}(\mathbf{h}_t)) + (1 - y_t) \log(1 - \sigma(f_{\text{disc}}(\mathbf{h}_t))) \right]$$

where:
- $\mathcal{R}$ is the set of real (non-padding) positions
- $y_t = 1$ if token $t$ was replaced, $0$ if original
- $f_{\text{disc}}: \mathbb{R}^d \to \mathbb{R}$ is a linear discriminator head

**Key advantage:** Supervises all $T$ positions, not just $\approx 0.15T$ masked positions. Provides 6-7× more gradient signal than vanilla MLM.

### 8.3 Temporal Stickiness Loss (Unsupervised)

Penalises the mean switch magnitude over real consecutive token boundaries:

$$\mathcal{L}_{\text{smooth}} = \frac{1}{|\mathcal{B}|} \sum_{(t-1,\,t) \in \mathcal{B}} s_t$$

where $\mathcal{B}$ is the set of consecutive real-token boundary pairs (padding boundaries excluded).

**Curriculum schedule:** Weighted by $w_{\text{curr}} \in [0.05, 1]$ that starts at $\lambda_{\min}=0.05$ immediately at step 0 and ramps linearly to 1.0 over `smooth_warmup_steps`. The non-zero baseline prevents the backbone from settling into an independent-sampling equilibrium during a blind warmup window.

**Purpose:** Forces prototypes to self-organise into long, linguistically coherent spans rather than flipping per-token. No external labels or fastText model required — the model discovers language boundaries purely from the temporal structure of the data.

### 8.4 Per-Token Prototype Commitment Loss ($\mathcal{L}_{\text{sharp}}$)

Minimise per-token assignment entropy over real (non-pad) tokens:

$$\mathcal{L}_{\text{sharp}} = -\mathbb{E}_t\left[\sum_{k=1}^{K} p^{(0)}_{t,k} \log p^{(0)}_{t,k}\right]$$

**Purpose:** Forces each token to commit sharply to one prototype rather than hedging probability mass uniformly across $K$ languages. Only the per-token entropy term is used — the batch-mean entropy (DINO-style balance term) is intentionally omitted to preserve the natural 65/35 Darija-French corpus imbalance.

### 8.5 Prototype Diversity Loss

Minimize pairwise cosine similarities to prevent collapse:

$$\mathcal{L}_{\text{div}} = \frac{2}{K(K-1)} \sum_{i=1}^{K-1} \sum_{j=i+1}^K \left(\frac{\boldsymbol{\ell}_i^\top \boldsymbol{\ell}_j}{\|\boldsymbol{\ell}_i\| \|\boldsymbol{\ell}_j\|}\right)^2$$

**Purpose:** Ensures prototypes remain geometrically separated. Without this, prototypes can collapse to identical vectors.

### 8.5 Combined Loss

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + w_{\text{rtd}} \mathcal{L}_{\text{RTD}} + \lambda_{\text{smooth}} \cdot w_{\text{curr}} \cdot \mathcal{L}_{\text{smooth}} + \lambda_{\text{sharp}} \mathcal{L}_{\text{sharp}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}$$

**Default weights:**
- $w_{\text{rtd}} = 50.0$ (ELECTRA standard)
- $\lambda_{\text{smooth}} = 15.0$ (15:50 ratio vs RTD — sufficient to compete)
- $\lambda_{\text{sharp}} = 1.0$ (per-token entropy penalty)
- $\lambda_{\text{div}} = 0.1$
- $w_{\text{curr}} = 0.05 + 0.95 \times \min(1,\, \text{step} / \texttt{smooth\_warmup\_steps})$, ramps from 0.05 → 1.0 over `smooth_warmup_steps` (default 5,000)

---

## 9. Gradient Flow Analysis

### 9.1 What Updates Prototypes

**Direct gradients:**
- $\mathcal{L}_{\text{smooth}}$: Through `get_switch_magnitudes()` → through `get_distributions()` → through $\mathbf{L}$ directly
- $\mathcal{L}_{\text{div}}$: Direct regularization on $\mathbf{L}$

**Indirect gradients:**
- $\mathcal{L}_{\text{gen}}$: Through embedding augmentation $\sum_k p^{(0)}_{t,k} \mathbf{e}_k$
- $\mathcal{L}_{\text{RTD}}$: Through embedding augmentation and attention biases

### 9.2 Key Insight

$\mathcal{L}_{\text{smooth}}$ and $\mathcal{L}_{\text{RTD}}$ apply opposing pressures on the prototypes: RTD rewards semantic discriminability per token while $\mathcal{L}_{\text{smooth}}$ rewards long same-language spans. The tension between these forces drives prototypes toward linguistically meaningful, temporally-sticky clusters — without any external labels.

---

## 10. Sentence-Level Representations

SBERTa does not use a [CLS] token. For sentence-level tasks, use **mean pooling**:

$$\mathbf{z} = \frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \mathbf{h}_t^{(L)}$$

where $\mathcal{R}$ is the set of real (non-padding) positions.

**Rationale:**
1. Code-switching is token-centric (language identity is per-token)
2. Mean pooling treats all languages democratically
3. ELECTRA's [CLS] receives no special sentence-level supervision (just RTD like every token)

---

## 11. Design Decisions

### 11.1 Why Two-Stage Language Distributions?

**Problem:** Need language info early (augmentation) but accurate detection requires context.

**Solution:**
- $p^{(0)}_t$: Fast, pre-contextual, used for augmentation (avoids circular dependency)
- $p^{(\text{ctx})}_t$: Accurate, context-refined, used for attention biases

### 11.2 Why Per-Head Compatibility Matrices?

**Old design:** Scalar $\beta_h$ per head
- Symmetric: $\beta_h p_i^\top p_j = \beta_h p_j^\top p_i$
- Limited expressiveness

**New design:** Matrix $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ per head
- Asymmetric: $p_i^\top \mathbf{C}_h p_j \neq p_j^\top \mathbf{C}_h p_i$ in general
- Learns language-specific affinities
- Only $K^2 H$ parameters (192 for base)

### 11.3 Why ELECTRA-Style RTD?

**Vanilla MLM:** Supervises $\approx 15\%$ of tokens per batch

**ELECTRA RTD:** Supervises $100\%$ of tokens per batch

**Result:** 6-7× more training signal for same data budget. ELECTRA matches BERT performance with 1/4 the compute.

### 11.4 Why Switch-Span Masking?

**Random masking:** Doesn't target code-switching structure

**Switch-span masking:** Masks entire language segments, forcing reconstruction from cross-language context. Directly targets code-switching objective.

### 11.5 Why Learnable Temperature?

**Fixed $\tau$:** One size fits all

**Learnable $\tau$:** Model optimizes its own sharpness. Different datasets may need different $\tau$. Stored as $\log \tau$ for unconstrained optimization.

---

## 12. Model Configurations

| Config | $d$ | $L$ | $H$ | FFN | Params |
|--------|-----|-----|-----|-----|--------|
| Small  | 256 | 4   | 4   | 1024 | ~16M  |
| Medium | 512 | 8   | 8   | 2048 | ~51M  |
| Base   | 768 | 12  | 12  | 3072 | ~124M |
| Large  | 1024| 24  | 16  | 4096 | ~355M |

All configurations use $K=4$ language prototypes by default.

---

## 13. Summary

SBERTa is a general-purpose code-switching architecture featuring:

1. **Explicit language modeling** via learnable prototypes and soft distributions
2. **Contextual refinement** to fix Latin-script ambiguity
3. **Efficient pre-training** via ELECTRA-style replaced token detection
4. **Fully unsupervised routing** via temporal stickiness loss (no external labels or classifiers)
5. **Language-aware attention** with per-head compatibility matrices
6. **Learnable hyperparameters** (temperature $\tau$, curriculum weight $w_{\text{curr}}$)

The architecture is language-agnostic and applicable to any code-switching scenario without modification.
