# SBERTa: Architecture Specification

**SBERTa** (**S**witching **B**idirectional **E**ncoder **R**epresentations from **T**ransformers **a**rchitecture) is a transformer architecture for code-switched text featuring explicit unsupervised language modeling and ELECTRA-style adversarial pre-training.

---

## 1. Core Design Philosophy

SBERTa makes two linguistic phenomena explicit architectural components:

1. **Language identity** — soft distributions over $K$ language prototypes
2. **Language boundaries** — continuous switch magnitudes between consecutive tokens

Both components are learned end-to-end without hardcoded language labels, making the architecture applicable to any code-switching scenario. The encoder's multi-layer self-attention mechanism refines language understanding through contextual processing.

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
- Used for embedding augmentation **and** attention biases — $p^{(0)}$ is the single language signal throughout the entire model
- The encoder's multi-layer self-attention refines language understanding through contextual processing without requiring a separate pre-refinement module

### 2.3 Switch Magnitudes

Continuous measure of language change between consecutive tokens:

$$s_t = 1 - p^{(0)}_t{}^\top p^{(0)}_{t-1}, \quad s_1 = 0$$

Since $p^{(0)}_t \in \Delta^K$, the inner product $p^{(0)}_t{}^\top p^{(0)}_{t-1} \in [0, 1]$, thus $s_t \in [0, 1]$.

**Interpretation:**
- $s_t \approx 0$: Same language as previous token
- $s_t \approx 1$: Complete language switch
- $s_t \in (0, 1)$: Partial or gradual transition

---

## 3. Language-Augmented Embeddings

### 3.1 Base Embeddings

$$\mathbf{h}_{\text{base},t} = \mathbf{E}_{\text{tok}}(x_t) + \mathbf{E}_{\text{pos}}(t)$$

No normalization or augmentation yet. Used to compute $p^{(0)}_t$ and $s_t$.

### 3.2 Augmentation

$$\mathbf{h}^{(0)}_t = \text{LayerNorm}\left(\mathbf{h}_{\text{base},t} + \sum_{k=1}^K p^{(0)}_{t,k} \, \mathbf{e}_k + s_t \, \mathbf{e}_{\text{sw}}\right)$$

where:
- $\mathbf{e}_k \in \mathbb{R}^d$ are learnable language embeddings (one per prototype)
- $\mathbf{e}_{\text{sw}} \in \mathbb{R}^d$ is a learnable switch embedding

**Interpretation:**
- $\sum_k p^{(0)}_{t,k} \mathbf{e}_k$: Soft language signal (weighted by distribution)
- $s_t \mathbf{e}_{\text{sw}}$: Boundary signal (scaled by switch magnitude)

---

## 4. Language-Aware Attention

### 4.1 Attention Score Computation

Standard transformer attention with two additive language-aware biases:

$$\text{score}_h(i, j) = \frac{\mathbf{Q}_{h,i} \mathbf{K}_{h,j}^\top}{\sqrt{d_h}} + p^{(0)}_i{}^\top \mathbf{C}_h \, p^{(0)}_j + \gamma \, s_j$$

where:
- $h \in \{1, \ldots, H\}$ indexes attention heads
- $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ is a per-head language compatibility matrix
- $\gamma \in \mathbb{R}$ is a global switch-position bias (shared across heads)

### 4.2 Language Compatibility Bias

$$p^{(0)}_i{}^\top \mathbf{C}_h \, p^{(0)}_j = \sum_{k=1}^K \sum_{\ell=1}^K p^{(0)}_{i,k} \, [\mathbf{C}_h]_{k\ell} \, p^{(0)}_{j,\ell}$$

**Properties:**
- Asymmetric: $[\mathbf{C}_h]_{k\ell} \neq [\mathbf{C}_h]_{\ell k}$ in general
- Learns language-specific affinities (e.g., Arabic $\leftrightarrow$ Arabizi high, French $\leftrightarrow$ Arabic medium)
- Initialized as identity: $\mathbf{C}_h = \mathbf{I}_K$ (equivalent to scalar $\beta_h = 1$)
- Only $K^2 H$ parameters (192 for base config with $K=4$, $H=12$)

### 4.3 Switch Position Bias

$$\gamma \, s_j$$

**Properties:**
- Encourages attention toward language boundaries when $\gamma > 0$
- Initialized to zero (model learns if useful)
- Shared across all heads

### 4.4 Fixed Language Distributions

Both $p^{(0)}_t$ and $s_t$ are computed once from raw base embeddings and remain fixed across all encoder layers. They are threaded through the encoder but not updated layer-by-layer.

---

## 5. Encoder Architecture

### 5.1 Layer Structure

Each encoder layer $\ell \in \{1, \ldots, L\}$ applies:

$$\mathbf{H}_{\text{attn}}^{(\ell)} = \text{MultiHeadAttention}(\mathbf{H}^{(\ell-1)}, p^{(0)}, s)$$

$$\mathbf{H}_{\text{mid}}^{(\ell)} = \text{LayerNorm}(\mathbf{H}^{(\ell-1)} + \text{Dropout}(\mathbf{H}_{\text{attn}}^{(\ell)}))$$

$$\mathbf{H}^{(\ell)} = \text{LayerNorm}(\mathbf{H}_{\text{mid}}^{(\ell)} + \text{Dropout}(\text{FFN}(\mathbf{H}_{\text{mid}}^{(\ell)})))$$

where $\mathbf{H}^{(0)} = [\mathbf{h}^{(0)}_1, \ldots, \mathbf{h}^{(0)}_T]$ are the augmented embeddings.

### 5.2 Post-LayerNorm

SBERTa uses post-LayerNorm (LayerNorm after residual) for stability during ELECTRA-style training.

---

## 6. ELECTRA-Style Pre-training

### 6.1 Architecture Overview

**Generator** (small):
- Hidden size: $d_{\text{gen}} = d / 4$
- Attention heads: $H_{\text{gen}} = H / 4$
- Layers: $L_{\text{gen}} = L / 4$
- Shares token embeddings with discriminator

**Discriminator** (full SBERTa):
- Full architecture ($d$, $H$, $L$)
- Binary classification head: $\mathbb{R}^d \to \mathbb{R}$

### 6.2 Switch-Span Masking

Instead of random 15% token masking, SBERTa masks entire language-homogeneous spans:

1. Compute dominant language per token: $\ell_t = \arg\max_k p^{(0)}_{t,k}$
2. Identify span boundaries where $\ell_t \neq \ell_{t-1}$
3. Randomly select whole spans until $\approx 15\%$ coverage

**Rationale:** Forces generator to reconstruct language segments from cross-language context, directly targeting code-switching.

### 6.3 Training Flow

1. Compute $p^{(0)}_t$ from original (unmasked) input
2. Apply switch-span masking using $p^{(0)}_t$
3. Generator proposes replacements for masked positions
4. Discriminator receives corrupted sequence (original + replacements)
5. Discriminator classifies each token as real or replaced

---

## 7. Training Objectives

### 7.1 Generator Loss

Masked language modeling on masked spans only:

$$\mathcal{L}_{\text{gen}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P_{\text{gen}}(x_t \mid \mathbf{x}_{\setminus \mathcal{M}})$$

where $\mathcal{M}$ is the set of masked positions.

### 7.2 Discriminator Loss (RTD)

Binary cross-entropy at every real (non-padding) token:

$$\mathcal{L}_{\text{RTD}} = -\frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \left[ y_t \log \sigma(f_{\text{disc}}(\mathbf{h}_t)) + (1 - y_t) \log(1 - \sigma(f_{\text{disc}}(\mathbf{h}_t))) \right]$$

where:
- $\mathcal{R}$ is the set of real (non-padding) positions
- $y_t = 1$ if token $t$ was replaced, $0$ if original
- $f_{\text{disc}}: \mathbb{R}^d \to \mathbb{R}$ is a linear discriminator head

**Key advantage:** Supervises all $T$ positions, not just $\approx 0.15T$ masked positions. Provides 6-7× more gradient signal than vanilla MLM.

### 7.3 Temporal Stickiness Loss (Unsupervised)

Penalises the mean switch magnitude over real consecutive token boundaries:

$$\mathcal{L}_{\text{smooth}} = \frac{1}{|\mathcal{B}|} \sum_{(t-1,\,t) \in \mathcal{B}} s_t$$

where $\mathcal{B}$ is the set of consecutive real-token boundary pairs (padding boundaries excluded).

**Curriculum schedule:** 

- **Burn-in phase** (steps 0 to `burnin_ratio × total_steps`, default 2%): $w_{\text{curr}} = 0$. The loss is completely gated off so prototypes can separate geometrically via $\mathcal{L}_{\text{div}}$ alone before temporal stickiness starts pulling them together.
- **Ramp phase** (duration `smooth_warmup_ratio × total_steps`, default 10%): $w_{\text{curr}}$ ramps linearly from `smooth_weight_min` (default 0.05) to 1.0.
- **Full strength** (after ramp completes): $w_{\text{curr}} = 1.0$.

The two-phase schedule prevents prototype collapse from simultaneous attraction (smooth) and repulsion (div) forces at initialization. Ratios scale automatically with total training steps.

**Purpose:** Forces prototypes to self-organise into long, linguistically coherent spans rather than flipping per-token. No external labels or fastText model required — the model discovers language boundaries purely from the temporal structure of the data.

### 7.4 Prototype Diversity Loss

Margin-based repulsion loss to prevent prototype collapse:

$$\mathcal{L}_{\text{div}} = \frac{1}{\binom{K}{2}} \sum_{i=1}^{K-1} \sum_{j=i+1}^K \left(\text{ReLU}\!\left(\frac{\boldsymbol{\ell}_i^\top \boldsymbol{\ell}_j}{\|\boldsymbol{\ell}_i\| \|\boldsymbol{\ell}_j\|} + m\right)\right)^2, \quad m = 0.1$$

The margin $m=0.1$ means the loss fires even when prototypes are near-orthogonal ($\cos \approx 0$), providing a constant repulsion gradient from step 0. This fixes the timing asymmetry where a pure $\cos^2$ formulation produces zero gradient at orthogonal initialisation while $\mathcal{L}_{\text{smooth}}$ is already pushing tokens toward shared prototypes.

**Boundary values:**
- At perfect orthogonality ($\cos = 0$): loss $= \text{ReLU}(0.1)^2 = 0.01$ per pair
- At collapse ($\cos = 1$): loss $= \text{ReLU}(1.1)^2 = 1.21$ per pair
- Only when $\cos < -m$ does the loss reach zero

**Purpose:** Ensures prototypes remain geometrically separated.

### 7.5 Combined Loss

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + w_{\text{rtd}} \mathcal{L}_{\text{RTD}} + \lambda_{\text{smooth}} \cdot w_{\text{curr}} \cdot \mathcal{L}_{\text{smooth}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_{\text{balance}} \mathcal{L}_{\text{balance}}$$

**Default weights:**
- $w_{\text{rtd}} = 50.0$ (ELECTRA standard)
- $\lambda_{\text{smooth}} = 15.0$ (15:50 ratio vs RTD)
- $\lambda_{\text{div}} = 1.0$ (conservative starting point; ablate upward if needed)
- $\lambda_{\text{balance}} = 1.0$ (soft minimum-usage; fires when prototype usage drops below `balance_min_usage_factor / K`, default factor=0.25 → 25% of uniform)
- $w_{\text{curr}}$: see curriculum schedule above (burn-in → ramp → full strength)

> **Note:** A per-token prototype commitment loss ($\mathcal{L}_{\text{sharp}}$) was considered but is not used. $\mathcal{L}_{\text{smooth}}$ and $\mathcal{L}_{\text{div}}$ together provide sufficient sharpening — explicit per-token entropy minimisation is redundant and can conflict with desirable soft distributions at language boundaries.

---

## 8. Gradient Flow Analysis

### 8.1 What Updates Prototypes

**Direct gradients:**
- $\mathcal{L}_{\text{smooth}}$: Through `get_switch_magnitudes()` → through `get_distributions()` → through $\mathbf{L}$ directly
- $\mathcal{L}_{\text{div}}$: Direct regularization on $\mathbf{L}$

**Indirect gradients:**
- $\mathcal{L}_{\text{gen}}$: Through embedding augmentation $\sum_k p^{(0)}_{t,k} \mathbf{e}_k$
- $\mathcal{L}_{\text{RTD}}$: Through embedding augmentation and attention biases

### 8.2 Key Insight

$\mathcal{L}_{\text{smooth}}$ and $\mathcal{L}_{\text{RTD}}$ apply opposing pressures on the prototypes: RTD rewards semantic discriminability per token while $\mathcal{L}_{\text{smooth}}$ rewards long same-language spans. The tension between these forces drives prototypes toward linguistically meaningful, temporally-sticky clusters — without any external labels.

---

## 9. Sentence-Level Representations

SBERTa does not use a [CLS] token. For sentence-level tasks, use **mean pooling**:

$$\mathbf{z} = \frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \mathbf{h}_t^{(L)}$$

where $\mathcal{R}$ is the set of real (non-padding) positions.

**Rationale:**
1. Code-switching is token-centric (language identity is per-token)
2. Mean pooling treats all languages democratically
3. ELECTRA's [CLS] receives no special sentence-level supervision (just RTD like every token)

---

## 10. Design Decisions

### 10.1 Why a Single Pre-Contextual Language Distribution?

**Problem:** Need language information early for embedding augmentation, but naively feeding augmented embeddings back into language detection creates a circular dependency.

**Solution:** Compute $p^{(0)}_t$ once from raw base embeddings (tok + pos, no augmentation) and use it everywhere — both for embedding augmentation and for attention biases $\mathbf{C}_h$ and $\gamma$. The encoder's 12 layers of full $T \times T$ self-attention then resolve Latin-script ambiguity (e.g., "chat" = French vs. English) far more powerfully than any separate windowed refinement module, making a two-stage distribution pipeline architecturally redundant.

### 10.2 Why Per-Head Compatibility Matrices?

**Old design:** Scalar $\beta_h$ per head
- Symmetric: $\beta_h p_i^\top p_j = \beta_h p_j^\top p_i$
- Limited expressiveness

**New design:** Matrix $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ per head
- Asymmetric: $p_i^\top \mathbf{C}_h p_j \neq p_j^\top \mathbf{C}_h p_i$ in general
- Learns language-specific affinities
- Only $K^2 H$ parameters (192 for base)

### 10.3 Why ELECTRA-Style RTD?

**Vanilla MLM:** Supervises $\approx 15\%$ of tokens per batch

**ELECTRA RTD:** Supervises $100\%$ of tokens per batch

**Result:** 6-7× more training signal for same data budget. ELECTRA matches BERT performance with 1/4 the compute.

### 10.4 Why Switch-Span Masking?

**Random masking:** Doesn't target code-switching structure

**Switch-span masking:** Masks entire language segments, forcing reconstruction from cross-language context. Directly targets code-switching objective.

### 10.5 Why Learnable Temperature?

**Fixed $\tau$:** One size fits all

**Learnable $\tau$:** Model optimizes its own sharpness. Different datasets may need different $\tau$. Stored as $\log \tau$ for unconstrained optimization. Floored at 0.25 to prevent collapse into a state where prototype imbalance recovery becomes impossible.

---

## 11. Model Configurations

| Config | $d$ | $L$ | $H$ | FFN | Params |
|--------|-----|-----|-----|-----|--------|
| Small  | 256 | 4   | 4   | 1024 | ~16M  |
| Medium | 512 | 8   | 8   | 2048 | ~51M  |
| Base   | 768 | 12  | 12  | 3072 | ~124M |
| Large  | 1024| 24  | 16  | 4096 | ~355M |

All configurations use $K=4$ language prototypes by default.

---

## 12. Summary

SBERTa is a general-purpose code-switching architecture featuring:

1. **Explicit language modeling** via learnable prototypes and soft distributions
2. **Single-stage language routing** — $p^{(0)}$ computed once from raw embeddings is used for both augmentation and attention biases; contextual disambiguation is handled by the encoder's 12 layers of full self-attention
3. **Efficient pre-training** via ELECTRA-style replaced token detection
4. **Fully unsupervised routing** via temporal stickiness loss (no external labels or classifiers)
5. **Language-aware attention** with per-head compatibility matrices
6. **Learnable hyperparameters** (temperature $\tau$, curriculum weight $w_{\text{curr}}$)

The architecture is language-agnostic and applicable to any code-switching scenario without modification.