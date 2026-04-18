# SBERTa: Architecture Specification

**SBERTa** (**S**witching **B**idirectional **E**ncoder **R**epresentations from **T**ransformers **a**rchitecture) is a transformer architecture for code-switched text featuring explicit unsupervised language modeling and ELECTRA-style adversarial pre-training.

---

## 1. Core Design Philosophy

SBERTa makes two linguistic phenomena explicit architectural components:

1. **Language identity** — soft distributions over $K$ language prototypes
2. **Language boundaries** — continuous switch magnitudes between consecutive tokens

Both components are learned end-to-end without hardcoded language labels, making the architecture applicable to any code-switching scenario. A Unicode script prior grounds the learned distributions in objective orthographic evidence, providing strong prototype separation from the very first step. The encoder's multi-layer self-attention mechanism handles all remaining contextual disambiguation.

---

## 2. Language Prototypes

### 2.1 Prototype Vectors

The model learns $K$ prototype vectors $\mathbf{L} = [\boldsymbol{\ell}_1, \ldots, \boldsymbol{\ell}_K] \in \mathbb{R}^{K \times d}$ representing language directions in embedding space.

**Initialization:** Orthogonal initialization scaled by 0.5 to avoid softmax saturation:
$$\mathbf{L} \sim \text{Orthogonal}(\mathbb{R}^{K \times d}), \quad \mathbf{L} \leftarrow 0.5 \cdot \mathbf{L}$$

### 2.2 Unicode Script Prior

A hard prior is derived from the Unicode script identity of each token's first non-neutral character. Each vocabulary token is assigned a script index $k \in \{0, \ldots, K-1\}$ based on the most frequent script in the vocabulary (e.g., Arabic = 0, Latin = 1). Neutral characters — digits, punctuation, symbols — receive no script assignment and are left at the uniform prior.

The per-token prior distribution is:

$$[\mathbf{p}^{\text{script}}_t]_k = \begin{cases} 1 & \text{if } x_t \text{ is assigned script } k \\ 0 & \text{if } x_t \text{ is assigned a different script} \\ \tfrac{1}{K} & \text{if } x_t \text{ has no script assignment (neutral)} \end{cases}$$

The prior is one-hot for tokens with a known script and uniform for neutral tokens. All softening is handled by the blend weight $w$ — there is no separate concentration parameter. An Arabic token gets $\mathbf{p}^{\text{script}} = [1, 0]$; blended at $w = 0.5$ this pulls $p^{(0)}$ to $0.5 \cdot p^{\text{learned}} + 0.5 \cdot [1, 0]$, a genuine signal toward prototype 0 whose strength is entirely controlled by $w$.

### 2.3 Pre-Contextual Language Distributions

For each token $t$, the learned distribution is computed from raw base embeddings $\mathbf{h}_{\text{base},t} = \mathbf{E}_{\text{tok}}(x_t) + \mathbf{E}_{\text{pos}}(t)$:

$$p^{\text{learned}}_t = \text{softmax}\left(\frac{\mathbf{h}_{\text{base},t} \mathbf{L}^\top}{\tau}\right) \in \Delta^K$$

This is blended with the Unicode script prior using a fixed weight $w \in [0, 1]$:

$$p^{(0)}_t = (1 - w)\, p^{\text{learned}}_t + w\, p^{\text{script}}_t$$

where $\tau$ is a learnable temperature parameter stored as $\log \tau$ for unconstrained optimization:
$$\tau = \exp(\log \tau), \quad \log \tau \in \mathbb{R}, \quad \tau \geq 0.25$$

**Properties:**
- Computed before contextual processing — no circular dependency
- The script prior ensures reliable prototype separation from step one, even before the learned component has trained
- Used for span masking, embedding augmentation, and attention biases — $p^{(0)}$ is the single language signal throughout the entire model
- The encoder's self-attention stack resolves all remaining ambiguity: loanwords, numerals, punctuation, Arabizi

### 2.4 Switch Magnitudes

Continuous measure of language change between consecutive tokens:

$$s_t = 1 - p^{(0)}_t{}^\top p^{(0)}_{t-1}, \quad s_1 = 0$$

Since $p^{(0)}_t \in \Delta^K$, the inner product $p^{(0)}_t{}^\top p^{(0)}_{t-1} \in [0, 1]$, thus $s_t \in [0, 1]$.

**Interpretation:**
- $s_t \approx 0$: Same language as previous token
- $s_t \approx 1$: Complete language switch
- $s_t \in (0, 1)$: Partial or gradual transition (ambiguous tokens, punctuation, numerals)

---

## 3. Language-Augmented Embeddings

### 3.1 Base Embeddings

$$\mathbf{h}_{\text{base},t} = \mathbf{E}_{\text{tok}}(x_t) + \mathbf{E}_{\text{pos}}(t)$$

No normalization or augmentation applied at this stage. Used to compute $p^{(0)}_t$ and $s_t$ before augmentation, avoiding any circular dependency.

### 3.2 Augmentation

$$\mathbf{h}^{(0)}_t = \text{LayerNorm}\left(\mathbf{h}_{\text{base},t} + \sum_{k=1}^K p^{(0)}_{t,k} \, \mathbf{e}_k + s_t \, \mathbf{e}_{\text{sw}}\right)$$

where:
- $\mathbf{e}_k \in \mathbb{R}^d$ are learnable language embeddings (one per prototype)
- $\mathbf{e}_{\text{sw}} \in \mathbb{R}^d$ is a learnable switch embedding

**Interpretation:**
- $\sum_k p^{(0)}_{t,k} \mathbf{e}_k$: Soft language signal weighted by the blended distribution
- $s_t \mathbf{e}_{\text{sw}}$: Boundary signal scaled by switch magnitude

---

## 4. Language-Aware Attention

### 4.1 Attention Score Computation

Standard transformer attention with two additive language-aware biases:

$$\text{score}_h(i, j) = \frac{\mathbf{Q}_{h,i} \mathbf{K}_{h,j}^\top}{\sqrt{d_h}} + p^{(0)}_i{}^\top \mathbf{C}_h \, p^{(0)}_j + \gamma \, s_j$$

where:
- $h \in \{1, \ldots, H\}$ indexes attention heads
- $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ is a per-head language compatibility matrix
- $\gamma \in \mathbb{R}$ is a global switch-position bias shared across all heads

### 4.2 Language Compatibility Bias

$$p^{(0)}_i{}^\top \mathbf{C}_h \, p^{(0)}_j = \sum_{k=1}^K \sum_{\ell=1}^K p^{(0)}_{i,k} \, [\mathbf{C}_h]_{k\ell} \, p^{(0)}_{j,\ell}$$

**Properties:**
- Asymmetric: $[\mathbf{C}_h]_{k\ell} \neq [\mathbf{C}_h]_{\ell k}$ in general — learns directional affinities (e.g., how strongly an Arabic query attends to a Latin key vs. the reverse)
- Initialized as identity plus small noise: $\mathbf{C}_h = \mathbf{I}_K + \epsilon$, so training begins from standard attention behaviour while allowing heads to specialize immediately
- Only $K^2 H$ additional parameters (48 for base config with $K=2$, $H=12$)

### 4.3 Switch Position Bias

$$\gamma \, s_j$$

**Properties:**
- Encourages attention toward language boundary positions
- Initialized to zero; the model learns whether this signal is useful
- Shared across all heads

### 4.4 Fixed Language Distributions

Both $p^{(0)}_t$ and $s_t$ are computed once from raw base embeddings and remain fixed across all encoder layers. They are threaded through the encoder as constants.

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
- Hidden size: $d_{\text{gen}} = d / 2$
- Attention heads: $H_{\text{gen}} = H / 2$
- Layers: $L_{\text{gen}} = L / 2$
- Shares token embedding weights with the discriminator (passed at forward time to avoid double registration in the module tree)

**Discriminator** (full SBERTa):
- Full architecture ($d$, $H$, $L$)
- Binary classification head: $\mathbb{R}^d \to \mathbb{R}$

### 6.2 Switch-Span Masking

Instead of random token masking, SBERTa masks entire language-homogeneous spans. Span boundaries are derived from the blended $p^{(0)}_t$, so that script identity informs masking decisions from step one:

1. Compute dominant language per token: $\ell_t = \arg\max_k p^{(0)}_{t,k}$
2. Identify span boundaries where $\ell_t \neq \ell_{t-1}$
3. Randomly select whole spans until $\approx 15\%$ of real tokens are covered, with a hard cap at $30\%$ to prevent a single dominant span from monopolising a short sequence

**Rationale:** Forces the generator to reconstruct a full language segment from cross-language context, directly targeting the code-switching objective.

### 6.3 Training Flow

1. Compute $p^{(0)}_t$ (blended with script prior) from the original unmasked input
2. Compute switch magnitudes $s_t$ from $p^{(0)}_t$
3. Apply switch-span masking using $p^{(0)}_t$
4. Generator proposes token replacements for all masked positions
5. Discriminator receives the corrupted sequence (original tokens at unmasked positions, generator samples at masked positions)
6. Discriminator classifies each real token as original or replaced

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
- $y_t = 1$ if token $t$ was replaced by the generator, $0$ if original
- $f_{\text{disc}}: \mathbb{R}^d \to \mathbb{R}$ is a linear discriminator head

**Key advantage:** Supervises all $T$ real positions, not just $\approx 0.15T$ masked positions. Provides 6–7× more gradient signal than vanilla MLM for the same data budget — particularly valuable for low-resource languages where corpus size is limited.

### 7.3 Temporal Stickiness Loss

Penalises the mean switch magnitude over qualifying consecutive real-token boundaries:

$$\mathcal{L}_{\text{smooth}} = \frac{1}{|\mathcal{B}|} \sum_{(t-1,\,t) \in \mathcal{B}} s_t$$

where $\mathcal{B}$ is the set of consecutive real-token boundary pairs that pass two filters:

1. **Cross-document boundaries** are excluded — when sequences are packed from multiple documents, the switch at a document join is an artefact of packing, not a genuine language transition.
2. **Cross-script boundaries** are excluded — a transition between tokens of known, distinct Unicode scripts (e.g., Arabic → Latin) is a true language switch and must not be penalised. $\mathcal{L}_{\text{smooth}}$ handles within-script ambiguity only: punctuation runs, numerals, Arabizi sequences, and other cases where the correct language assignment depends on context rather than orthography.

Applied at a constant static weight $\lambda_{\text{smooth}}$ from the beginning of training.

**Purpose:** Forces prototypes to self-organise into long, linguistically coherent spans rather than flipping per-token. No external labels or language identification models required.

### 7.4 Prototype Diversity Loss

Exponential repulsion loss to maintain geometric separation between prototypes:

$$\mathcal{L}_{\text{div}} = \frac{1}{\binom{K}{2}} \sum_{i < j} \left(\exp\!\left(\frac{\boldsymbol{\ell}_i^\top \boldsymbol{\ell}_j}{\|\boldsymbol{\ell}_i\| \|\boldsymbol{\ell}_j\|}\right) - e^{-1}\right)$$

**Properties:**
- Always positive with a non-zero gradient — exponentially steeper as prototypes approach alignment
- Effective weight $\lambda_{\text{div}} = 0$ in the $K=2$ configuration, where the script prior structurally guarantees separation. Available for $K > 2$ experiments.

### 7.5 Combined Loss

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + w_{\text{rtd}} \cdot \mathcal{L}_{\text{RTD}} + \lambda_{\text{smooth}} \cdot \mathcal{L}_{\text{smooth}} + \lambda_{\text{div}} \cdot \mathcal{L}_{\text{div}}$$

**Default weights:**
- $w_{\text{rtd}} = 15.0$
- $\lambda_{\text{smooth}} = 5.0$
- $\lambda_{\text{div}} = 0.0$

> **Note:** A per-token prototype commitment loss ($\mathcal{L}_{\text{sharp}}$) was considered but is not used. $\mathcal{L}_{\text{smooth}}$ provides sufficient sharpening — explicit per-token entropy minimisation conflicts with desirable soft distributions at language boundaries (punctuation, numerals, loanwords).

---

## 8. Gradient Flow Analysis

### 8.1 What Updates Prototypes

**Direct gradients:**
- $\mathcal{L}_{\text{smooth}}$: Through `get_switch_magnitudes()` → through `get_distributions()` → through $\mathbf{L}$ directly
- $\mathcal{L}_{\text{div}}$ (when $\lambda_{\text{div}} > 0$): Direct regularization on $\mathbf{L}$

**Indirect gradients:**
- $\mathcal{L}_{\text{gen}}$: Through embedding augmentation $\sum_k p^{(0)}_{t,k} \mathbf{e}_k$
- $\mathcal{L}_{\text{RTD}}$: Through embedding augmentation and attention biases

### 8.2 Key Insight

$\mathcal{L}_{\text{smooth}}$ and $\mathcal{L}_{\text{RTD}}$ apply opposing pressures on the prototypes: RTD rewards semantic discriminability per token while $\mathcal{L}_{\text{smooth}}$ rewards long same-language spans. The tension between these forces drives prototypes toward linguistically meaningful, temporally-sticky clusters without any external labels. The script prior ensures this tension operates on already-separated prototypes rather than competing over a shared random initialisation.

---

## 9. Sentence-Level Representations

SBERTa does not use a [CLS] token. For sentence-level tasks, use **mean pooling**:

$$\mathbf{z} = \frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \mathbf{h}_t^{(L)}$$

where $\mathcal{R}$ is the set of real (non-padding) positions.

**Rationale:**
1. Code-switching is token-centric — language identity is a per-token property
2. Mean pooling treats all languages democratically
3. ELECTRA's [CLS] receives no special sentence-level supervision (just RTD like every other token)

---

## 10. Design Decisions

### 10.1 Why a Unicode Script Prior?

The prototype mechanism is learned, but learning requires the prototypes to be meaningfully separated before the language signal is useful. The Unicode script prior provides this separation using the objective fact that Arabic script and Latin script are orthographically distinct. Neutral characters — punctuation, digits, emojis — receive the uniform prior and are assigned purely by learned context, which is the correct behaviour: their language identity genuinely depends on their neighbours.

### 10.2 Why a Single Blended Pre-Contextual Distribution?

Language information is needed before contextual processing, but feeding augmented embeddings back into language detection creates a circular dependency. The solution is to compute $p^{(0)}_t$ once from raw base embeddings (tok + pos only), blend with the script prior, and use the result everywhere — for span masking, embedding augmentation, and attention biases. The encoder's layers of full $T \times T$ self-attention then resolve all remaining ambiguity, making a separate pre-refinement stage architecturally redundant.

### 10.3 Why Per-Head Compatibility Matrices?

A scalar bias per head is symmetric and limited in expressiveness. A matrix $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ per head is asymmetric, allowing the model to learn that an Arabic query attending to a Latin key should receive a different bias than a Latin query attending to an Arabic key. The cost is only $K^2 H$ additional parameters — 48 for the base configuration.

### 10.4 Why ELECTRA-Style RTD?

Vanilla MLM supervises $\approx 15\%$ of tokens per batch. ELECTRA RTD supervises $100\%$, providing 6–7× more gradient signal for the same data budget. For low-resource languages like Algerian Darija, this sample efficiency is critical.

### 10.5 Why Switch-Span Masking?

Random token masking ignores the structure of code-switching. Masking entire language-homogeneous spans forces the generator to reconstruct a full language segment from cross-language context, which is precisely the capability SBERTa is pre-trained to develop. Span boundaries are derived from the blended $p^{(0)}_t$ so that script identity contributes from step one.

### 10.6 Why Learnable Temperature?

A fixed $\tau$ imposes a single sharpness on all prototype distributions. A learnable $\tau$, stored as $\log \tau$ for unconstrained optimization, allows the model to find its own operating point jointly with all other parameters. The floor of 0.25 prevents the softmax from collapsing into a near-one-hot distribution.

---

## 11. Model Configurations

| Config | $d$ | $L$ | $H$ | FFN | $K$ | Params |
|--------|-----|-----|-----|-----|-----|--------|
| Small  | 256 | 4   | 4   | 1024  | 2 | ~16M  |
| Medium | 512 | 8   | 8   | 2048  | 2 | ~51M  |
| Base   | 768 | 12  | 12  | 3072  | 2 | ~124M |
| Large  | 1024| 24  | 16  | 4096  | 2 | ~355M |

$K = 2$ for the primary Arabic/Latin (Darija) use case. The architecture supports arbitrary $K$ — increasing $K$ requires setting $\lambda_{\text{div}} > 0$ and extending the script prior assignment logic to cover additional scripts.

---

## 12. Summary

SBERTa is a general-purpose code-switching architecture featuring:

1. **Explicit language modeling** via learnable prototypes with soft distributions, grounded by a Unicode script prior
2. **Single-stage language routing** — $p^{(0)}$ computed once from raw embeddings and blended with the script prior serves as the language signal for span masking, embedding augmentation, and all attention biases
3. **Efficient pre-training** via ELECTRA-style replaced token detection (6–7× gradient signal vs. vanilla MLM)
4. **Fully unsupervised language boundary discovery** via temporal stickiness loss with cross-script and cross-document exclusion filters
5. **Language-aware attention** with per-head $K \times K$ asymmetric compatibility matrices
6. **Stable prototype geometry from step one** via the Unicode script prior
7. **Learnable temperature** $\tau$ jointly optimized with all model weights