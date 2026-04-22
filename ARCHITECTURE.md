# SBERTa: Architecture Specification

**SBERTa** (**S**witching **B**idirectional **E**ncoder **R**epresentations from **T**ransformers **a**rchitecture) is a universal, zero-knowledge transformer architecture for code-switched text. It features explicit unsupervised language modeling, Optimal Transport clustering, and stable ELECTRA-style pre-training.

---

## 1. Core Design Philosophy

SBERTa treats code-switching as a structural signal rather than noise. It makes two linguistic phenomena explicit architectural components:

1. **Language identity** — soft distributions over $K$ language prototypes.
2. **Language boundaries** — continuous pairwise divergence between token representations.

The model is **fully zero-knowledge**. It uses no Unicode priors, script IDs, or dictionaries. Language structure emerges purely from the Masked Language Modeling (MLM) distributional objective, stabilized by a Sinkhorn-Knopp clustering loss. This makes SBERTa a universal standard applicable to any K-language mixture (multi-script like Darija, or mono-script like Spanglish/Hinglish).

---

## 2. Two-Phase Encoder

SBERTa avoids the "bootstrap paradox" of assigning language probabilities to noisy, raw token embeddings. Instead, it employs a two-phase design:

1. **Phase 1 — Context (layers $0 \dots n_{\text{base}}-1$):** Standard self-attention layers with no language bias injected. Uses Pre-LayerNorm (Pre-LN) and PyTorch's Scaled Dot Product Attention (FlashAttention) fast-path. MLM distributional pressure forces contextualized, semantically rich representations ($\mathbf{h}_{\text{base}}$) to emerge naturally.
2. **Language Assignment Pivot:** Language distributions ($\mathbf{p}$) and pairwise divergences ($\boldsymbol{\Delta}$) are computed from $\mathbf{h}_{\text{base}}$ at the boundary between phases. Because they are based on context, they are meaningful from the start.
3. **Phase 2 — Language-Aware (layers $n_{\text{base}} \dots L-1$):** Language signals are routed into the attention mechanism via learned compatibility structures, guiding the rest of the network.

---

## 3. Language Prototypes and Clustering

### 3.1 Prototype Vectors

The model learns $K$ prototype vectors $\mathbf{L} = [\boldsymbol{\ell}_1, \ldots, \boldsymbol{\ell}_K]^\top \in \mathbb{R}^{K \times d}$ representing language directions in the embedding space.

**Initialization:** Orthogonal initialization scaled by 0.5 to avoid softmax saturation:
$$\mathbf{L} \sim \text{Orthogonal}(\mathbb{R}^{K \times d}), \quad \mathbf{L} \leftarrow 0.5 \cdot \mathbf{L}$$

### 3.2 Contextual Language Distributions

For each token $t$, the distribution is computed from the output of Phase 1, $\mathbf{h}_{\text{base},t}$:

$$\mathbf{p}_t = \text{softmax}\left(\frac{\mathbf{L}_{\text{norm}} \mathbf{h}_{\text{base},t}^\top}{\tau}\right) \in \Delta^K$$

(where $\mathbf{L}_{\text{norm}}$ is $\mathbf{L}$ L2-normalized along the hidden dimension, making $p$ depend on cosine similarity rather than raw magnitude).

$\tau$ is a precision parameter with a numerical stability floor, parameterized smoothly via softplus:
$$\tau = \tau_{\min} + \text{softplus}(\rho), \quad \rho \in \mathbb{R}, \quad \tau_{\min} = 0.25$$

This guarantees $\tau \geq 0.25$ without gradient discontinuity, preventing division-by-zero instability while allowing the model to learn arbitrarily soft assignments during early training.

### 3.3 Sinkhorn-Knopp with Adaptive EMA Prior ($\mathcal{L}_{\text{cluster}}$)

To prevent the prototypes from collapsing into a single generic language, SBERTa uses **Optimal Transport (Sinkhorn-Knopp)**.

Instead of forcing a rigid $1/K$ uniform split, it uses an **Adaptive EMA prior** that tracks the batch marginals over time. This allows the model to dynamically discover the true language distribution of the corpus. The algorithm produces a soft assignment matrix $\mathbf{Q}$ satisfying:
1. Every token has a valid probability distribution (rows sum to 1).
2. Every prototype receives a share of the batch's total mass proportional to the learned EMA prior.

The algorithm converges in 3–5 iterations and operates on batch-level statistics, adding negligible overhead.

A Cross-Entropy loss forces the model's distributions to match these balanced targets:
$$\mathcal{L}_{\text{cluster}} = \text{CrossEntropy}(\mathbf{p}, \mathbf{Q})$$

### 3.4 Pairwise Language Divergence

The fundamental structural measure is the pairwise divergence between any two positions:
$$\delta_{ij} = 1 - \mathbf{p}_i^\top \mathbf{p}_j \in [0, 1]$$

**Interpretation:**
- $\delta_{ij} \approx 0$: Positions $i$ and $j$ share the same language distribution.
- $\delta_{ij} \approx 1$: Maximally distinct language states.
- $\delta_{ij} \in (0,1)$: Partial overlap (mixed tokens, transition zones, borrowings).

The sequential switch magnitude is the natural restriction to consecutive tokens ($s_t = \delta_{t,t-1}$). This requires no extra parameters and generalizes the concept of a language boundary from a local sequence property to a pairwise structural relationship.

### 3.5 Orthogonality Regularization ($\mathcal{L}_{\text{ortho}}$)

While Sinkhorn-Knopp dynamically prevents prototype collapse via transport constraints, an explicit geometric prior on prototype directions accelerates early training and guards against pathological initializations. Let $\mathbf{L}_n$ denote row-normalized prototypes. The regularization penalizes off-diagonal correlation in their Gram matrix:
$$\mathcal{L}_{\text{ortho}} = \frac{1}{K^2}\left|\mathbf{L}_n \mathbf{L}_n^\top - \mathbf{I}\right|_F^2$$

This is a soft constraint; the transport loss remains the primary diversity mechanism, while $\mathcal{L}_{\text{ortho}}$ ensures prototypes maintain geometric separation independent of batch assignment dynamics.

---

## 4. Language-Aware Attention (Phase 2)

In Phase 2 layers, standard transformer attention is augmented with a structural prior that interacts with semantic similarity through learned per-head scaling. All language-specific parameters are initialized to zero, meaning every head begins as a standard transformer and discovers language structure organically during training.

The attention score for head $h$ is:
$$\text{score}_h(i, j) = \frac{\mathbf{Q}_{h,i} \mathbf{K}_{h,j}^\top}{\sqrt{d_h}} + \beta_h \cdot \mathbf{p}_i^\top \mathbf{C}_h \mathbf{p}_j + \gamma_h \cdot \delta_{ij}$$

where:
- $h \in \{1, \ldots, H\}$ indexes attention heads.
- $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ is a per-head asymmetric language compatibility matrix, initialized to $\mathbf{0}$.
- $\beta_h, \gamma_h \in \mathbb{R}$ are per-head learnable scalars, initialized to $0.01$. They allow each head to discover how much structural information to incorporate; some heads may specialize in pure semantics while others become cross-lingual routers.
- $\delta_{ij} = 1 - \mathbf{p}_i^\top \mathbf{p}_j$ is the pairwise language divergence.

The per-head scalars ensure that the language terms are on a comparable scale with the semantic term. The divergence $\delta_{ij}$ is inherently query-dependent, allowing the model to learn that an Arabic query attending to a Latin key should receive a different structural bias than a Latin query attending to an Arabic key. Because $\mathbf{C}_h$ and the scalars start at zero, the model initially trains as a vanilla transformer; language-aware routing emerges only where the data justifies it.

---

## 5. ELECTRA-Style Pre-training & GDES

### 5.1 Architecture Overview

**Generator** (small):
- Hidden size: $d_{\text{gen}} = d / 2$
- Attention heads: $H_{\text{gen}} = H / 2$
- Layers: $L_{\text{gen}} = L / 2$

**Discriminator** (full SBERTa):
- Full two-phase architecture
- Binary classification head: $\mathbb{R}^d \to \mathbb{R}$

The generator is smaller to ensure it produces plausible but detectable errors. Because the token embedding table is shared at full dimension $d$, the generator receives high-quality input representations despite its reduced capacity.

### 5.2 Gradient-Disentangled Embedding Sharing (GDES)

To solve the notorious instability of ELECTRA models, SBERTa implements DeBERTaV3's **GDES**.

The generator and discriminator share the token embedding table, but gradients from the discriminator's RTD loss do not flow into it. Only the generator's MLM loss updates the shared embeddings.
This prevents the "gradient tug-of-war" where the discriminator tries to push similar embeddings apart to detect fakes. The generator builds clean semantic clusters, and the discriminator detaches its embedding lookup (`stop_embedding_grad=True`), ensuring the Sinkhorn algorithm clusters true semantic representations.

### 5.3 Geometric Span Masking

Instead of standard random token masking, SBERTa masks geometric spans.
Span lengths are sampled from a Geometric distribution $\text{Geom}(p)$ with a mean length of $1/p$ tokens.
Crucially, this masking is language-agnostic and completely independent of $\mathbf{p}$. This allows the generator to focus purely on reconstructing context without relying on language predictions.

### 5.4 Phase-Separated Layer Normalisation

The encoder uses two independent `LayerNorm` modules rather than one shared final norm:

| Norm | Input | Gradient source | Purpose |
|------|-------|-----------------|---------|
| `phase1_norm` | $\mathbf{H}_{\text{base}}$ (Phase 1 output) | $\mathcal{L}_{\text{cluster}}$ only | Normalise for prototype cosine similarity |
| `final_norm` | $\mathbf{H}$ (Phase 2 output) | $\mathcal{L}_{\text{RTD}}$ only | Normalise for RTD binary head |

A shared norm would receive gradients from both objectives through representations at entirely different stages of processing, creating implicit competition between the clustering and discrimination objectives over shared normalisation parameters. Keeping them independent ensures each norm is optimised for its specific representational role.

## 6. Training Objectives

### 6.1 Generator Loss (MLM)

Masked language modeling on geometrically-masked spans:
$$\mathcal{L}_{\text{gen}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P_{\text{gen}}(x_t \mid \mathbf{x}_{\setminus \mathcal{M}})$$

### 6.2 Discriminator Loss (RTD)

Binary cross-entropy at every real token position:
$$\mathcal{L}_{\text{RTD}} = -\frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \text{BCE}(f_{\text{disc}}(\mathbf{h}_t), y_t)$$
Supervises all $T$ positions, providing 6-7× more gradient signal than vanilla MLM.

### 6.3 Combined Loss

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + w_{\text{rtd}} \cdot \mathcal{L}_{\text{RTD}} + \lambda_{\text{cluster}} \cdot \mathcal{L}_{\text{cluster}} + \lambda_{\text{ortho}} \cdot \mathcal{L}_{\text{ortho}}$$

**Default weights:**
- $w_{\text{rtd}} = 15.0$
- $\lambda_{\text{cluster}} = 3.0$
- $\lambda_{\text{ortho}} = 1.0$

---

## 7. Sentence-Level Representations

SBERTa does not use a [CLS] token. For sentence-level tasks, mean pooling over real tokens provides the semantic centroid:
$$\mathbf{z}_{\text{sem}} = \frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \mathbf{h}_t^{(L)}$$

Because code-switching structure is often discriminative, the language trajectory $\{\mathbf{p}_t\}_{t \in \mathcal{R}}$ provides a complementary structural signature. The sequential switch magnitude $s_t = 1 - \mathbf{p}_t^\top \mathbf{p}_{t-1}$ is derived directly from $\mathbf{p}$ and is returned as a diagnostic from `forward_phase1`; it is not an independent model output. For tasks requiring explicit switch-awareness, representations can be augmented with a pooled summary of the language trajectory. By default, mean pooling treats all languages democratically while the Phase-2 hidden states retain full language-aware context.

---

## 8. Design Decisions

### 8.1 Why a Two-Phase Encoder?
Deriving language distributions from raw embeddings is noisy and unreliable (the "bootstrap paradox"). Phase 1 builds rich, language-agnostic contextual representations first. When the language pivot happens, it has semantic context, meaning prototypes accurately cluster concepts rather than arbitrary surface features.

### 8.2 Why Sinkhorn-Knopp with Adaptive EMA Prior?
In purely unsupervised setups, models easily suffer from "prototype collapse". Sinkhorn mathematically guarantees balanced assignments. By using an Adaptive EMA prior rather than a uniform split (1/K), the model discovers the true natural frequency of the corpus's languages instead of fighting against it.

### 8.3 Why GDES (Gradient-Disentangled Embedding Sharing)?
ELECTRA models suffer from a "gradient tug-of-war": the generator tries to make embeddings semantically rich, while the discriminator rips them apart to detect fakes. By preventing the RTD loss from updating the token embeddings, the embeddings remain semantically pure, which is an absolute necessity for unsupervised language clustering to work.

### 8.4 Why Geometric Span Masking?
Unlike masking that relies on early noisy predictions, geometric masking is language-agnostic. It forces the generator to reconstruct contiguous blocks of text without needing hints from the discriminator.

### 8.5 Why Zero-Initialized Language Parameters?
All per-head compatibility matrices $\mathbf{C}_h$ and structural scalars $\beta_h, \gamma_h$ are initialized to zero (or near-zero). This means the model begins training as a perfectly standard transformer. Language-aware routing emerges organically during optimization rather than being imposed a priori. Heads that find structural signals useful amplify them; heads that do not remain standard semantic attention heads.

### 8.6 Why Pairwise Divergence?
Defining switch magnitude as a purely sequential property $s_t = 1 - \mathbf{p}_t^\top \mathbf{p}_{t-1}$ is intuitive but insufficient for non-local attention, where the relevant structural relationship is between arbitrary positions $i$ and $j$. The pairwise divergence $\delta_{ij}$ generalizes this concept naturally, requires no parameters, and makes the attention bias inherently query-dependent.

### 8.7 Why Soft Prototype Assignments?
$K$ specifies the number of language attractors, not a hard partition. Because assignments are soft, ambiguous or mixed tokens naturally distribute mass across prototypes. The Adaptive EMA prior ensures that if the true number of dominant languages is less than $K$, the excess prototypes receive vanishingly small mass without destabilizing training.

### 8.8 Why a Smooth Temperature Floor?
A hard clamp on temperature introduces a gradient discontinuity and an arbitrary architectural constraint. Parameterizing $\tau$ via softplus with a stability floor $\tau_{\min} = 0.25$ guarantees numerical safety while preserving differentiability. The model can learn arbitrarily high temperatures (soft assignments) but is protected from precision collapse.

---

## 9. Model Configurations

| Config | $d$ | $L$ ($n_{\text{base}}$) | $H$ | FFN | $K$ | Params |
|--------|-----|-----|-----|-----|-----|--------|
| Small  | 256 | 4 (2) | 4   | 1024  | 2 | ~17M  |
| Medium | 512 | 8 (4) | 8   | 2048  | 2 | ~55M  |
| Base   | 768 | 12 (6)| 12  | 3072  | 2 | ~136M |
| Large  | 1024| 24 (12)| 16  | 4096  | 2 | ~394M |

$K = 2$ is optimal for bilingual mixtures. The architecture natively supports arbitrary $K$ for multi-lingual code-switching by simply changing the config.