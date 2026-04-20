# SBERTa: Architecture Specification

**SBERTa** (**S**witching **B**idirectional **E**ncoder **R**epresentations from **T**ransformers **a**rchitecture) is a universal, zero-knowledge transformer architecture for code-switched text. It features explicit unsupervised language modeling, Optimal Transport clustering, and stable ELECTRA-style pre-training.

---

## 1. Core Design Philosophy

SBERTa treats code-switching as a structural signal rather than noise. It makes two linguistic phenomena explicit architectural components:

1. **Language identity** — soft distributions over $K$ language prototypes.
2. **Language boundaries** — continuous switch magnitudes between consecutive tokens.

The model is **fully zero-knowledge**. It uses no Unicode priors, script IDs, or dictionaries. Language structure emerges purely from the Masked Language Modeling (MLM) distributional objective, stabilized by a Sinkhorn-Knopp clustering loss. This makes SBERTa a universal standard applicable to any K-language mixture (multi-script like Darija, or mono-script like Spanglish/Hinglish).

---

## 2. Two-Phase Encoder

SBERTa avoids the "bootstrap paradox" of assigning language probabilities to noisy, raw token embeddings. Instead, it employs a two-phase design:

1. **Phase 1 — Context (layers $0 \dots n_{\text{base}}-1$):** Standard self-attention layers with no language bias injected. MLM distributional pressure forces contextualized, semantically rich representations ($\mathbf{h}_{\text{base}}$) to emerge naturally.
2. **Language Assignment Pivot:** Language distributions ($p$) and switch magnitudes ($s$) are computed from $\mathbf{h}_{\text{base}}$ at the boundary between phases. Because they are based on context, they are meaningful from the start.
3. **Phase 2 — Language-Aware (layers $n_{\text{base}} \dots L-1$):** Language signals are routed into the attention mechanism via compatibility matrices, guiding the rest of the network.

---

## 3. Language Prototypes and Clustering

### 3.1 Prototype Vectors

The model learns $K$ prototype vectors $\mathbf{L} = [\boldsymbol{\ell}_1, \ldots, \boldsymbol{\ell}_K] \in \mathbb{R}^{K \times d}$ representing language directions in the embedding space.

**Initialization:** Orthogonal initialization scaled by 0.5 to avoid softmax saturation:
$$\mathbf{L} \sim \text{Orthogonal}(\mathbb{R}^{K \times d}), \quad \mathbf{L} \leftarrow 0.5 \cdot \mathbf{L}$$

### 3.2 Contextual Language Distributions

For each token $t$, the distribution is computed from the output of Phase 1, $\mathbf{h}_{\text{base},t}$:

$$p_t = \text{softmax}\left(\frac{\mathbf{h}_{\text{base},t} \mathbf{L}^\top}{\tau}\right) \in \Delta^K$$

where $\tau$ is a learnable temperature parameter stored as $\log \tau$:
$$\tau = \exp(\log \tau), \quad \log \tau \in \mathbb{R}, \quad \tau \geq 0.25$$

### 3.3 Sinkhorn-Knopp with Adaptive EMA Prior ($\mathcal{L}_{\text{cluster}}$)

To prevent the prototypes from collapsing into a single generic language, SBERTa uses **Optimal Transport (Sinkhorn-Knopp)**.

Instead of forcing a rigid $1/K$ uniform split, it uses an **Adaptive EMA prior** that tracks the batch marginals over time. This allows the model to dynamically discover the true language distribution of the corpus. The algorithm produces a soft assignment matrix $\mathbf{Q}$ satisfying:
1. Every token has a valid probability distribution (rows sum to 1).
2. Every prototype receives a share of the batch's total mass proportional to the learned EMA prior.

A Cross-Entropy loss forces the model's distributions to match these balanced targets:
$$\mathcal{L}_{\text{cluster}} = \text{CrossEntropy}(p, \mathbf{Q})$$

### 3.4 Orthogonality Regularization ($\mathcal{L}_{\text{ortho}}$)

Directly regularizes prototype geometry by penalizing off-diagonal entries of the Gram matrix of normalized prototypes, preventing vectors from drifting together regardless of assignment dynamics:
$$\mathcal{L}_{\text{ortho}} = (\mathbf{L}_n \mathbf{L}_n^\top - \mathbf{I})^2.\text{mean}()$$

### 3.5 Switch Magnitudes

Continuous measure of language change between consecutive tokens:

$$s_t = 1 - p_t^\top p_{t-1}, \quad s_1 = 0$$

**Interpretation:**
- $s_t \approx 0$: Same language as previous token.
- $s_t \approx 1$: Complete language switch.
- $s_t \in (0, 1)$: Partial or gradual transition (ambiguous tokens, punctuation).

---

## 4. Language-Aware Attention (Phase 2)

In Phase 2 layers, SBERTa applies standard transformer attention with two additive language-aware biases:

$$\text{score}_h(i, j) = \frac{\mathbf{Q}_{h,i} \mathbf{K}_{h,j}^\top}{\sqrt{d_h}} + p_i^\top \mathbf{C}_h \, p_j + \gamma \, s_j$$

where:
- $h \in \{1, \ldots, H\}$ indexes attention heads.
- $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ is a per-head asymmetric language compatibility matrix.
- $\gamma \in \mathbb{R}$ is a global switch-position bias shared across all heads.

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

### 5.2 Gradient-Disentangled Embedding Sharing (GDES)

To solve the notorious instability of ELECTRA models, SBERTa implements DeBERTaV3's **GDES**.

The generator and discriminator share the token embedding table, but **gradients from the discriminator's RTD loss do not flow into it**. Only the generator's MLM loss updates the shared embeddings.
This prevents the "gradient tug-of-war" where the discriminator tries to push similar embeddings apart to detect fakes. The generator builds clean semantic clusters, and the discriminator detaches its embedding lookup (`stop_embedding_grad=True`), ensuring the Sinkhorn algorithm clusters true semantic representations.

### 5.3 Geometric Span Masking

Instead of standard random token masking, SBERTa masks geometric spans.
Span lengths are sampled from a Geometric distribution $\text{Geom}(p)$ with a mean length of $1/p$ tokens. 
Crucially, this masking is language-agnostic and completely independent of $p$. This allows the generator to focus purely on reconstructing context without relying on language predictions.

---

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

SBERTa does not use a [CLS] token. For sentence-level tasks, use **mean pooling**:
$$\mathbf{z} = \frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \mathbf{h}_t^{(L)}$$
This is because code-switching is a per-token property, and mean pooling treats all languages democratically across the sequence.

---

## 8. Design Decisions

### 8.1 Why a Two-Phase Encoder?
Deriving language distributions from raw embeddings is noisy and unreliable (the "bootstrap paradox"). Phase 1 builds rich, language-agnostic contextual representations first. When the language pivot happens, it has semantic context, meaning prototypes accurately cluster concepts rather than arbitrary surface features.

### 8.2 Why Sinkhorn-Knopp with Adaptive EMA Prior?
In purely unsupervised setups, models easily suffer from "prototype collapse". Sinkhorn mathematically guarantees balanced assignments. By using an Adaptive EMA prior rather than a uniform split ($1/K$), the model discovers the true natural frequency of the corpus's languages instead of fighting against it.

### 8.3 Why GDES (Gradient-Disentangled Embedding Sharing)?
ELECTRA models suffer from a "gradient tug-of-war": the generator tries to make embeddings semantically rich, while the discriminator rips them apart to detect fakes. By preventing the RTD loss from updating the token embeddings, the embeddings remain semantically pure, which is an absolute necessity for unsupervised language clustering to work.

### 8.4 Why Geometric Span Masking?
Unlike masking that relies on early noisy predictions, geometric masking is language-agnostic. It forces the generator to reconstruct contiguous blocks of text without needing hints from the discriminator.

### 8.5 Why Per-Head Compatibility Matrices?
A scalar bias per head is symmetric and limited in expressiveness. A matrix $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ per head is asymmetric, allowing the model to learn that an Arabic query attending to a Latin key should receive a different bias than a Latin query attending to an Arabic key. The cost is only $K^2 H$ additional parameters.

---

## 9. Model Configurations

| Config | $d$ | $L$ ($n_{\text{base}}$) | $H$ | FFN | $K$ | Params |
|--------|-----|-----|-----|-----|-----|--------|
| Small  | 256 | 4 (2) | 4   | 1024  | 2 | ~17M  |
| Medium | 512 | 8 (4) | 8   | 2048  | 2 | ~55M  |
| Base   | 768 | 12 (6)| 12  | 3072  | 2 | ~136M |
| Large  | 1024| 24 (12)| 16  | 4096  | 2 | ~394M |

$K = 2$ is optimal for bilingual mixtures. The architecture natively supports arbitrary $K$ for multi-lingual code-switching by simply changing the config.