# SBERTa: Architecture Specification

**SBERTa** (**S**witching **B**idirectional **E**ncoder **R**epresentations from **T**ransformers **a**rchitecture) is a universal, zero-knowledge transformer architecture for code-switched text. It features explicit unsupervised language modeling, Optimal Transport clustering, and stable ELECTRA-style pre-training.

---

## 1. Core Design Philosophy

SBERTa treats code-switching as a structural signal rather than noise. It makes two linguistic phenomena explicit architectural components:

1. **Language identity** — soft distributions over $K$ language prototypes.
2. **Language boundaries** — continuous switch magnitudes between consecutive tokens.

The model is **fully zero-knowledge**. It uses no Unicode priors, script IDs, or dictionaries. Language structure emerges purely from the Masked Language Modeling (MLM) distributional objective, stabilized by a Sinkhorn-Knopp clustering loss. This makes SBERTa a universal standard applicable to any K-language mixture (multi-script like Darija, or mono-script like Spanglish/Hinglish).

---

## 2. Language Prototypes and Clustering

### 2.1 Prototype Vectors

The model learns $K$ prototype vectors $\mathbf{L} = [\boldsymbol{\ell}_1, \ldots, \boldsymbol{\ell}_K] \in \mathbb{R}^{K \times d}$ representing language directions in the embedding space.

**Initialization:** Orthogonal initialization scaled by 0.5 to avoid softmax saturation:
$$\mathbf{L} \sim \text{Orthogonal}(\mathbb{R}^{K \times d}), \quad \mathbf{L} \leftarrow 0.5 \cdot \mathbf{L}$$

### 2.2 Pre-Contextual Language Distributions

For each token $t$, the distribution is computed purely from raw base embeddings $\mathbf{h}_{\text{base},t} = \mathbf{E}_{\text{tok}}(x_t) + \mathbf{E}_{\text{pos}}(t)$:

$$p^{(0)}_t = \text{softmax}\left(\frac{\mathbf{h}_{\text{base},t} \mathbf{L}^\top}{\tau}\right) \in \Delta^K$$

where $\tau$ is a learnable temperature parameter stored as $\log \tau$ for unconstrained optimization:
$$\tau = \exp(\log \tau), \quad \log \tau \in \mathbb{R}, \quad \tau \geq 0.25$$

### 2.3 Sinkhorn-Knopp Equipartition ($\mathcal{L}_{\text{cluster}}$)

To prevent the prototypes from collapsing into a single generic language, SBERTa uses **Optimal Transport (Sinkhorn-Knopp)**.

During the forward pass, the raw similarities between the batch's real token embeddings and the $K$ prototypes are passed to the Sinkhorn algorithm. It iteratively normalizes the matrix to produce a soft assignment matrix $\mathbf{Q}$ satisfying two constraints:
1. Every token has a valid probability distribution (rows sum to 1).
2. Every prototype receives an equal share of the batch's total mass (columns sum to $N/K$).

A Cross-Entropy loss forces the model's distributions to match these balanced targets:
$$\mathcal{L}_{\text{cluster}} = \text{CrossEntropy}(p^{(0)}, \mathbf{Q})$$

**Why it matters:** This mathematical guarantee prevents collapse without requiring heuristic usage thresholds or hardcoded script rules. The prototypes are forced to find the $K$ most distinct clusters in the embedding space.

### 2.4 Switch Magnitudes

Continuous measure of language change between consecutive tokens:

$$s_t = 1 - p^{(0)}_t{}^\top p^{(0)}_{t-1}, \quad s_1 = 0$$

Since $p^{(0)}_t \in \Delta^K$, the inner product $p^{(0)}_t{}^\top p^{(0)}_{t-1} \in [0, 1]$, thus $s_t \in [0, 1]$.

**Interpretation:**
- $s_t \approx 0$: Same language as previous token.
- $s_t \approx 1$: Complete language switch.
- $s_t \in (0, 1)$: Partial or gradual transition (ambiguous tokens, punctuation).

---

## 3. Language-Augmented Embeddings

### 3.1 Base Embeddings

$$\mathbf{h}_{\text{base},t} = \mathbf{E}_{\text{tok}}(x_t) + \mathbf{E}_{\text{pos}}(t)$$

No normalization or augmentation applied at this stage. Used to compute $p^{(0)}_t$ and $s_t$ before augmentation, avoiding circular dependency.

### 3.2 Augmentation

$$\mathbf{h}^{(0)}_t = \text{LayerNorm}\left(\mathbf{h}_{\text{base},t} + \sum_{k=1}^K p^{(0)}_{t,k} \, \mathbf{e}_k + s_t \, \mathbf{e}_{\text{sw}}\right)$$

where:
- $\mathbf{e}_k \in \mathbb{R}^d$ are learnable language embeddings (one per prototype).
- $\mathbf{e}_{\text{sw}} \in \mathbb{R}^d$ is a learnable switch embedding.

---

## 4. Language-Aware Attention

### 4.1 Attention Score Computation

Standard transformer attention with two additive language-aware biases:

$$\text{score}_h(i, j) = \frac{\mathbf{Q}_{h,i} \mathbf{K}_{h,j}^\top}{\sqrt{d_h}} + p^{(0)}_i{}^\top \mathbf{C}_h \, p^{(0)}_j + \gamma \, s_j$$

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
- Full architecture ($d$, $H$, $L$)
- Binary classification head: $\mathbb{R}^d \to \mathbb{R}$

### 5.2 Gradient-Disentangled Embedding Sharing (GDES)

To solve the notorious instability of ELECTRA models, SBERTa implements DeBERTaV3's **GDES**.

The generator and discriminator share the token embedding table, but **gradients from the discriminator's RTD loss do not flow into it**. Only the generator's MLM loss updates the shared embeddings.
This prevents the "gradient tug-of-war" where the discriminator tries to push similar embeddings apart to detect fakes. The generator builds clean semantic clusters, and the discriminator detaches its embedding lookup (`stop_embedding_grad=True`), ensuring the Sinkhorn algorithm clusters true semantic representations.

### 5.3 Geometric Span Masking

Instead of standard random token masking, SBERTa masks geometric spans.
Span lengths are sampled from a Geometric distribution $\text{Geom}(p)$ with a mean length of $1/p$ tokens. 
Crucially, this masking is language-agnostic and completely independent of $p^{(0)}$. This allows the generator to focus purely on reconstructing context (vital for the embeddings to cluster semantically) without relying on untrained language predictions in early steps.

---

## 6. Training Objectives

### 6.1 Generator Loss (MLM)

Masked language modeling on geometrically-masked spans:
$$\mathcal{L}_{\text{gen}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P_{\text{gen}}(x_t \mid \mathbf{x}_{\setminus \mathcal{M}})$$

### 6.2 Discriminator Loss (RTD)

Binary cross-entropy at every real token position:
$$\mathcal{L}_{\text{RTD}} = -\frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \text{BCE}(f_{\text{disc}}(\mathbf{h}_t), y_t)$$
Supervises all $T$ positions, providing 6-7× more gradient signal than vanilla MLM.

### 6.3 Temporal Stickiness Loss

Penalizes the mean switch magnitude over qualifying consecutive boundaries:
$$\mathcal{L}_{\text{smooth}} = \frac{1}{|\mathcal{B}|} \sum_{(t-1,\,t) \in \mathcal{B}} s_t$$
Cross-document boundaries are excluded. Acts as a Markov prior pushing prototypes to form long, linguistically coherent spans.

### 6.4 Combined Loss

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + w_{\text{rtd}} \cdot \mathcal{L}_{\text{RTD}} + \lambda_{\text{smooth}} \cdot \mathcal{L}_{\text{smooth}} + \lambda_{\text{cluster}} \cdot \mathcal{L}_{\text{cluster}}$$

**Default weights:**
- $w_{\text{rtd}} = 15.0$
- $\lambda_{\text{smooth}} = 5.0$
- $\lambda_{\text{cluster}} = 1.0$

---

## 7. Gradient Flow Analysis

### 7.1 What Updates Prototypes

**Direct gradients:**
- $\mathcal{L}_{\text{cluster}}$: Directly shapes the prototypes $\mathbf{L}$ to match the balanced target distribution $\mathbf{Q}$.
- $\mathcal{L}_{\text{smooth}}$: Through `get_switch_magnitudes()` → through `get_distributions()` → through $\mathbf{L}$ directly, penalizing rapid switching.

**Indirect gradients:**
- $\mathcal{L}_{\text{RTD}}$: Through embedding augmentation and attention biases, pushing the prototypes to aid in fake detection.

### 7.2 Key Insight

Because of **GDES**, the token embeddings are updated *only* by the Generator's $\mathcal{L}_{\text{gen}}$. This means the embeddings naturally form semantic clusters based on text context. The Sinkhorn algorithm ($\mathcal{L}_{\text{cluster}}$) then looks at these pristine semantic embeddings and easily divides them into $K$ prototypes without fighting against the discriminator's RTD loss. This creates an incredibly stable learning dynamic.

---

## 8. Sentence-Level Representations

SBERTa does not use a [CLS] token. For sentence-level tasks, use **mean pooling**:
$$\mathbf{z} = \frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \mathbf{h}_t^{(L)}$$
This is because code-switching is a per-token property, and mean pooling treats all languages democratically across the sequence.

---

## 9. Design Decisions

### 9.1 Why Sinkhorn-Knopp Clustering?
In purely unsupervised setups without Sinkhorn, models easily suffer from "prototype collapse" (assigning all text to one language). Sinkhorn mathematically guarantees equipartition (equal distribution of tokens to prototypes) without relying on fragile heuristic thresholds, moving averages, or hardcoded dictionaries. 

### 9.2 Why GDES (Gradient-Disentangled Embedding Sharing)?
ELECTRA models suffer from a "gradient tug-of-war": the generator tries to make embeddings semantically rich, while the discriminator rips them apart to detect fakes. By preventing the RTD loss from updating the token embeddings, the embeddings remain semantically pure, which is an absolute necessity for unsupervised language clustering to work.

### 9.3 Why Geometric Span Masking?
Unlike the original SBERTa masking which relied on $p^{(0)}$ predictions that were noisy early in training, geometric masking is language-agnostic. It forces the generator to reconstruct contiguous blocks of text without needing any hints from the discriminator.

### 9.4 Why Per-Head Compatibility Matrices?
A scalar bias per head is symmetric and limited in expressiveness. A matrix $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ per head is asymmetric, allowing the model to learn that an Arabic query attending to a Latin key should receive a different bias than a Latin query attending to an Arabic key. The cost is only $K^2 H$ additional parameters.

---

## 10. Model Configurations

| Config | $d$ | $L$ | $H$ | FFN | $K$ | Params |
|--------|-----|-----|-----|-----|-----|--------|
| Small  | 256 | 4   | 4   | 1024  | 2 | ~16M  |
| Medium | 512 | 8   | 8   | 2048  | 2 | ~51M  |
| Base   | 768 | 12  | 12  | 3072  | 2 | ~124M |
| Large  | 1024| 24  | 16  | 4096  | 2 | ~355M |

$K = 2$ is optimal for bilingual mixtures (e.g. Arabic/Latin). The architecture natively supports arbitrary $K$ for multi-lingual code-switching by simply changing the config.

---

## 11. Summary

SBERTa is a universal, zero-knowledge code-switching architecture featuring:

1. **Explicit language modeling** via learned prototypes optimized by Optimal Transport (Sinkhorn-Knopp).
2. **Stable Pre-training** via ELECTRA-style RTD with Gradient-Disentangled Embedding Sharing (GDES).
3. **Geometric Span Masking** to ensure language-agnostic generator training.
4. **Fully unsupervised language boundary discovery** via temporal stickiness loss.
5. **Language-aware attention** with per-head $K \times K$ asymmetric compatibility matrices.
6. **Sentence-level representations** via democratic mean-pooling.