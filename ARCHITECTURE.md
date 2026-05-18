# SBERTa: Architecture Specification

**SBERTa** (**S**witching **B**idirectional **E**ncoder **R**epresentations from **T**ransformers architecture) is a Darija-first transformer encoder for Algerian code-switched text. It is designed for Arabic script, Arabizi, French switches, Tamazight-influenced lexical material, and noisy social-media spelling.

The architecture keeps the stable parts of BERT/ELECTRA pre-training and adds an explicit, scheduled code-switching mechanism. The training recipe is intentionally narrow: a few public knobs, fixed internal constants, and measurable gates.

---

## 1. Core Design Philosophy

SBERTa treats Algerian code-switching as structure. It makes two phenomena explicit:

1. **Latent language/script state** — soft distributions over $K$ learned prototype states.
2. **Switching structure** — pairwise divergence between token-level prototype distributions.

The model does not use supervised language labels, script dictionaries, or Unicode-derived language tags in the encoder. However, the project scope is not “universal by default”: the current recipe is tuned for Algerian Darija, where the useful latent states are expected to cover Arabic-script Darija/MSA-like text, Arabizi/Roman script, French, and other local or borrowed forms.

---

## 2. Two-Phase Encoder

SBERTa avoids assigning language probabilities directly from raw token embeddings. Instead, the encoder is split into:

1. **Phase 1 — Context:** standard Pre-LN transformer layers with no language bias.
2. **Prototype Pivot:** soft prototype distributions are computed from contextual Phase-1 states.
3. **Phase 2 — Language-Aware Context:** later layers may use pairwise prototype structure inside attention.

This does not assume that switch boundaries are meaningful at random initialization. The language signal is treated as a learned diagnostic and is only allowed to affect attention after a scheduled warmup.

---

## 3. Language Prototypes

### 3.1 Prototype Vectors

The model learns $K$ prototype vectors:

$$\mathbf{L} = [\ell_1, \dots, \ell_K]^\top \in \mathbb{R}^{K \times d}$$

The Darija presets use:

$$K = 4$$

These states are soft attractors, not hard labels. They are intended to separate major modes in Algerian text:

- Arabic-script Darija / MSA-like material,
- Arabizi / Roman-script Darija,
- French,
- other local, borrowed, or mixed forms.

Prototypes are orthogonally initialized and scaled to avoid early softmax saturation.

### 3.2 Contextual Distributions

After Phase 1, token $t$ receives:

$$\mathbf{p}_t = \text{softmax}\left(\frac{\mathbf{h}_t \mathbf{L}_{norm}^\top}{\tau}\right)$$

where $\mathbf{L}_{norm}$ is row-normalized. The default temperature is fixed at:

$$\tau = 0.5$$

Temperature is kept fixed in the default recipe to remove a low-ROI training knob and avoid early assignment sharpening.

---

## 4. Sinkhorn Clustering And Prior

Sinkhorn-Knopp produces soft assignment targets $\mathbf{Q}$ with valid row marginals and prototype usage controlled by an adaptive prior.

The important stability correction is:

> The EMA prior is updated from the model’s raw probabilities $\mathbf{p}$, not from the Sinkhorn-constrained assignments $\mathbf{Q}$.

Sinkhorn uses the prior as its column target. Updating the prior from $\mathbf{Q}$ would make the prior mostly repeat its own constraint. Updating it from $\mathbf{p}$ lets it track the model’s unconstrained view of the corpus distribution.

The clustering loss is:

$$\mathcal{L}_{cluster} = \text{CrossEntropy}(\text{scores}, \mathbf{Q})$$

It is multiplied by a schedule:

$$\lambda_{cluster}(t) = 3.0 \cdot \text{cluster\_scale}(t)$$

---

## 5. Pairwise Switching Structure

The pairwise language divergence is:

$$\delta_{ij} = 1 - \mathbf{p}_i^\top \mathbf{p}_j$$

Interpretation:

- $\delta_{ij} \approx 0$: similar prototype state.
- $\delta_{ij} \approx 1$: strongly different prototype state.
- intermediate values: mixed, transitional, or ambiguous tokens.

The sequential switch magnitude is:

$$s_t = 1 - \mathbf{p}_t^\top \mathbf{p}_{t-1}$$

This is logged as a diagnostic. It is not treated as guaranteed boundary detection before training.

---

## 6. Language-Aware Attention

In Phase 2, attention can use prototype compatibility:

$$
\text{score}_h(i,j) =
\frac{\mathbf{Q}_{h,i}\mathbf{K}_{h,j}^{\top}}{\sqrt{d_h}}
+ \alpha(t)\left[
\beta_h \mathbf{p}_i^\top \mathbf{C}_h \mathbf{p}_j
+ \gamma_h \delta_{ij}
\right]
$$

where:

- $\mathbf{C}_h \in \mathbb{R}^{K \times K}$ is a per-head compatibility matrix,
- $\beta_h$ and $\gamma_h$ are learned per-head scalars,
- $\alpha(t)$ is the scheduled `lang_bias_scale`.

At the start of training, `lang_bias_scale = 0`. This makes the model train as a standard transformer before it relies on learned code-switching structure.

---

## 7. ELECTRA-Style Pre-Training And GDES

SBERTa uses a small generator and a full discriminator.

**Generator**

- Receives geometrically masked input.
- Predicts masked tokens with MLM loss.
- Shares the token embedding table with the discriminator.

**Discriminator**

- Receives the generator-corrupted sequence.
- Predicts whether each real token is original or replaced.
- Uses GDES: discriminator gradients do not update shared token embeddings.

This follows the ELECTRA and DeBERTaV3 lesson that RTD is powerful but can destabilize embeddings if the discriminator is allowed to push semantically close tokens apart.

Special tokens are excluded from both masking and generator replacement:

- `[PAD]`
- `[UNK]`
- `[MASK]`
- `[SEP]`

---

## 8. Training Objective

The full objective is:

$$
\mathcal{L} =
\mathcal{L}_{gen}
+ 15.0 \cdot \mathcal{L}_{RTD}
+ \text{cluster\_scale}(t) \cdot 3.0 \cdot \mathcal{L}_{cluster}
+ \text{ortho\_scale}(t) \cdot 1.0 \cdot \mathcal{L}_{ortho}
$$

where:

- $\mathcal{L}_{gen}$ is generator MLM over masked positions,
- $\mathcal{L}_{RTD}$ is discriminator binary cross-entropy over real tokens,
- $\mathcal{L}_{cluster}$ is Sinkhorn target matching,
- $\mathcal{L}_{ortho}$ keeps prototype vectors geometrically separated.

---

## 9. Training Schedule

The default recipe has three stages:

| Stage | Token progress | Behavior |
| --- | ---: | --- |
| A | 0-5% | Generator MLM + RTD only |
| B | 5-15% | Cluster and orthogonality losses ramp in |
| C | 15-25% | Language-aware attention ramps in if prototype entropy is healthy |

Language-aware attention remains disabled if prototype entropy is below the configured threshold. This prevents Phase 2 from specializing around collapsed or noisy assignments.

---

## 10. Fixed Recipe Constants

These are intentionally not public CLI knobs:

| Setting | Value |
| --- | ---: |
| Vocabulary size | 50,000 |
| Max sequence length | 512 |
| Number of prototype states | 4 |
| Dropout | 0.1 |
| MLM probability | 0.15 |
| RTD weight | 15.0 |
| Generator size divisor | 2 |
| Span mask geometric p | 0.2 |
| Sinkhorn epsilon / iterations | 0.1 / 20 |
| Prototype temperature | 0.5 fixed |
| Prior EMA momentum | 0.98 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Warmup | 6% of estimated steps |
| Gradient clipping | 1.0 |

The public training knobs are limited to preset, corpus, tokenizer, total tokens, micro-batch size, gradient accumulation, and run ID.

---

## 11. Sentence Representations

SBERTa does not use `[CLS]`. Sentence-level representations use mean pooling over non-padding final hidden states:

$$
\mathbf{z} =
\frac{1}{|\mathcal{R}|}
\sum_{t \in \mathcal{R}} \mathbf{h}_t^{(L)}
$$

For tasks where switching structure matters, downstream heads can optionally consume pooled prototype trajectories or switch statistics. The default benchmark harness uses mean-pooled encoder states for a clean DziriBERT comparison.

---

## 12. Presets

| Preset | $d$ | Layers | Phase split | Heads | FFN | $K$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `darija-small` | 256 | 4 | 2 + 2 | 4 | 1024 | 4 |
| `darija-medium` | 512 | 8 | 4 + 4 | 8 | 2048 | 4 |
| `darija-base` | 768 | 12 | 6 + 6 | 12 | 3072 | 4 |

`darija-medium` is the recommended single-GPU default. `darija-base` is the DziriBERT-scale comparison target.

---

## 13. Stability Metrics

A run is not considered healthy just because loss decreases. The trainer logs:

- generator loss and perplexity,
- RTD accuracy, precision, recall, and F1,
- replacement rate,
- cluster and orthogonality loss,
- prototype entropy,
- adaptive prototype prior,
- switch magnitude mean/max,
- gradient norm,
- AMP skipped steps.

These metrics are part of the training contract, not optional decoration.
