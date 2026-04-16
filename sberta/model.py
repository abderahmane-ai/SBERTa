"""
SBERTa model implementation.

Architecture summary
--------------------
Pre-training flow (per forward pass):

  1.  h_base   = E_tok(x) + E_pos(t)
  2.  p⁽⁰⁾_t  = softmax(h_base · Lᵀ / τ)              [pre-contextual language dist]
  3.  p⁽ctx⁾_t = ContextualLanguageRefinement(h_base)  [context-refined language dist]
  4.  s_t      = 1 − p⁽⁰⁾_t ᵀ p⁽⁰⁾_{t−1}, s₁ = 0    [continuous switch magnitude]
  5.  h⁽⁰⁾    = LN(h_base + Σₖ p⁽⁰⁾_{t,k} eₖ + sₜ · e_sw)
  6.  Each encoder layer ℓ:
        S_h(i,j) = (Qᵢ·Kⱼ)/√dₕ + pᵢᵀ Cₕ pⱼ + γ·sⱼ   [Cₕ ∈ ℝ^{K×K} per head]
        H⁽ˡ⁾   = LN(H + MHA(H, p⁽ctx⁾, s)) + FFN(·))

Pre-training objectives (ELECTRA-style RTD):
  Generator:     SBERTaGenerator (hidden_size // generator_size_divisor) — MLM on
                 switch-span-masked input; proposes plausible token replacements.
  Discriminator: full SBERTa — RTD binary classification at every token position,
                 giving 6-7× more gradient signal than vanilla 15%-masked MLM.

  L = L_gen + w_rtd · L_RTD + λ_smooth · w_curr · L_smooth + λ_sharp · L_sharp + λ_div · L_div

  L_gen        : generator MLM (span-masked positions only; normalised by n_masked)
  L_RTD        : replaced token detection — supervises every real token position
  L_smooth     : unsupervised temporal stickiness — mean switch magnitude penalised
                 with a curriculum schedule; no external labels required
  L_sharp      : per-token prototype commitment — minimises assignment entropy so
                 each token commits to one language rather than hedging uniformly
  L_div        : prototype diversity — prevents prototype collapse
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SBERTaConfig


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _masked_mean_pool(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool hidden states over real (non-padding) token positions.

    Args:
        H:    (B, T, d)
        mask: (B, T) binary — 1 for real tokens, 0 for padding
    Returns:
        (B, d)
    """
    mask_f = mask.float().unsqueeze(-1)              # (B, T, 1)
    return (H * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


def _switch_span_mask(
    p: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask_prob: float = 0.15,
) -> torch.Tensor:
    """
    Switch-span masking for the ELECTRA generator.

    Groups consecutive tokens with the same dominant language (argmax p) into
    language-homogeneous spans, then randomly selects whole spans until
    ~target_mask_prob of real tokens are covered.

    Masking entire spans — rather than random individual tokens — forces the
    generator to reconstruct a language segment from cross-language context,
    which directly targets SBERTa's code-switching objective.

    Args:
        p:                (B, T, K) pre-contextual language distributions
                          (computed from the original unmasked input)
        attention_mask:   (B, T) binary — 1 for real tokens, 0 for padding
        target_mask_prob: target fraction of real tokens to mask
    Returns:
        span_mask: (B, T) bool — True at positions selected for masking
    """
    B, T, _ = p.shape
    lang_ids = p.argmax(dim=-1)                      # (B, T)
    span_mask = torch.zeros(B, T, dtype=torch.bool, device=p.device)

    for b in range(B):
        real_len = int(attention_mask[b].sum().item())
        if real_len == 0:
            continue
        langs = lang_ids[b, :real_len]

        # Span start positions: index 0 always starts a span; later positions
        # start a span wherever the dominant language changes.
        is_start = torch.cat([
            torch.ones(1, dtype=torch.bool, device=p.device),
            langs[1:] != langs[:-1],
        ])                                           # (real_len,)
        starts = is_start.nonzero(as_tuple=True)[0]  # (n_spans,)
        ends = torch.cat([starts[1:], starts.new_tensor([real_len])])

        target_n = max(1, int(real_len * target_mask_prob))
        starts_list = starts.tolist()
        ends_list = ends.tolist()

        n_masked = 0
        for idx in torch.randperm(len(starts_list), device=p.device).tolist():
            if n_masked >= target_n:
                break
            s, e = starts_list[idx], ends_list[idx]
            span_mask[b, s:e] = True
            n_masked += e - s

    return span_mask                                 # (B, T)


# ─── Contextual Language Refinement ──────────────────────────────────────────


class ContextualLanguageRefinement(nn.Module):
    """
    Stage-2 context-aware language distribution computation.

    A lightweight windowed self-attention module (±window tokens) over h_base
    that produces context-refined language distributions p⁽ctx⁾. These fix the
    fundamental weakness of purely pre-contextual prototype assignment, which
    fails for Latin-script ambiguity:

      'chat'  — French (cat) or English depending on surrounding words
      'la'    — French negation or Arabic لا depending on script context
      'est'   — French or Spanish copula
      'ma'    — French possessive or Arabic particle
      Arabizi tokens: context-dependent relative to adjacent Arabic/French spans

    p⁽ctx⁾ is used for the attention compatibility biases (Cₕ and γ) in every
    encoder layer. Embedding augmentation still uses the pre-contextual p⁽⁰⁾ to
    avoid a circular dependency.
    """

    def __init__(self, config: SBERTaConfig, window: int = 3) -> None:
        super().__init__()
        d: int = config.hidden_size
        d_small: int = d // 4             # lightweight projection keeps the module cheap
        self.window: int = window
        self.scale: float = math.sqrt(d_small)

        self.W_Q: nn.Linear = nn.Linear(d, d_small, bias=False)
        self.W_K: nn.Linear = nn.Linear(d, d_small, bias=False)
        self.proj: nn.Linear = nn.Linear(d, config.num_languages, bias=False)

    def _windowed_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Additive attention mask: 0.0 inside ±window, −inf outside.
        Shape: (T, T).
        """
        positions = torch.arange(T, device=device)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (T, T)
        return torch.where(
            dist <= self.window,
            torch.zeros(T, T, device=device),
            torch.full((T, T), float("-inf"), device=device),
        )

    def forward(self, h_base: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_base: (B, T, d) — raw base embeddings before language augmentation
            tau:    scalar temperature tensor (LanguagePrototypes.tau)
        Returns:
            p_ctx: (B, T, K) — context-refined language distributions
        """
        T = h_base.size(1)
        Q = self.W_Q(h_base)             # (B, T, d_small)
        K_ = self.W_K(h_base)            # (B, T, d_small)

        scores = (
            torch.bmm(Q, K_.transpose(1, 2)) / self.scale
            + self._windowed_mask(T, h_base.device).unsqueeze(0)
        )                                # (B, T, T)

        ctx = torch.bmm(F.softmax(scores, dim=-1), h_base)  # (B, T, d)
        return F.softmax(self.proj(ctx) / tau, dim=-1)       # (B, T, K)


# ─── Language Prototypes ──────────────────────────────────────────────────────


class LanguagePrototypes(nn.Module):
    """
    K learned prototype vectors L ∈ ℝ^{K×d} with learnable temperature τ.

    Responsibilities:
      · Compute pre-contextual language distributions p⁽⁰⁾_t from raw h_base.
      · Compute continuous switch magnitudes s_t.
      · Return prototype diversity regularisation loss L_div.

    Temperature τ is stored as log_τ for unconstrained optimisation and
    retrieved via the .tau property as exp(log_τ), always positive.
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        self.K: int = config.num_languages

        # L ∈ ℝ^{K×d}: orthogonal init → well-separated starting geometry
        self.prototypes: nn.Parameter = nn.Parameter(
            torch.empty(config.num_languages, config.hidden_size)
        )
        with torch.no_grad():
            nn.init.orthogonal_(self.prototypes)
            self.prototypes.mul_(0.5)    # scale down to avoid softmax saturation at τ=0.5

        # Temperature: log_τ for unconstrained optimisation
        log_tau_init = math.log(config.proto_temperature)
        if config.learnable_temperature:
            self.log_tau: nn.Parameter = nn.Parameter(torch.tensor(log_tau_init))
        else:
            self.register_buffer("log_tau", torch.tensor(log_tau_init))

    @property
    def tau(self) -> torch.Tensor:
        """Always-positive temperature: τ = exp(log_τ), floored at 0.25.

        The floor prevents the model from sharpening past the point where
        prototype imbalance recovery becomes impossible. Without it, a
        collapsed prototype at very low τ mathematically locks others out.
        """
        return self.log_tau.exp().clamp(min=0.25)

    def get_distributions(self, h: torch.Tensor) -> torch.Tensor:
        """
        p⁽⁰⁾_t = softmax(h_t Lᵀ / τ)  — pre-contextual language distributions.

        Args:
            h: (B, T, d) — raw base embeddings (tok + pos, no augmentation)
        Returns:
            p: (B, T, K)
        """
        return F.softmax(h @ self.prototypes.T / self.tau, dim=-1)

    def get_switch_magnitudes(self, p: torch.Tensor) -> torch.Tensor:
        """
        s_t = 1 − p_t ᵀ p_{t−1},  s₁ = 0.

        Since p ∈ Δᴷ, the inner product pᵢᵀpⱼ ∈ [0, 1] and s_t ∈ [0, 1].

        Args:
            p: (B, T, K)
        Returns:
            s: (B, T)
        """
        sim = (p[:, 1:] * p[:, :-1]).sum(dim=-1)              # (B, T−1)
        zeros = torch.zeros(p.size(0), 1, device=p.device, dtype=p.dtype)
        return torch.cat([zeros, 1.0 - sim], dim=1)           # (B, T)


    def diversity_loss(self) -> torch.Tensor:
        """
        Margin-based repulsion loss:
            L_div = (2 / K(K−1)) · Σ_{i<j} relu(cos(ℓᵢ, ℓⱼ) + margin)²

        The margin (default 0.1) means the loss fires even when prototypes are
        near-orthogonal (cos ≈ 0), giving a constant repulsion gradient from
        step 0.  This fixes the timing asymmetry where the original cos²
        formulation produced zero gradient at initialisation while L_smooth
        was already pushing tokens toward shared prototypes.

        At perfect orthogonality (cos = 0): loss = relu(0.1)² = 0.01 per pair.
        At collapse (cos = 1):              loss = relu(1.1)² = 1.21 per pair.
        Only when cos < −margin does the loss reach zero (better than orthogonal).
        """
        _margin: float = 0.1
        L_n = F.normalize(self.prototypes, dim=-1)             # (K, d)
        cos = L_n @ L_n.T                                      # (K, K)
        mask = torch.triu(
            torch.ones(self.K, self.K, device=cos.device), diagonal=1
        )
        return (F.relu(cos + _margin).pow(2) * mask).sum() / (self.K * (self.K - 1) / 2.0)


# ─── Input Embeddings ─────────────────────────────────────────────────────────


class SBERTaEmbeddings(nn.Module):
    """
    h⁽⁰⁾ = LN(E_tok(x) + E_pos(t) + Σₖ p⁽⁰⁾_{t,k} eₖ + sₜ · e_sw)

    Construction is split into two stages so that p⁽⁰⁾ and s are derived from
    the raw (tok + pos) embedding before augmentation, avoiding the circular
    dependency that would arise if prototypes depended on the augmented vector.
    Encoder attention layers use the separately-computed context-refined p⁽ctx⁾.
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        d: int = config.hidden_size
        self.token_embeddings: nn.Embedding = nn.Embedding(
            config.vocab_size, d, padding_idx=config.pad_token_id
        )
        self.position_embeddings: nn.Embedding = nn.Embedding(
            config.max_position_embeddings, d
        )
        # eₖ^(lang): one learnable vector per language prototype slot
        self.language_embeddings: nn.Embedding = nn.Embedding(config.num_languages, d)
        # e^(switch): a single learnable vector scaled by switch magnitude
        self.switch_embedding: nn.Parameter = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.switch_embedding, mean=0.0, std=0.02)
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.dropout: nn.Dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_base(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Stage 1: tok + pos embeddings without normalisation or augmentation.
        Used to compute p⁽⁰⁾ and s before augmentation.

        Args:
            input_ids: (B, T)
        Returns:
            h_base: (B, T, d)
        """
        T: int = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return (
            self.token_embeddings(input_ids)
            + self.position_embeddings(positions)
        )

    def augment(
        self,
        h_base: torch.Tensor,
        p: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage 2: add language and switch signals, apply LN + dropout.
        Always uses pre-contextual p⁽⁰⁾ to avoid circular dependency.

        Args:
            h_base: (B, T, d)
            p:      (B, T, K) — pre-contextual p⁽⁰⁾
            s:      (B, T)    — switch magnitudes
        Returns:
            h: (B, T, d)
        """
        lang_emb = p @ self.language_embeddings.weight    # (B, T, d)
        sw_emb = s.unsqueeze(-1) * self.switch_embedding  # (B, T, d)
        return self.dropout(self.layer_norm(h_base + lang_emb + sw_emb))


# ─── Language-Aware Multi-Head Self-Attention ─────────────────────────────────


class SBERTaAttention(nn.Module):
    """
    Multi-head self-attention with per-head K×K language-compatibility biases.

      S_h(i,j) = (Qᵢ · Kⱼ) / √dₕ  +  pᵢᵀ Cₕ pⱼ  +  γ · sⱼ

    pᵢᵀ Cₕ pⱼ  — Cₕ ∈ ℝ^{K×K} per head, replacing the previous scalar β_h.
                   Initialised to identity (matching original β_h behaviour).
                   Learns asymmetric language affinities: Arabic↔Arabizi high,
                   French↔Arabic medium, etc. Adds only K²×H parameters (192
                   for base config) at negligible cost.
    γ · sⱼ      — global switch-position bias toward language boundary tokens.

    Both Cₕ (as identity) and γ (as zero) are initialised so that training
    starts from standard attention and learns biases only when beneficial.
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        d, H, K = config.hidden_size, config.num_attention_heads, config.num_languages
        assert d % H == 0, "hidden_size must be divisible by num_attention_heads"
        dh = d // H

        self.H, self.dh, self.d, self.K = H, dh, d, K
        self.scale: float = math.sqrt(dh)

        self.W_Q: nn.Linear = nn.Linear(d, d, bias=False)
        self.W_K: nn.Linear = nn.Linear(d, d, bias=False)
        self.W_V: nn.Linear = nn.Linear(d, d, bias=False)
        self.W_O: nn.Linear = nn.Linear(d, d)

        # Per-head K×K language compatibility matrix Cₕ ∈ ℝ^{H×K×K}
        # Identity init → equivalent to scalar β_h = 1 at the start of training
        self.compat: nn.Parameter = nn.Parameter(
            torch.eye(K).unsqueeze(0).expand(H, -1, -1).clone()  # (H, K, K)
        )
        # Global switch-position scalar γ, init to zero
        self.gamma: nn.Parameter = nn.Parameter(torch.zeros(1))
        self.attn_drop: nn.Dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d) → (B, H, T, dh)"""
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.dh).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, dh) → (B, T, d)"""
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d)

    def forward(
        self,
        H: torch.Tensor,                               # (B, T, d)
        p: torch.Tensor,                               # (B, T, K) — p⁽ctx⁾
        s: torch.Tensor,                               # (B, T)
        additive_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, T)
    ) -> torch.Tensor:
        B, T, _ = H.shape

        Q = self._split(self.W_Q(H))                   # (B, H, T, dh)
        K_ = self._split(self.W_K(H))                  # (B, H, T, dh)
        V = self._split(self.W_V(H))                   # (B, H, T, dh)

        scores = torch.matmul(Q, K_.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # pᵢᵀ Cₕ pⱼ: for each head h, bias[b,h,i,j] = p[b,i] @ compat[h] @ p[b,j]
        p_compat = torch.einsum("bik,hkl->bhil", p, self.compat)      # (B, H, T, K)
        scores = scores + torch.einsum("bhil,bjl->bhij", p_compat, p) # (B, H, T, T)

        # γ · sⱼ
        scores = scores + self.gamma * s.view(B, 1, 1, T)

        if additive_mask is not None:
            scores = scores + additive_mask

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        return self.W_O(self._merge(torch.matmul(attn, V)))


# ─── SBERTa Encoder Layer ─────────────────────────────────────────────────────


class SBERTaLayer(nn.Module):
    """
    One SBERTa encoder layer (post-LN):

      H_attn = SBERTaAttention(H, p⁽ctx⁾, s)
      H_mid  = LN(H + dropout(H_attn))
      H⁽ˡ⁾  = LN(H_mid + dropout(FFN(H_mid)))
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        d: int = config.hidden_size
        self.attention: SBERTaAttention = SBERTaAttention(config)
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(d, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, d),
        )
        self.norm_attn: nn.LayerNorm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.norm_ffn: nn.LayerNorm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.dropout: nn.Dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        H: torch.Tensor,
        p: torch.Tensor,                                # (B, T, K) — p⁽ctx⁾
        s: torch.Tensor,                                # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) binary
    ) -> torch.Tensor:
        if attention_mask is not None:
            additive: Optional[torch.Tensor] = (
                (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -10_000.0
            )
        else:
            additive = None

        H_attn = self.attention(H, p, s, additive)
        H_mid = self.norm_attn(H + self.dropout(H_attn))
        return self.norm_ffn(H_mid + self.dropout(self.ffn(H_mid)))


# ─── Generator ────────────────────────────────────────────────────────────────


class _GeneratorLayer(nn.Module):
    """Standard post-LN transformer layer used internally by SBERTaGenerator."""

    def __init__(
        self, d: int, n_heads: int, ffn_dim: int, dropout: float, eps: float
    ) -> None:
        super().__init__()
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(d, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d)
        )
        self.norm1: nn.LayerNorm = nn.LayerNorm(d, eps=eps)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d, eps=eps)
        self.drop: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x + self.drop(attn_out))
        return self.norm2(x + self.drop(self.ffn(x)))


class SBERTaGenerator(nn.Module):
    """
    Lightweight MLM generator for ELECTRA-style RTD pre-training.

    Operates at hidden_size // generator_size_divisor (d_gen). Receives the
    discriminator's token embedding table as a weight tensor at forward time
    (avoids double-registration in the module tree while preserving the tie).

    Architecture:
      input_proj (d → d_gen) + pos_emb → LN → [_GeneratorLayer × n_layers]
        → mlm_dense → mlm_norm → output_proj (d_gen → d) → tied decoder → logits
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        d: int = config.hidden_size
        d_gen: int = d // config.generator_size_divisor
        n_heads: int = max(1, config.num_attention_heads // config.generator_size_divisor)
        n_layers: int = max(1, config.num_hidden_layers // config.generator_size_divisor)

        # Project shared token embeddings (d) down to generator space (d_gen)
        self.input_proj: nn.Linear = nn.Linear(d, d_gen, bias=False)
        self.pos_emb: nn.Embedding = nn.Embedding(config.max_position_embeddings, d_gen)
        self.norm_in: nn.LayerNorm = nn.LayerNorm(d_gen, eps=config.layer_norm_eps)

        self.layers: nn.ModuleList = nn.ModuleList([
            _GeneratorLayer(
                d_gen, n_heads, d_gen * 4,
                config.hidden_dropout_prob, config.layer_norm_eps
            )
            for _ in range(n_layers)
        ])

        # MLM head: d_gen → d_gen → d → V (tied decoder applied by caller)
        self.mlm_dense: nn.Linear = nn.Linear(d_gen, d_gen)
        self.mlm_norm: nn.LayerNorm = nn.LayerNorm(d_gen, eps=config.layer_norm_eps)
        self.output_proj: nn.Linear = nn.Linear(d_gen, d, bias=False)
        self.mlm_bias: nn.Parameter = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        token_emb_weight: torch.Tensor,    # shared with discriminator; passed at call-time
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:        (B, T) — span-masked token ids ([MASK] at selected spans)
            token_emb_weight: (V, d) — discriminator's token embedding weight (tied)
            attention_mask:   (B, T) binary
        Returns:
            logits: (B, T, V) — MLM logits over vocabulary
        """
        T = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        # Embed and project from d → d_gen
        h = self.norm_in(
            self.input_proj(F.embedding(input_ids, token_emb_weight))
            + self.pos_emb(positions)
        )                                                        # (B, T, d_gen)

        # MHA key_padding_mask: True where padded
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        for layer in self.layers:
            h = layer(h, pad_mask)

        # Project back to d for tied decoding
        h_mlm = self.mlm_norm(F.gelu(self.mlm_dense(h)))       # (B, T, d_gen)
        return self.output_proj(h_mlm) @ token_emb_weight.T + self.mlm_bias  # (B, T, V)


# ─── Bare SBERTa Encoder ──────────────────────────────────────────────────────


class SBERTaModel(nn.Module):
    """
    Bare SBERTa encoder with two-stage language distribution computation.

    Stage 1 (pre-contextual): p⁽⁰⁾_t from raw h_base — used for embedding
    augmentation; avoids circular dependency.

    Stage 2 (context-refined): p⁽ctx⁾_t from windowed attention over h_base —
    used for attention compatibility biases Cₕ and γ; fixes Latin-script
    ambiguity where p⁽⁰⁾ is unreliable.

    Typical fine-tuning usage:
        model = SBERTaModel(config)
        H, p_ctx, s = model(input_ids, attention_mask)
        sentence_repr = _masked_mean_pool(H, attention_mask)
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        self.config: SBERTaConfig = config
        self.prototypes: LanguagePrototypes = LanguagePrototypes(config)
        self.refinement: ContextualLanguageRefinement = ContextualLanguageRefinement(config)
        self.embeddings: SBERTaEmbeddings = SBERTaEmbeddings(config)
        self.layers: nn.ModuleList = nn.ModuleList(
            [SBERTaLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (B, T)
            attention_mask: (B, T) binary — 1 for real tokens, 0 for padding
        Returns:
            H:     (B, T, d) — final encoder hidden states
            p_ctx: (B, T, K) — context-refined language distributions
            s:     (B, T)    — switch magnitudes (derived from pre-contextual p⁽⁰⁾)
        """
        # Stage 1: tok + pos (no LN, no language augmentation yet)
        h_base = self.embeddings.get_base(input_ids)             # (B, T, d)

        # Stage 2a: pre-contextual language assignments + switch magnitudes
        p0 = self.prototypes.get_distributions(h_base)           # (B, T, K)
        s = self.prototypes.get_switch_magnitudes(p0)            # (B, T)

        # Stage 2b: context-refined language distributions (windowed attention)
        p_ctx = self.refinement(h_base, self.prototypes.tau)     # (B, T, K)

        # Stage 3: augmented h⁽⁰⁾ uses p⁽⁰⁾ — circular-dependency free
        H = self.embeddings.augment(h_base, p0, s)               # (B, T, d)

        # Stage 4: encoder stack — attention biases use p⁽ctx⁾
        for layer in self.layers:
            H = layer(H, p_ctx, s, attention_mask)

        return H, p_ctx, s


# ─── Pre-training Wrapper ─────────────────────────────────────────────────────


class SBERTaForPreTraining(nn.Module):
    """
    SBERTa with ELECTRA-style RTD pre-training.

    Generator (1/generator_size_divisor size):
      Receives switch-span-masked input; proposes plausible token replacements
      via MLM. Shares the token embedding weight with the discriminator —
      the weight is passed at forward time to avoid double module-tree
      registration while preserving the tie for state_dict and the optimiser.

    Discriminator (full SBERTa):
      Receives corrupted sequence (original tokens + generator replacements);
      classifies every real token as real or replaced. Supervised at all T
      positions — not just the ~15% masked — giving 6-7× more gradient signal
      than vanilla MLM for the same data budget.

    Training signals:
      L_gen        : generator MLM on masked spans (trains generator)
      L_RTD        : discriminator real/replaced BCE (trains full SBERTa)
      L_smooth     : unsupervised temporal stickiness — mean switch magnitude
                     penalised immediately from step 0 (w_min=0.05, ramps to 1.0
                     over smooth_warmup_steps) to self-organise prototypes into
                     linguistically coherent blocks; no external labels required
      L_sharp      : per-token prototype commitment — minimises assignment
                     entropy over real tokens, forcing sharp language choices
      L_div        : prototype diversity (prevents prototype collapse)

    Combined loss:
      L = L_gen + w_rtd · L_RTD + λ_smooth · w_curr · L_smooth
            + λ_sharp · L_sharp + λ_div · L_div

      w_curr = 0.05 + 0.95 × min(1, step / smooth_warmup_steps)
      Starts non-zero at step 0 to prevent the independent-sampling equilibrium.
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        self.config: SBERTaConfig = config
        self.sberta: SBERTaModel = SBERTaModel(config)
        self.generator: SBERTaGenerator = SBERTaGenerator(config)

        # Discriminator RTD head: binary real (0) vs replaced (1)
        self.rtd_head: nn.Linear = nn.Linear(config.hidden_size, 1)

        self.apply(self._init_weights)

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def _tok_w(self) -> torch.Tensor:
        """Discriminator token embedding weight — shared with the generator."""
        return self.sberta.embeddings.token_embeddings.weight

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,                         # (B, T)
        attention_mask: Optional[torch.Tensor] = None,   # (B, T) binary
        global_step: int = 0,                            # current training step (curriculum)
    ) -> dict:
        """
        Args:
            input_ids:      (B, T) — original (unmasked) token ids.
            attention_mask: (B, T) — 1 for real tokens, 0 for padding.
            global_step:    current optimiser step; used to scale L_smooth.
        Returns:
            dict with 'loss' (scalar) and per-component .item() loss values.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        real = attention_mask.bool()                               # (B, T)

        # ── Step 1: span mask from original input's language structure ────
        # p_pre is computed on unmasked input so span boundaries reflect true
        # language identity rather than the [MASK] token distribution.
        h_base = self.sberta.embeddings.get_base(input_ids)
        p_pre = self.sberta.prototypes.get_distributions(h_base)   # (B, T, K)
        s_pre = self.sberta.prototypes.get_switch_magnitudes(p_pre)  # (B, T)

        span_mask = _switch_span_mask(p_pre, attention_mask, self.config.mlm_probability)

        masked_ids = input_ids.clone()
        masked_ids[span_mask] = self.config.mask_token_id

        # ── Step 2: generator — single forward for both loss and sampling ─
        # Gradients flow through gen_logits for L_gen.
        # Sampling uses .detach() + no_grad so the corruption graph is clean.
        gen_logits = self.generator(masked_ids, self._tok_w, attention_mask)  # (B, T, V)

        with torch.no_grad():
            gen_tokens = torch.multinomial(
                F.softmax(gen_logits[span_mask].detach(), dim=-1),
                num_samples=1,
            ).squeeze(-1)                                          # (N_masked,)

        corrupted_ids = input_ids.clone()
        corrupted_ids[span_mask] = gen_tokens
        is_replaced = (corrupted_ids != input_ids).float()         # (B, T)

        # ── Step 3: discriminator forward on corrupted sequence ───────────
        H, p_ctx, s = self.sberta(corrupted_ids, attention_mask)

        # ── L_gen (generator MLM) ────────────────────────────────────────
        gen_labels = input_ids.new_full(input_ids.shape, -100)
        gen_labels[span_mask] = input_ids[span_mask]
        n_masked = max(int(span_mask.sum().item()), 1)
        loss_gen = F.cross_entropy(
            gen_logits.view(-1, self.config.vocab_size),
            gen_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        ) / n_masked

        # ── L_RTD (discriminator) ────────────────────────────────────────
        rtd_logits = self.rtd_head(H).squeeze(-1)                  # (B, T)
        loss_rtd = F.binary_cross_entropy_with_logits(
            rtd_logits[real],
            is_replaced[real],
            reduction="mean",
        )

        # RTD accuracy — fraction of real tokens correctly classified.
        # A discriminator predicting all-real scores ~(1 - replace_rate) ≈ 85%
        # without learning anything.  Meaningful accuracy is well above that.
        with torch.no_grad():
            rtd_preds = (rtd_logits[real] > 0.0).float()
            rtd_acc = (rtd_preds == is_replaced[real]).float().mean().item()

        # ── L_smooth (unsupervised temporal stickiness) ───────────────────
        # Penalise the mean switch magnitude over real consecutive token
        # boundaries.  Minimising E[s_t] forces the K prototypes to form long,
        # linguistically coherent spans rather than flipping per-token —
        # without any external labels or fastText dependency.
        #
        # Curriculum: starts immediately at λ_min=0.05 (never fully zero) and
        # ramps linearly to 1.0 over smooth_warmup_steps. The non-zero baseline
        # prevents the backbone from baking in statistically-independent
        # prototype assignments during a "blind" window that later smooth
        # pressure cannot escape.
        _lambda_min: float = 0.05
        smooth_weight = _lambda_min + (1.0 - _lambda_min) * min(
            1.0,
            global_step / max(1, self.config.smooth_warmup_steps),
        )
        # Only calculate s_t over boundaries where both the current and
        # previous token are real (excludes pad→real and real→pad edges).
        switch_mask = real[:, 1:] & real[:, :-1]            # (B, T-1)
        if switch_mask.any():
            loss_smooth = s_pre[:, 1:][switch_mask].mean()
        else:
            loss_smooth = input_ids.new_zeros(())

        # ── L_sharp (per-token prototype commitment) ──────────────────────
        # Minimise per-token assignment entropy over real (non-pad) tokens.
        # Forces each token to commit sharply to one prototype rather than
        # maintaining a near-uniform distribution across K.
        # This is the safe half of the information-theoretic objective:
        # we do NOT add the batch-balance term (H of the mean distribution)
        # which would force equal usage across K and would break the 65/35
        # Darija-French corpus ratio by hallucinating language boundaries.
        p0_real = p_pre[real]                                # (N_real, K)
        loss_sharp = -(p0_real * (p0_real + 1e-9).log()).sum(dim=-1).mean()

        # ── L_div ────────────────────────────────────────────────────────
        loss_div = self.sberta.prototypes.diversity_loss()

        # ── L_balance (soft minimum-usage) ───────────────────────────────
        # Fires only when a prototype's mean usage drops below min_usage
        # (1 / K*4 = 6.25% for K=4).  Does NOT force equal distribution —
        # it only rescues dying prototypes, preserving the natural 65/35
        # corpus ratio.  Complements L_sharp which rewards confident
        # assignments without caring which prototype receives them.
        _min_usage: float = 1.0 / (self.config.num_languages * 4)   # 6.25% for K=4
        mean_usage = p0_real.mean(dim=0)                             # (K,)
        loss_balance = F.relu(_min_usage - mean_usage).mean()

        # ── Combined loss ─────────────────────────────────────────────────
        cfg = self.config
        loss = (
            loss_gen
            + cfg.rtd_weight * loss_rtd
            + cfg.lambda_smooth * smooth_weight * loss_smooth
            + cfg.lambda_div * loss_div
            + cfg.lambda_sharp * loss_sharp
            + cfg.lambda_balance * loss_balance
        )

        return {
            "loss":              loss,
            "loss_gen":          loss_gen.item(),
            "loss_rtd":          loss_rtd.item(),
            "loss_smooth":       loss_smooth.item(),
            "smooth_weight":     smooth_weight,
            "loss_div":          loss_div.item(),
            "loss_sharp":        loss_sharp.item(),
            "loss_balance":      loss_balance.item(),
            "rtd_acc":           rtd_acc,
            "n_masked":          n_masked,
            "language_probs":    p_ctx,    # (B, T, K) — context-refined, for monitoring
            "switch_magnitudes": s,        # (B, T) — from corrupted input, for analysis
        }

    def _init_weights(self, module: nn.Module) -> None:
        """
        BERT-style initialisation: N(0, 0.02) for Linear and Embedding.

        LanguagePrototypes and SBERTaAttention are explicitly skipped because
        they set their own carefully designed initialisations in __init__:
          · LanguagePrototypes.prototypes — orthogonal init scaled by 0.5
          · SBERTaAttention.compat        — identity init (starts from standard attention)
          · SBERTaAttention.gamma         — zero init
        Overwriting these with N(0, 0.02) would destroy the intended geometry
        before training even begins.
        """
        if isinstance(module, LanguagePrototypes):
            # Orthogonal init + scale handled in LanguagePrototypes.__init__
            return
        if isinstance(module, SBERTaAttention):
            # Apply standard init to projection weights only;
            # compat (identity) and gamma (zero) keep their custom inits.
            nn.init.normal_(module.W_Q.weight, mean=0.0, std=0.02)
            nn.init.normal_(module.W_K.weight, mean=0.0, std=0.02)
            nn.init.normal_(module.W_V.weight, mean=0.0, std=0.02)
            nn.init.normal_(module.W_O.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.W_O.bias)
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_encoder(self) -> SBERTaModel:
        """Return the bare encoder for downstream fine-tuning."""
        return self.sberta