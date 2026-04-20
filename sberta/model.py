"""
SBERTa model implementation.

Architecture summary
--------------------
Design principle: earn the language signal before using it.

Two-phase encoder:

  Phase 1 — Context (layers 0 … n_base_layers−1):
    Standard attention, no language bias. MLM/RTD distributional pressure
    forces language-clustered representations to emerge naturally, as in
    multilingual BERT. No language signal is injected at this stage.

  Language Assignment Pivot:
    p_t = softmax(H_base_t · Lᵀ / τ)
    s_t = 1 − p_t ᵀ p_{t−1}
    Applied to Phase 1 output — contextual and meaningful. Sinkhorn
    clustering and prototype orthogonality are enforced here.

  Phase 2 — Language-Aware (layers n_base_layers … num_hidden_layers−1):
    S_h(i,j) = (Qᵢ·Kⱼ)/√dₕ + pᵢᵀ Cₕ pⱼ + γ·sⱼ
    K×K compatibility matrices receive a real signal and learn asymmetric
    cross-language attention affinities. Temporal coherence across language
    spans emerges implicitly — no smoothing penalty required.

Pre-training objectives (ELECTRA-style RTD):
  Generator:     SBERTaGenerator (hidden_size // generator_size_divisor) — MLM on
                 geometrically-masked spans; proposes plausible token replacements.
  Discriminator: full SBERTa — RTD binary classification at every token position,
                 giving 6-7× more gradient signal than vanilla 15%-masked MLM.

  L = L_gen + w_rtd · L_RTD + λ_cluster · L_cluster + λ_ortho · L_ortho

  L_gen        : generator MLM (span-masked positions only; normalised by n_masked)
  L_RTD        : replaced token detection — supervises every real token position
  L_cluster    : Sinkhorn-Knopp on Phase 1 (contextual) representations with
                 adaptive EMA prior — discovers corpus language distribution
                 online; no hardcoded prior, works on any K-language mixture
  L_ortho      : (L_n L_nᵀ − I)².mean() — directly regularises prototype geometry,
                 preventing prototype vectors from drifting together regardless of
                 what the assignment losses are doing

The model is fully zero-knowledge: no Unicode priors, no script IDs, no
dictionaries. Language structure emerges purely from the MLM distributional
objective. Works on any K-language mixture.
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


def _geometric_span_mask(
    attention_mask: torch.Tensor,
    mask_prob: float,
    geo_p: float,
    min_len: int,
    max_len: int,
) -> torch.Tensor:
    """
    Language-agnostic span masking for the ELECTRA generator.

    Span lengths are sampled from Geometric(geo_p) via inverse CDF, giving a
    mean span of 1/geo_p tokens (5 at default geo_p=0.2). Spans are placed at
    random non-overlapping positions until mask_prob of real tokens are covered.

    All random number generation is performed in two bulk CPU calls before any
    loop begins, eliminating per-iteration CUDA kernel launches.

    Args:
        attention_mask: (B, T) binary — 1 for real tokens, 0 for padding
        mask_prob:      target fraction of real tokens to mask
        geo_p:          geometric distribution parameter (0 < geo_p ≤ 1)
        min_len:        minimum span length in tokens
        max_len:        maximum span length in tokens
    Returns:
        span_mask: (B, T) bool — True at positions selected for masking
    """
    B, T = attention_mask.shape
    device = attention_mask.device
    log1mp = math.log(1.0 - geo_p)

    real_lens = attention_mask.sum(dim=1).cpu()
    perms = [torch.randperm(max(int(real_lens[b].item()), 1)) for b in range(B)]
    us       = torch.rand(B, T).clamp_(min=1e-9)
    raw_lens = us.log_().div_(log1mp).ceil_().clamp_(min_len, max_len).int()
    span_mask = torch.zeros(B, T, dtype=torch.bool)

    for b in range(B):
        real_len = int(real_lens[b].item())
        if real_len == 0:
            continue
        target_n    = max(1, int(real_len * mask_prob))
        n_masked    = 0
        span_lens_b = raw_lens[b].tolist()
        for idx, start in enumerate(perms[b].tolist()):
            if n_masked >= target_n:
                break
            if start >= real_len or span_mask[b, start]:
                continue
            span_len = min(span_lens_b[idx], real_len - start, target_n - n_masked)
            if span_len <= 0:
                continue
            span_mask[b, start : start + span_len] = True
            n_masked += span_len

    return span_mask.to(device)


@torch.no_grad()
def _sinkhorn(
    scores: torch.Tensor,
    epsilon: float,
    n_iters: int,
    prototype_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp optimal transport with support for non-uniform prototype marginals.

    Produces a soft assignment matrix Q satisfying:
      · Q.sum(dim=1) = 1                          — each token is fully assigned
      · Q.sum(dim=0)[k] ≈ prototype_weights[k]    — each prototype receives its
                                                     target share of total mass

    prototype_weights is supplied as the adaptive EMA prior, which is initialised
    uniform and converges to the true corpus language distribution within ~500
    steps. Sinkhorn therefore enforces the discovered distribution rather than
    fighting the corpus, and requires no hardcoded prior.

    Args:
        scores:             (N, K) raw dot-product similarities (pre-softmax)
        epsilon:            entropy regularization — lower → harder assignments
        n_iters:            Sinkhorn iterations (20 for convergence at all scales)
        prototype_weights:  (K,) target column marginals, must sum to 1.
                            None → uniform 1/K
    Returns:
        Q: (N, K) soft assignment — rows sum to 1, col k sums to N * weights[k]
    """
    N, K = scores.shape

    if prototype_weights is None:
        prototype_weights = scores.new_ones(K) / K

    shifted = scores - scores.max(dim=-1, keepdim=True).values
    shifted = shifted.clamp(min=-10.0 / epsilon)
    Q = torch.exp(shifted / epsilon)
    Q /= Q.sum()

    for _ in range(n_iters):
        col_sums = Q.sum(dim=0, keepdim=True).clamp(min=1e-8)
        Q = Q * (prototype_weights.unsqueeze(0) / col_sums)
        row_sums = Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
        Q = Q / (row_sums * N)

    return Q * N


# ─── Language Prototypes ──────────────────────────────────────────────────────


class LanguagePrototypes(nn.Module):
    """
    K learned prototype vectors L ∈ ℝ^{K×d} with learnable temperature τ.

    Applied at the Phase 1 / Phase 2 boundary, where H_base is contextual
    and distributions p are therefore meaningful from step one.

    Responsibilities:
      · Compute language distributions p_t from contextual Phase 1 output.
      · Compute continuous switch magnitudes s_t.

    Temperature τ is stored as log_τ for unconstrained optimisation.
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        self.K: int = config.num_languages

        self.prototypes: nn.Parameter = nn.Parameter(
            torch.empty(config.num_languages, config.hidden_size)
        )
        with torch.no_grad():
            nn.init.orthogonal_(self.prototypes)
            self.prototypes.mul_(0.5)

        log_tau_init = math.log(config.proto_temperature)
        if config.learnable_temperature:
            self.log_tau: nn.Parameter = nn.Parameter(torch.tensor(log_tau_init))
        else:
            self.register_buffer("log_tau", torch.tensor(log_tau_init))

    @property
    def tau(self) -> torch.Tensor:
        """Always-positive temperature: τ = exp(log_τ), floored at 0.25."""
        return self.log_tau.exp().clamp(min=0.25)

    def get_distributions(self, h: torch.Tensor) -> torch.Tensor:
        """
        p_t = softmax(h_t Lᵀ / τ)

        Args:
            h: (B, T, d) — Phase 1 contextual representations
        Returns:
            p: (B, T, K)
        """
        L_norm = F.normalize(self.prototypes, dim=-1)
        return F.softmax(h @ L_norm.T / self.tau, dim=-1)

    def get_switch_magnitudes(self, p: torch.Tensor) -> torch.Tensor:
        """
        s_t = 1 − p_t ᵀ p_{t−1},  s₁ = 0.

        Args:
            p: (B, T, K)
        Returns:
            s: (B, T)
        """
        sim = (p[:, 1:] * p[:, :-1]).sum(dim=-1)
        zeros = torch.zeros(p.size(0), 1, device=p.device, dtype=p.dtype)
        return torch.cat([zeros, 1.0 - sim], dim=1)


# ─── Input Embeddings ─────────────────────────────────────────────────────────


class SBERTaEmbeddings(nn.Module):
    """
    h = LN(E_tok(x) + E_pos(t))

    Pure token + positional embeddings with no language augmentation.
    Language signal enters the sequence after Phase 1 has produced
    contextual representations — not at the raw embedding stage.
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
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.dropout: nn.Dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, stop_grad: bool = False) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T)
            stop_grad: If True, detach token embeddings (GDES for discriminator)
        Returns:
            h: (B, T, d)
        """
        T: int = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        tok_emb = self.token_embeddings(input_ids)
        if stop_grad:
            tok_emb = tok_emb.detach()
        return self.dropout(self.layer_norm(tok_emb + self.position_embeddings(positions)))


# ─── Attention ────────────────────────────────────────────────────────────────


class SBERTaAttention(nn.Module):
    """
    Multi-head self-attention with an optional per-head K×K language-compatibility
    bias and switch-position scalar, used in Phase 2 layers.

    Phase 1 layers instantiate this module with use_lang_bias=False, in which
    case the attention reduces to standard scaled dot-product attention and the
    compat / gamma parameters are not created.

    Phase 2 score:
      S_h(i,j) = (Qᵢ · Kⱼ) / √dₕ  +  pᵢᵀ Cₕ pⱼ  +  γ · sⱼ

    Cₕ ∈ ℝ^{K×K} per head, initialised to identity (starts from standard
    attention; learns asymmetric language affinities as training progresses).
    γ initialised to zero.
    """

    def __init__(self, config: SBERTaConfig, use_lang_bias: bool = True) -> None:
        super().__init__()
        d, H, K = config.hidden_size, config.num_attention_heads, config.num_languages
        assert d % H == 0
        dh = d // H

        self.H, self.dh, self.d = H, dh, d
        self.use_lang_bias = use_lang_bias
        self.scale: float = math.sqrt(dh)

        self.W_Q: nn.Linear = nn.Linear(d, d, bias=False)
        self.W_K: nn.Linear = nn.Linear(d, d, bias=False)
        self.W_V: nn.Linear = nn.Linear(d, d, bias=False)
        self.W_O: nn.Linear = nn.Linear(d, d)
        self.attn_drop: nn.Dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if use_lang_bias:
            # Per-head K×K language compatibility matrix Cₕ ∈ ℝ^{H×K×K}
            self.compat: nn.Parameter = nn.Parameter(
                torch.eye(K).unsqueeze(0).expand(H, -1, -1).clone()
                + 0.01 * torch.randn(H, K, K)
            )
            # Global switch-position scalar γ, init to zero
            self.gamma: nn.Parameter = nn.Parameter(torch.zeros(1))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.dh).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d)

    def forward(
        self,
        H: torch.Tensor,                               # (B, T, d)
        p: Optional[torch.Tensor] = None,              # (B, T, K) — required for Phase 2
        s: Optional[torch.Tensor] = None,              # (B, T)    — required for Phase 2
        additive_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, T)
    ) -> torch.Tensor:
        B, T, _ = H.shape

        Q  = self._split(self.W_Q(H))
        K_ = self._split(self.W_K(H))
        V  = self._split(self.W_V(H))

        scores = torch.matmul(Q, K_.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if self.use_lang_bias and p is not None and s is not None:
            p_compat = torch.einsum("bik,hkl->bhil", p, self.compat)      # (B, H, T, K)
            scores   = scores + torch.einsum("bhil,bjl->bhij", p_compat, p)
            scores   = scores + self.gamma * s.view(B, 1, 1, T)

        if additive_mask is not None:
            scores = scores + additive_mask

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        return self.W_O(self._merge(torch.matmul(attn, V)))


# ─── Encoder Layer ────────────────────────────────────────────────────────────


class SBERTaLayer(nn.Module):
    """
    One SBERTa encoder layer (post-LN).

    When use_lang_bias=False (Phase 1): standard BERT-style transformer layer.
    When use_lang_bias=True  (Phase 2): language-aware layer with K×K compat bias.

    The same module class is used for both phases; the distinction is made at
    construction time and stored in the attention sub-module.
    """

    def __init__(self, config: SBERTaConfig, use_lang_bias: bool = True) -> None:
        super().__init__()
        d: int = config.hidden_size
        self.attention: SBERTaAttention = SBERTaAttention(config, use_lang_bias)
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(d, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, d),
        )
        self.norm_attn: nn.LayerNorm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.norm_ffn: nn.LayerNorm  = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.dropout: nn.Dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        H: torch.Tensor,
        p: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            additive: Optional[torch.Tensor] = (
                (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -10_000.0
            )
        else:
            additive = None

        H_attn = self.attention(self.norm_attn(H), p, s, additive)
        H_mid  = H + self.dropout(H_attn)
        return H_mid + self.dropout(self.ffn(self.norm_ffn(H_mid)))


# ─── Generator ────────────────────────────────────────────────────────────────


class _GeneratorLayer(nn.Module):
    """Standard Pre-LN transformer layer used internally by SBERTaGenerator."""

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
        nx = self.norm1(x)
        attn_out, _ = self.attn(
            nx, nx, nx, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.drop(attn_out)
        return x + self.drop(self.ffn(self.norm2(x)))


class SBERTaGenerator(nn.Module):
    """
    Lightweight MLM generator for ELECTRA-style RTD pre-training.

    Operates at hidden_size // generator_size_divisor (d_gen). Receives the
    discriminator's token embedding table as a weight tensor at forward time
    (avoids double-registration in the module tree while preserving the tie).
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        d: int = config.hidden_size
        d_gen: int = d // config.generator_size_divisor
        n_heads: int = max(1, config.num_attention_heads // config.generator_size_divisor)
        n_layers: int = max(1, config.num_hidden_layers // config.generator_size_divisor)

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

        self.final_norm: nn.LayerNorm = nn.LayerNorm(d_gen, eps=config.layer_norm_eps)
        self.mlm_dense: nn.Linear  = nn.Linear(d_gen, d_gen)
        self.mlm_norm: nn.LayerNorm = nn.LayerNorm(d_gen, eps=config.layer_norm_eps)
        self.output_proj: nn.Linear = nn.Linear(d_gen, d, bias=False)
        self.mlm_bias: nn.Parameter = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        token_emb_weight: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:        (B, T) — span-masked token ids
            token_emb_weight: (V, d) — discriminator's token embedding weight (tied)
            attention_mask:   (B, T) binary
        Returns:
            logits: (B, T, V)
        """
        T = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        h = self.norm_in(
            self.input_proj(F.embedding(input_ids, token_emb_weight))
            + self.pos_emb(positions)
        )
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        for layer in self.layers:
            h = layer(h, pad_mask)
        h = self.final_norm(h)
        h_mlm = self.mlm_norm(F.gelu(self.mlm_dense(h)))
        return self.output_proj(h_mlm) @ token_emb_weight.T + self.mlm_bias


# ─── Bare SBERTa Encoder ──────────────────────────────────────────────────────


class SBERTaModel(nn.Module):
    """
    Bare SBERTa encoder — zero-knowledge, script-agnostic, two-phase.

    forward_phase1: runs Phase 1 layers only and returns H_base, p, s.
                    Used by the pre-training wrapper to compute clustering
                    losses on clean (unmasked) contextual representations
                    without running the full Phase 2 forward pass.

    forward: full two-phase pass, returning H_final, p, s, H_base.
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        self.config: SBERTaConfig = config
        self.prototypes: LanguagePrototypes = LanguagePrototypes(config)
        self.embeddings: SBERTaEmbeddings = SBERTaEmbeddings(config)
        self.layers: nn.ModuleList = nn.ModuleList([
            SBERTaLayer(config, use_lang_bias=(i >= config.n_base_layers))
            for i in range(config.num_hidden_layers)
        ])
        self.final_norm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward_phase1(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stop_embedding_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase 1 forward only — contextual representations before language assignment.

        Args:
            input_ids:           (B, T)
            attention_mask:      (B, T) binary
            stop_embedding_grad: If True, detach token embeddings (GDES)
        Returns:
            H_base: (B, T, d) — Phase 1 contextual hidden states
            p:      (B, T, K) — language distributions from H_base
            s:      (B, T)    — switch magnitudes
        """
        H = self.embeddings(input_ids, stop_grad=stop_embedding_grad)
        for layer in self.layers[: self.config.n_base_layers]:
            H = layer(H, attention_mask=attention_mask)
        H_base = H
        p = self.prototypes.get_distributions(self.final_norm(H_base))
        s = self.prototypes.get_switch_magnitudes(p)
        return H_base, p, s

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stop_embedding_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full two-phase forward.

        Args:
            input_ids:           (B, T)
            attention_mask:      (B, T) binary
            stop_embedding_grad: If True, detach token embeddings (GDES)
        Returns:
            H:      (B, T, d) — Phase 2 final hidden states
            p:      (B, T, K) — language distributions (from Phase 1 output)
            s:      (B, T)    — switch magnitudes
            H_base: (B, T, d) — Phase 1 hidden states (pivot point)
        """
        H_base, p, s = self.forward_phase1(input_ids, attention_mask, stop_embedding_grad)
        H = H_base
        for layer in self.layers[self.config.n_base_layers :]:
            H = layer(H, p, s, attention_mask)
        H = self.final_norm(H)
        return H, p, s, H_base


# ─── Pre-training Wrapper ─────────────────────────────────────────────────────


class SBERTaForPreTraining(nn.Module):
    """
    SBERTa with ELECTRA-style RTD pre-training — universal, zero-knowledge.

    Generator (1/generator_size_divisor size):
      Receives geometrically-masked input; proposes plausible token replacements
      via MLM. Shares the token embedding weight with the discriminator.

    Discriminator (full SBERTa):
      Receives corrupted sequence; classifies every real token as real or
      replaced. Supervised at all T positions — not just the ~15% masked —
      giving 6-7× more gradient signal than vanilla MLM.

    Loss components:
      L_gen     : generator MLM on geometrically-masked spans
      L_RTD     : discriminator real/replaced BCE at every real token (GDES)
      L_cluster : Sinkhorn-Knopp on Phase 1 contextual representations;
                  prior is adaptive EMA — discovers corpus language distribution
                  within ~500 steps, no hardcoded marginals required
      L_ortho   : (L_n L_nᵀ − I)².mean() — regularises prototype geometry
                  directly, preventing prototype vectors from drifting together
                  independently of the assignment losses

    Training efficiency:
      The pre-training wrapper calls forward_phase1 on the clean (unmasked)
      input to compute clustering losses, then runs the full discriminator
      forward on the corrupted sequence. Phase 2 is never executed redundantly
      on the clean sequence.
    """

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        self.config: SBERTaConfig = config
        self.sberta: SBERTaModel = SBERTaModel(config)
        self.generator: SBERTaGenerator = SBERTaGenerator(config)
        self.rtd_head: nn.Linear = nn.Linear(config.hidden_size, 1)

        # Adaptive EMA prior — initialised uniform; updated each forward pass
        # from Sinkhorn batch marginals; converges to true corpus distribution.
        # Registered as a buffer: lives on the correct device, included in
        # state_dict, not an optimised parameter.
        self.register_buffer(
            "_prototype_prior",
            torch.ones(config.num_languages) / config.num_languages,
        )

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
    ) -> dict:
        """
        Args:
            input_ids:      (B, T) — original (unmasked) token ids.
            attention_mask: (B, T) — 1 for real tokens, 0 for padding.
        Returns:
            dict with scalar 'loss' and per-component loss values.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        cfg  = self.config
        real = attention_mask.bool()

        # ── Step 1: Phase 1 forward on clean input ────────────────────────
        # H_base is contextual — meaningful for prototype assignment.
        # p and s are derived from real (unmasked) language structure.
        H_base, p, s = self.sberta.forward_phase1(input_ids, attention_mask)

        # ── L_cluster: Sinkhorn on Phase 1 contextual representations ─────
        # Scores mirror get_distributions exactly (same dot-product / tau).
        real_h       = self.sberta.final_norm(H_base)[real]                  # (N_real, d)
        L_norm       = F.normalize(self.sberta.prototypes.prototypes, dim=-1)
        proto_scores = (
            real_h @ L_norm.T
            / self.sberta.prototypes.tau
        )                                                                    # (N_real, K)
        Q = _sinkhorn(
            proto_scores.detach(),
            cfg.sinkhorn_epsilon,
            cfg.sinkhorn_iters,
            prototype_weights=self._prototype_prior,
        )
        loss_cluster = F.cross_entropy(proto_scores, Q)

        # ── Adaptive EMA prior update ─────────────────────────────────────
        # Tracks the true corpus language distribution from Sinkhorn marginals.
        # No gradient — buffer update only.
        with torch.no_grad():
            batch_marginal = Q.sum(dim=0)
            batch_marginal = batch_marginal / batch_marginal.sum().clamp(min=1e-8)
            self._prototype_prior.mul_(cfg.prior_ema_momentum).add_(
                batch_marginal * (1.0 - cfg.prior_ema_momentum)
            )

        # ── L_ortho: prototype geometry ───────────────────────────────────
        # Penalises off-diagonal entries of the Gram matrix of normalised
        # prototypes. Prevents prototype vectors from drifting together
        # regardless of assignment dynamics.
        L_n       = F.normalize(self.sberta.prototypes.prototypes, dim=-1)  # (K, d)
        gram      = L_n @ L_n.T                                              # (K, K)
        eye       = torch.eye(cfg.num_languages, device=gram.device)
        loss_ortho = (gram - eye).pow(2).mean()

        # ── Step 2: geometric span masking ────────────────────────────────
        span_mask = _geometric_span_mask(
            attention_mask, cfg.mlm_probability,
            cfg.span_mask_geo_p, cfg.span_mask_min_len, cfg.span_mask_max_len,
        )
        masked_ids = input_ids.clone()
        masked_ids[span_mask] = cfg.mask_token_id

        # ── Step 3: generator — MLM logits + token sampling ───────────────
        gen_logits = self.generator(masked_ids, self._tok_w, attention_mask)  # (B, T, V)

        with torch.no_grad():
            gen_tokens = torch.multinomial(
                F.softmax(gen_logits[span_mask].detach(), dim=-1),
                num_samples=1,
            ).squeeze(-1)

        corrupted_ids = input_ids.clone()
        corrupted_ids[span_mask] = gen_tokens
        is_replaced   = (corrupted_ids != input_ids).float()

        # ── Step 4: discriminator on corrupted sequence ───────────────────
        # GDES: RTD gradients do not flow into the shared embedding table.
        # Only the generator's MLM loss shapes the shared embeddings.
        # Full two-phase forward — Phase 2 runs on corrupted input only.
        H, _, _, _ = self.sberta(corrupted_ids, attention_mask, stop_embedding_grad=True)

        # ── L_gen ─────────────────────────────────────────────────────────
        gen_labels = input_ids.new_full(input_ids.shape, -100)
        gen_labels[span_mask] = input_ids[span_mask]
        n_masked = max(int(span_mask.sum().item()), 1)
        loss_gen = F.cross_entropy(
            gen_logits.view(-1, cfg.vocab_size),
            gen_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        ) / n_masked

        # ── L_RTD ─────────────────────────────────────────────────────────
        rtd_logits = self.rtd_head(H).squeeze(-1)
        loss_rtd   = F.binary_cross_entropy_with_logits(
            rtd_logits[real], is_replaced[real], reduction="mean"
        )

        with torch.no_grad():
            rtd_preds = (rtd_logits[real] > 0.0).float()
            rtd_acc   = (rtd_preds == is_replaced[real]).float().mean().item()

        # ── Combined loss ─────────────────────────────────────────────────
        loss = (
            loss_gen
            + cfg.rtd_weight    * loss_rtd
            + cfg.lambda_cluster * loss_cluster
            + cfg.lambda_ortho  * loss_ortho
        )

        return {
            "loss":              loss,
            "loss_gen":          loss_gen.item(),
            "loss_rtd":          loss_rtd.item(),
            "loss_cluster":      loss_cluster.item(),
            "loss_ortho":        loss_ortho.item(),
            "rtd_acc":           rtd_acc,
            "n_masked":          n_masked,
            "language_probs":    p,    # (B, T, K) — from unmasked Phase 1 output
            "switch_magnitudes": s,    # (B, T)    — from unmasked Phase 1 output
        }

    def _init_weights(self, module: nn.Module) -> None:
        """
        BERT-style initialisation: N(0, 0.02) for Linear and Embedding.

        LanguagePrototypes and SBERTaAttention (Phase 2) are skipped because
        they set their own initialisations in __init__:
          · LanguagePrototypes.prototypes — orthogonal init scaled by 0.5
          · SBERTaAttention.compat        — identity init
          · SBERTaAttention.gamma         — zero init
        """
        if isinstance(module, LanguagePrototypes):
            return
        if isinstance(module, SBERTaAttention):
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