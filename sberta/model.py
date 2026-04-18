"""
SBERTa model implementation.

Architecture summary
--------------------
Pre-training flow (per forward pass):

  1.  h_base   = E_tok(x) + E_pos(t)
  2.  p⁽⁰⁾_t  = softmax(h_base · Lᵀ / τ)              [pre-contextual language dist]
  3.  s_t      = 1 − p⁽⁰⁾_t ᵀ p⁽⁰⁾_{t−1}, s₁ = 0    [continuous switch magnitude]
  4.  h⁽⁰⁾    = LN(h_base + Σₖ p⁽⁰⁾_{t,k} eₖ + sₜ · e_sw)
  5.  Each encoder layer ℓ:
        S_h(i,j) = (Qᵢ·Kⱼ)/√dₕ + p⁽⁰⁾ᵢᵀ Cₕ p⁽⁰⁾ⱼ + γ·sⱼ   [Cₕ ∈ ℝ^{K×K} per head]
        H⁽ˡ⁾   = LN(H + MHA(H, p⁽⁰⁾, s)) + FFN(·))

Pre-training objectives (ELECTRA-style RTD):
  Generator:     SBERTaGenerator (hidden_size // generator_size_divisor) — MLM on
                 switch-span-masked input; proposes plausible token replacements.
  Discriminator: full SBERTa — RTD binary classification at every token position,
                 giving 6-7× more gradient signal than vanilla 15%-masked MLM.

  L = L_gen + w_rtd · L_RTD + λ_smooth · L_smooth + λ_div · L_div

  L_gen        : generator MLM (span-masked positions only; normalised by n_masked)
  L_RTD        : replaced token detection — supervises every real token position
  L_smooth     : unsupervised temporal stickiness — mean switch magnitude penalised
                 at within-script boundaries (cross-script transitions excluded via
                 Unicode script prior); no curriculum needed
  L_div        : prototype diversity — disabled (lambda_div=0.0) with script prior;
                 kept for future K>2 experiments

The Unicode script prior (script_prior_weight=0.5) blends learned prototype
distributions with hard Unicode script signals, providing strong separation from
step 1 and eliminating the need for burn-in curricula or balance losses.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SBERTaConfig
import unicodedata as _unicodedata

def _build_script_ids(
    vocab_size: int,
    pieces: list[str],
) -> torch.Tensor:
    from collections import Counter

    def _first_script(piece: str) -> Optional[str]:
        for ch in piece.lstrip("\u2581"):
            try:
                name = _unicodedata.name(ch)
            except ValueError:
                continue
            script = name.split()[0]
            if script in ("DIGIT", "SPACE", "NO-BREAK", "ZERO", "BYTE"):
                continue
            return script
        return None

    script_counts: Counter = Counter()
    token_scripts: list[Optional[str]] = []
    for piece in pieces:
        s = _first_script(piece)
        token_scripts.append(s)
        if s is not None:
            script_counts[s] += 1

    script_to_id: dict[str, int] = {
        s: i for i, (s, _) in enumerate(script_counts.most_common())
    }

    script_ids = torch.full((vocab_size,), -1, dtype=torch.long)
    for token_id, s in enumerate(token_scripts):
        if s is not None and s in script_to_id:
            script_ids[token_id] = script_to_id[s]

    return script_ids



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

        target_n  = max(1, int(real_len * target_mask_prob))
        # Hard cap at 30% to prevent a single dominant-language span from
        # covering the entire sequence when prototypes collapse.
        max_mask_n = int(real_len * min(target_mask_prob * 2.0, 0.30))
        starts_list = starts.tolist()
        ends_list = ends.tolist()

        n_masked = 0
        for idx in torch.randperm(len(starts_list), device=p.device).tolist():
            if n_masked >= target_n or n_masked >= max_mask_n:
                break
            s, e = starts_list[idx], ends_list[idx]
            # Skip spans that would overshoot the hard cap by more than one span
            if n_masked + (e - s) > max_mask_n:
                continue
            span_mask[b, s:e] = True
            n_masked += e - s

    return span_mask                                 # (B, T)


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
        """Always-positive temperature: τ = exp(log_τ), floored at 0.25."""
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
        Exponential repulsion loss that never reaches zero.
        Always positive, always has gradient, exponentially steeper near collapse.
        """
        L_n = F.normalize(self.prototypes, dim=-1)             # (K, d)
        cos = L_n @ L_n.T                                      # (K, K)
        mask = torch.triu(
            torch.ones(self.K, self.K, device=cos.device), diagonal=1
        )
        repulsion = torch.exp(cos) - math.exp(-1.0)
        return (repulsion * mask).sum() / (self.K * (self.K - 1) / 2.0)


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

    def get_base(self, input_ids: torch.Tensor, stop_grad: bool = False) -> torch.Tensor:
        """
        Stage 1: tok + pos embeddings without normalisation or augmentation.
        Used to compute p⁽⁰⁾ and s before augmentation.

        Args:
            input_ids: (B, T)
            stop_grad: If True, detach token embeddings (GDES for discriminator)
        Returns:
            h_base: (B, T, d)
        """
        T: int = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        tok_emb = self.token_embeddings(input_ids)
        if stop_grad:
            tok_emb = tok_emb.detach()
        return tok_emb + self.position_embeddings(positions)

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
                   French↔Arabic medium, etc. Adds only K²×H parameters (48
                   for base config with K=2, H=12) at negligible cost.
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
        # Identity init + small noise breaks symmetry so heads specialize from step 1
        self.compat: nn.Parameter = nn.Parameter(
            torch.eye(K).unsqueeze(0).expand(H, -1, -1).clone() + 0.01 * torch.randn(H, K, K)
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
    Bare SBERTa encoder with single-stage language distribution computation.

    p⁽⁰⁾_t is computed from raw h_base (tok + pos) and used for:
      · Embedding augmentation (avoids circular dependency)
      · Attention compatibility biases C_h and γ (encoder refines through context)

    The encoder's 12 layers of full self-attention resolve ambiguity through
    contextual processing, making a separate windowed refinement step redundant.
    """

    def __init__(
        self,
        config: SBERTaConfig,
        script_ids: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.config: SBERTaConfig = config
        self.prototypes: LanguagePrototypes = LanguagePrototypes(config)
        self.embeddings: SBERTaEmbeddings = SBERTaEmbeddings(config)
        self.layers: nn.ModuleList = nn.ModuleList(
            [SBERTaLayer(config) for _ in range(config.num_hidden_layers)]
        )
        if script_ids is not None:
            self.register_buffer("script_ids", script_ids)

    def _compute_script_prior(
        self,
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not hasattr(self, "script_ids"):
            return None

        K = self.config.num_languages
        B, T = input_ids.shape

        token_script = self.script_ids[input_ids]

        prior = torch.full((B, T, K), 1.0 / K, device=input_ids.device)
        for k in range(K):
            assigned = (token_script == k)
            if assigned.any():
                prior[assigned] = 0.0
                prior[assigned, k] = 1.0

        return prior

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stop_embedding_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:           (B, T)
            attention_mask:      (B, T) binary — 1 for real tokens, 0 for padding
            stop_embedding_grad: If True, detach token embeddings (GDES)
        Returns:
            H:  (B, T, d) — final encoder hidden states
            p0: (B, T, K) — pre-contextual language distributions
            s:  (B, T)    — switch magnitudes
        """
        # Stage 1: tok + pos (no LN, no language augmentation yet)
        h_base = self.embeddings.get_base(input_ids, stop_grad=stop_embedding_grad)  # (B, T, d)

        # Stage 2: pre-contextual language assignments + switch magnitudes
        p_learned = self.prototypes.get_distributions(h_base)           # (B, T, K)
        p_script  = self._compute_script_prior(input_ids)

        if p_script is not None:
            w = self.config.script_prior_weight
            p0 = (1.0 - w) * p_learned + w * p_script
        else:
            p0 = p_learned

        s = self.prototypes.get_switch_magnitudes(p0)            # (B, T)

        # Stage 3: augmented h⁽⁰⁾ uses p⁽⁰⁾ — circular-dependency free
        H = self.embeddings.augment(h_base, p0, s)               # (B, T, d)

        # Stage 4: encoder stack — attention biases use p⁽⁰⁾
        # The encoder refines language understanding through contextual processing
        for layer in self.layers:
            H = layer(H, p0, s, attention_mask)

        return H, p0, s


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
    """

    def __init__(
        self,
        config: SBERTaConfig,
        script_ids: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.config: SBERTaConfig = config
        self.sberta: SBERTaModel = SBERTaModel(config, script_ids=script_ids)
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
        segment_ids: Optional[torch.Tensor] = None,      # (B, T) document boundaries
    ) -> dict:
        """
        Args:
            input_ids:      (B, T) — original (unmasked) token ids.
            attention_mask: (B, T) — 1 for real tokens, 0 for padding.
            segment_ids:    (B, T) — document segment IDs for packed sequences.
                            Used to mask cross-document boundaries in L_smooth.
                            If None, assumes single document per sequence.
        Returns:
            dict with 'loss' (scalar) and per-component .item() loss values.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        cfg = self.config
        real = attention_mask.bool()                               # (B, T)

        # ── Step 1: span mask from original input's language structure ────
        # p_pre is computed on unmasked input so span boundaries reflect true
        # language identity rather than the [MASK] token distribution.
        # Blend with script prior to avoid masking random noise in early training.
        h_base = self.sberta.embeddings.get_base(input_ids)
        p_learned = self.sberta.prototypes.get_distributions(h_base)   # (B, T, K)
        p_script  = self.sberta._compute_script_prior(input_ids)
        
        if p_script is not None:
            w = cfg.script_prior_weight
            p_pre = (1.0 - w) * p_learned + w * p_script
        else:
            p_pre = p_learned
        
        s_pre = self.sberta.prototypes.get_switch_magnitudes(p_pre)  # (B, T)

        span_mask = _switch_span_mask(p_pre, attention_mask, cfg.mlm_probability)

        masked_ids = input_ids.clone()
        masked_ids[span_mask] = cfg.mask_token_id

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
        # GDES: stop RTD gradients from flowing into the shared embedding table.
        # The discriminator's task (detect replaced tokens) should not shape the
        # embeddings that feed prototype assignment and language distributions.
        # Only the generator's MLM loss updates the shared embeddings.
        H, p0, s = self.sberta(corrupted_ids, attention_mask, stop_embedding_grad=True)

        # ── L_gen (generator MLM) ────────────────────────────────────────
        gen_labels = input_ids.new_full(input_ids.shape, -100)
        gen_labels[span_mask] = input_ids[span_mask]
        n_masked = max(int(span_mask.sum().item()), 1)
        loss_gen = F.cross_entropy(
            gen_logits.view(-1, cfg.vocab_size),
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
        # Mask cross-document boundaries when segment_ids is provided
        if segment_ids is not None:
            cross_doc = segment_ids[:, 1:] != segment_ids[:, :-1]    # (B, T-1)
            switch_mask = real[:, 1:] & real[:, :-1] & ~cross_doc    # (B, T-1)
        else:
            switch_mask = real[:, 1:] & real[:, :-1]                 # (B, T-1)
        
        # Exclude cross-script boundaries (Arabic↔Latin) from smoothing penalty
        if hasattr(self.sberta, "script_ids"):
            s_left  = self.sberta.script_ids[input_ids[:, :-1]]
            s_right = self.sberta.script_ids[input_ids[:, 1: ]]
            both_known    = (s_left >= 0) & (s_right >= 0)
            cross_script  = both_known & (s_left != s_right)
            switch_mask   = switch_mask & ~cross_script
        
        loss_smooth = s_pre[:, 1:][switch_mask].mean() if switch_mask.any() else input_ids.new_zeros(())

        # ── L_div (prototype diversity) ───────────────────────────────────
        # Kept for future K>2 experiments; disabled by lambda_div=0.0 with script prior
        loss_div = self.sberta.prototypes.diversity_loss() if cfg.lambda_div > 0 else input_ids.new_zeros(())

        # ── Combined loss ─────────────────────────────────────────────────
        loss = (
            loss_gen
            + cfg.rtd_weight * loss_rtd
            + cfg.lambda_smooth * loss_smooth
            + cfg.lambda_div * loss_div
        )

        return {
            "loss":              loss,
            "loss_gen":          loss_gen.item(),
            "loss_rtd":          loss_rtd.item(),
            "loss_smooth":       loss_smooth.item(),
            "loss_div":          loss_div.item(),
            "rtd_acc":           rtd_acc,
            "n_masked":          n_masked,
            "language_probs":    p_pre,    # (B, T, K) — from unmasked input, consistent with losses
            "switch_magnitudes": s_pre,    # (B, T) — from unmasked input, consistent with losses
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