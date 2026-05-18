"""Darija-first SBERTa model with staged code-switching structure."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SBERTaConfig


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _geometric_span_mask(
    attention_mask: torch.Tensor,
    mask_prob: float,
    geo_p: float,
    min_len: int,
    max_len: int,
    input_ids: Optional[torch.Tensor] = None,
    special_token_ids: Optional[set[int]] = None,
) -> torch.Tensor:
    """Mask contiguous eligible spans for the ELECTRA generator."""
    B, T = attention_mask.shape
    device = attention_mask.device
    log1mp = math.log(1.0 - geo_p)

    eligible = attention_mask.bool().cpu()
    if input_ids is not None and special_token_ids:
        ids_cpu = input_ids.cpu()
        for token_id in special_token_ids:
            eligible &= ids_cpu.ne(token_id)

    perms = [torch.randperm(max(int(eligible[b].sum().item()), 1)) for b in range(B)]
    us = torch.rand(B, T).clamp_(min=1e-9)
    raw_lens = us.log_().div_(log1mp).ceil_().clamp_(min_len, max_len).int()
    span_mask = torch.zeros(B, T, dtype=torch.bool)

    for b in range(B):
        positions = eligible[b].nonzero(as_tuple=False).flatten()
        n_eligible = int(positions.numel())
        if n_eligible == 0:
            continue
        target_n = max(1, int(n_eligible * mask_prob))
        n_masked = 0
        span_lens_b = raw_lens[b].tolist()
        for idx, pos_idx in enumerate(perms[b].tolist()):
            if n_masked >= target_n:
                break
            start = int(positions[pos_idx].item())
            if span_mask[b, start]:
                continue
            span_len = min(span_lens_b[idx], target_n - n_masked)
            for j in range(start, min(T, start + span_len)):
                if not eligible[b, j] or span_mask[b, j]:
                    break
                span_mask[b, j] = True
                n_masked += 1
                if n_masked >= target_n:
                    break

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
    """Prototype vectors used after Phase 1 has built context."""

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        self.K: int = config.num_languages

        self.prototypes: nn.Parameter = nn.Parameter(
            torch.empty(config.num_languages, config.hidden_size)
        )
        with torch.no_grad():
            nn.init.orthogonal_(self.prototypes)
            self.prototypes.mul_(0.5)

        tau_init = config.proto_temperature
        tau_min = 0.25
        rho_init = math.log(math.exp(tau_init - tau_min) - 1.0)
        
        if config.learnable_temperature:
            self.rho: nn.Parameter = nn.Parameter(torch.tensor(rho_init))
        else:
            self.register_buffer("rho", torch.tensor(rho_init))

    @property
    def tau(self) -> torch.Tensor:
        """Always-positive temperature: τ = 0.25 + softplus(ρ)."""
        return 0.25 + F.softplus(self.rho)

    def get_distributions(self, h: torch.Tensor) -> torch.Tensor:
        """
        p_t = softmax(h_t L_normᵀ / τ)
        (where L_norm is L L2-normalized along the hidden dimension, making
        p depend on cosine similarity rather than raw magnitude).

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
    """Token and position embeddings."""

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
    """Self-attention with optional ramped language bias."""

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
            self.compat: nn.Parameter = nn.Parameter(torch.zeros(H, K, K))
            # Per-head structural scalars
            self.beta: nn.Parameter = nn.Parameter(torch.full((H, 1, 1), 0.01))
            self.gamma: nn.Parameter = nn.Parameter(torch.full((H, 1, 1), 0.01))

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
        additive_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, T)
        lang_bias_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            H:            (B, T, d) — input hidden states
            p:            (B, T, K) — language distributions; None for Phase 1 layers
            additive_mask:(B, 1, 1, T) — additive padding mask (-10 000 at pad positions)
        Returns:
            (B, T, d)
        """

        Q  = self._split(self.W_Q(H))
        K_ = self._split(self.W_K(H))
        V  = self._split(self.W_V(H))

        if not self.use_lang_bias:
            dropout_p = self.attn_drop.p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                Q, K_, V,
                attn_mask=additive_mask,
                dropout_p=dropout_p,
            )
            return self.W_O(self._merge(out))

        scores = torch.matmul(Q, K_.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if self.use_lang_bias and p is not None and lang_bias_scale > 0.0:
            p_compat = torch.einsum("bik,hkl->bhil", p, self.compat)      # (B, H, T, K)
            sem_bias = torch.einsum("bhil,bjl->bhij", p_compat, p)        # (B, H, T, T)
            delta_ij = 1.0 - torch.matmul(p, p.transpose(-1, -2)).unsqueeze(1) # (B, 1, T, T)
            
            scores = scores + lang_bias_scale * (
                self.beta * sem_bias + self.gamma * delta_ij
            )

        if additive_mask is not None:
            scores = scores + additive_mask

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        return self.W_O(self._merge(torch.matmul(attn, V)))


# ─── Encoder Layer ────────────────────────────────────────────────────────────


class SBERTaLayer(nn.Module):
    """One Pre-LN encoder layer."""

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
        attention_mask: Optional[torch.Tensor] = None,
        lang_bias_scale: float = 1.0,
    ) -> torch.Tensor:
        if attention_mask is not None:
            additive: Optional[torch.Tensor] = (
                (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -10_000.0
            )
        else:
            additive = None

        H_attn = self.attention(self.norm_attn(H), p, additive, lang_bias_scale)
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
    """Two-phase encoder for Algerian Darija code-switching."""

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        self.config: SBERTaConfig = config
        self.prototypes: LanguagePrototypes = LanguagePrototypes(config)
        self.embeddings: SBERTaEmbeddings = SBERTaEmbeddings(config)
        self.layers: nn.ModuleList = nn.ModuleList([
            SBERTaLayer(config, use_lang_bias=(i >= config.n_base_layers))
            for i in range(config.num_hidden_layers)
        ])
        self.phase1_norm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_norm:  nn.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward_phase1(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stop_embedding_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase 1 forward only — contextual representations before language assignment.

        Args:
            input_ids:           (B, T)
            attention_mask:      (B, T) binary
            stop_embedding_grad: If True, detach token embeddings (GDES)
        Returns:
            H_base:      (B, T, d) — Phase 1 contextual hidden states
            p:           (B, T, K) — language distributions from H_base
            s:           (B, T)    — switch magnitudes
            H_base_norm: (B, T, d) — phase1_norm(H_base); cached so
                                     SBERTaForPreTraining can reuse it for
                                     L_cluster without a second LN call.
                                     Receives L_cluster gradients only.
        """
        H = self.embeddings(input_ids, stop_grad=stop_embedding_grad)
        for layer in self.layers[: self.config.n_base_layers]:
            H = layer(H, attention_mask=attention_mask)
        H_base = H
        H_base_norm = self.phase1_norm(H_base)
        p = self.prototypes.get_distributions(H_base_norm)
        s = self.prototypes.get_switch_magnitudes(p)
        return H_base, p, s, H_base_norm

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stop_embedding_grad: bool = False,
        lang_bias_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        H_base, p, s, _ = self.forward_phase1(input_ids, attention_mask, stop_embedding_grad)
        H = H_base
        for layer in self.layers[self.config.n_base_layers :]:
            H = layer(H, p, attention_mask, lang_bias_scale)
        H = self.final_norm(H)
        return H, p, s, H_base


# ─── Pre-training Wrapper ─────────────────────────────────────────────────────


class SBERTaForPreTraining(nn.Module):
    """SBERTa with ELECTRA-style pre-training and staged structure losses."""

    def __init__(self, config: SBERTaConfig) -> None:
        super().__init__()
        self.config: SBERTaConfig = config
        self.sberta: SBERTaModel = SBERTaModel(config)
        self.generator: SBERTaGenerator = SBERTaGenerator(config)
        self.rtd_head: nn.Linear = nn.Linear(config.hidden_size, 1)

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
        cluster_scale: float = 1.0,
        ortho_scale: float = 1.0,
        lang_bias_scale: float = 1.0,
    ) -> dict:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        cfg  = self.config
        real = attention_mask.bool()
        special_ids = {
            cfg.pad_token_id,
            cfg.unk_token_id,
            cfg.mask_token_id,
            cfg.sep_token_id,
        }

        H_base, p, s, H_base_norm = self.sberta.forward_phase1(input_ids, attention_mask)

        L_norm = F.normalize(self.sberta.prototypes.prototypes, dim=-1)  # (K, d)

        real_h       = H_base_norm[real]                                    # (N_real, d)
        proto_scores = real_h @ L_norm.T / self.sberta.prototypes.tau      # (N_real, K)
        Q = _sinkhorn(
            proto_scores.detach(),
            cfg.sinkhorn_epsilon,
            cfg.sinkhorn_iters,
            prototype_weights=self._prototype_prior,
        )
        loss_cluster = F.cross_entropy(proto_scores, Q)

        with torch.no_grad():
            batch_marginal = p[real].mean(dim=0)
            batch_marginal = batch_marginal / batch_marginal.sum().clamp(min=1e-8)
            self._prototype_prior.mul_(cfg.prior_ema_momentum).add_(
                batch_marginal * (1.0 - cfg.prior_ema_momentum)
            )

        gram       = L_norm @ L_norm.T                                      # (K, K)
        eye        = torch.eye(cfg.num_languages, device=gram.device)
        loss_ortho = (gram - eye).pow(2).mean()

        span_mask = _geometric_span_mask(
            attention_mask, cfg.mlm_probability,
            cfg.span_mask_geo_p, cfg.span_mask_min_len, cfg.span_mask_max_len,
            input_ids=input_ids,
            special_token_ids=special_ids,
        )
        masked_ids = input_ids.clone()
        masked_ids[span_mask] = cfg.mask_token_id

        gen_logits = self.generator(masked_ids, self._tok_w, attention_mask)  # (B, T, V)

        with torch.no_grad():
            gen_tokens = input_ids.new_empty((0,))
            if span_mask.any():
                sample_logits = gen_logits[span_mask].detach().clone()
                sample_logits[:, list(special_ids)] = torch.finfo(sample_logits.dtype).min
                gen_tokens = torch.multinomial(
                    F.softmax(sample_logits, dim=-1),
                    num_samples=1,
                ).squeeze(-1)
            sampled_special_count = sum(
                int(gen_tokens.eq(token_id).sum().item()) for token_id in special_ids
            )
            masked_special_count = sum(
                int((span_mask & input_ids.eq(token_id)).sum().item()) for token_id in special_ids
            )

        corrupted_ids = input_ids.clone()
        if span_mask.any():
            corrupted_ids[span_mask] = gen_tokens
        is_replaced   = (corrupted_ids != input_ids).float()

        H, _, _, _ = self.sberta(
            corrupted_ids,
            attention_mask,
            stop_embedding_grad=True,
            lang_bias_scale=lang_bias_scale,
        )

        gen_labels = input_ids.new_full(input_ids.shape, -100)
        gen_labels[span_mask] = input_ids[span_mask]
        n_masked = max(int(span_mask.sum().item()), 1)
        loss_gen = F.cross_entropy(
            gen_logits.view(-1, cfg.vocab_size),
            gen_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        ) / n_masked

        rtd_logits = self.rtd_head(H).squeeze(-1)
        loss_rtd   = F.binary_cross_entropy_with_logits(
            rtd_logits[real], is_replaced[real], reduction="mean"
        )

        with torch.no_grad():
            rtd_preds = (rtd_logits[real] > 0.0).float()
            rtd_labels = is_replaced[real]
            tp = ((rtd_preds == 1) & (rtd_labels == 1)).sum().float()
            fp = ((rtd_preds == 1) & (rtd_labels == 0)).sum().float()
            fn = ((rtd_preds == 0) & (rtd_labels == 1)).sum().float()
            rtd_acc = (rtd_preds == rtd_labels).float().mean().item()
            rtd_precision = (tp / (tp + fp).clamp(min=1.0)).item()
            rtd_recall = (tp / (tp + fn).clamp(min=1.0)).item()
            rtd_f1 = (
                2.0 * rtd_precision * rtd_recall
                / max(rtd_precision + rtd_recall, 1e-8)
            )
            replaced_rate = rtd_labels.mean().item()

        loss = (
            loss_gen
            + cfg.rtd_weight    * loss_rtd
            + cluster_scale * cfg.lambda_cluster * loss_cluster
            + ortho_scale * cfg.lambda_ortho  * loss_ortho
        )

        return {
            "loss":              loss,
            "loss_gen":          loss_gen.item(),
            "loss_rtd":          loss_rtd.item(),
            "loss_cluster":      loss_cluster.item(),
            "loss_ortho":        loss_ortho.item(),
            "rtd_acc":           rtd_acc,
            "rtd_precision":     rtd_precision,
            "rtd_recall":        rtd_recall,
            "rtd_f1":            rtd_f1,
            "replaced_rate":     replaced_rate,
            "sampled_special_count": sampled_special_count,
            "masked_special_count":  masked_special_count,
            "n_masked":          n_masked,
            "cluster_scale":     cluster_scale,
            "ortho_scale":       ortho_scale,
            "lang_bias_scale":   lang_bias_scale,
            "language_probs":    p,    # (B, T, K) — from unmasked Phase 1 output
            "switch_magnitudes": s,    # (B, T)    — from unmasked Phase 1 output
        }

    def _init_weights(self, module: nn.Module) -> None:
        """
        BERT-style initialisation: N(0, 0.02) for Linear and Embedding.

        LanguagePrototypes and SBERTaAttention (Phase 2) are skipped because
        they set their own initialisations in __init__:
          · LanguagePrototypes.prototypes — orthogonal init scaled by 0.5
          · SBERTaAttention.compat        — zero init
          · SBERTaAttention.beta          — 0.01 init
          · SBERTaAttention.gamma         — 0.01 init
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
