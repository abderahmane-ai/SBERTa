"""SBERTa configuration dataclass."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SBERTaConfig:
    # ── Vocabulary & sequence ──────────────────────────────────────────────
    vocab_size: int = 50_265
    max_position_embeddings: int = 512
    pad_token_id: int = 0
    mask_token_id: int = 2

    # ── Architecture ──────────────────────────────────────────────────────
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-12

    # ── Two-phase encoder ─────────────────────────────────────────────────
    # Phase 1 (layers 0 … n_base_layers−1): standard attention, no language
    # bias injected. MLM/RTD distributional pressure forces language-clustered
    # representations to emerge naturally, as in multilingual BERT.
    #
    # Language prototypes are assigned from Phase 1 output — contextual and
    # meaningful — before any language signal enters the attention stack.
    #
    # Phase 2 (layers n_base_layers … num_hidden_layers−1): language-aware
    # attention with per-head K×K compatibility matrices and switch bias.
    # p is a real signal by the time it reaches these layers.
    #
    # Constraint: 0 < n_base_layers < num_hidden_layers.
    n_base_layers: int = 6

    # ── Code-switching ────────────────────────────────────────────────────
    num_languages: int = 2
    proto_temperature: float = 0.5   # stored as log_τ; learnable=False avoids
                                     # tau decay that sharpens Sinkhorn past
                                     # convergence
    learnable_temperature: bool = False

    # ── Prototype prior — adaptive EMA ────────────────────────────────────
    # Initialised to uniform (1/K); updated each forward pass from the
    # Sinkhorn batch marginals. Converges to the true corpus language
    # distribution within ~500 optimiser steps without any hardcoded signal.
    # Works on any K-language mixture at any corpus proportion.
    prior_ema_momentum: float = 0.95  # faster adaptation for small corpus (<50M tokens)
                                      # use 0.99 for large corpus (>100M tokens)

    # ── Regularisation ────────────────────────────────────────────────────
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # ── Pre-training (ELECTRA-style RTD) ──────────────────────────────────
    mlm_probability: float = 0.15
    rtd_weight: float = 15.0
    generator_size_divisor: int = 2

    # Geometric span masking — mean span length = 1/span_mask_geo_p tokens
    span_mask_geo_p: float = 0.2
    span_mask_min_len: int = 1
    span_mask_max_len: int = 10

    # Sinkhorn-Knopp clustering (SwAV-style collapse prevention)
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iters: int = 20

    # ── Loss weights ──────────────────────────────────────────────────────
    lambda_cluster: float = 3.0   # Sinkhorn prototype equipartition
    lambda_ortho:   float = 1.0   # prototype geometry: (L_n L_nᵀ − I)² mean

    # ─────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        assert 0 < self.n_base_layers < self.num_hidden_layers, (
            f"n_base_layers ({self.n_base_layers}) must satisfy "
            f"0 < n_base_layers < num_hidden_layers ({self.num_hidden_layers})"
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SBERTaConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    # ── Convenience factories ────────────────────────────────────────────
    @classmethod
    def small(cls) -> "SBERTaConfig":
        """Quick-experiment config (~16 M params)."""
        return cls(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            n_base_layers=2,
        )

    @classmethod
    def base(cls) -> "SBERTaConfig":
        """BERT-base equivalent size (~124 M params)."""
        return cls()            # all defaults; n_base_layers=6 of 12

    @classmethod
    def medium(cls) -> "SBERTaConfig":
        """Mid-range config suitable for low-resource experiments (~51 M params)."""
        return cls(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            n_base_layers=4,
        )

    @classmethod
    def large(cls) -> "SBERTaConfig":
        """BERT-large equivalent size (requires massive data, ~355 M params)."""
        return cls(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            n_base_layers=12,
        )