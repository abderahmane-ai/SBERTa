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

    # ── Code-switching ────────────────────────────────────────────────────
    num_languages: int = 3
    proto_temperature: float = 0.5
    learnable_temperature: bool = False

    # ── Regularisation ────────────────────────────────────────────────────
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # ── Pre-training (ELECTRA-style RTD) ──────────────────────────────────
    mlm_probability: float = 0.15
    rtd_weight: float = 30.0
    generator_size_divisor: int = 3
    lambda_smooth: float = 5.0
    smooth_warmup_ratio: float = 0.15
    smooth_weight_min: float = 0.05
    burnin_ratio: float = 0.05
    lambda_div: float = 5.0
    lambda_balance: float = 5.0
    balance_min_usage_factor: float = 0.5

    # ────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
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
        )

    @classmethod
    def base(cls) -> "SBERTaConfig":
        """BERT-base equivalent size (~124 M params)."""
        return cls()            # all defaults

    @classmethod
    def medium(cls) -> "SBERTaConfig":
        """Mid-range config suitable for low-resource experiments (~51 M params)."""
        return cls(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
        )

    @classmethod
    def large(cls) -> "SBERTaConfig":
        """BERT-large equivalent size (requires massive data, ~355 M params)."""
        return cls(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )