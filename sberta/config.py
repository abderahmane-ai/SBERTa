"""Configuration for the Darija-first SBERTa recipe."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SBERTaConfig:
    vocab_size: int = 50_000
    max_position_embeddings: int = 512
    pad_token_id: int = 0
    unk_token_id: int = 1
    mask_token_id: int = 2
    sep_token_id: int = 3

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    n_base_layers: int = 6
    layer_norm_eps: float = 1e-12

    num_languages: int = 4
    proto_temperature: float = 0.5
    learnable_temperature: bool = False
    prior_ema_momentum: float = 0.98

    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    mlm_probability: float = 0.15
    rtd_weight: float = 15.0
    generator_size_divisor: int = 2
    span_mask_geo_p: float = 0.2
    span_mask_min_len: int = 1
    span_mask_max_len: int = 10

    sinkhorn_epsilon: float = 0.1
    sinkhorn_iters: int = 20
    lambda_cluster: float = 3.0
    lambda_ortho: float = 1.0

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 512
    warmup_ratio: float = 0.06
    log_every: int = 100
    checkpoint_every_tokens: int = 25_000_000
    num_workers: int = 0

    cluster_start_ratio: float = 0.05
    cluster_ramp_ratio: float = 0.10
    lang_start_ratio: float = 0.15
    lang_ramp_ratio: float = 0.10
    lang_min_entropy: float = 0.70

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if not 0 < self.n_base_layers < self.num_hidden_layers:
            raise ValueError("n_base_layers must be inside the encoder stack")
        if self.num_languages < 2:
            raise ValueError("num_languages must be at least 2")
        if self.generator_size_divisor < 1:
            raise ValueError("generator_size_divisor must be positive")
        if self.max_position_embeddings < self.max_length:
            raise ValueError("max_position_embeddings must cover max_length")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SBERTaConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def darija_small(cls) -> "SBERTaConfig":
        return cls(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            n_base_layers=2,
        )

    @classmethod
    def darija_medium(cls) -> "SBERTaConfig":
        return cls(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            n_base_layers=4,
        )

    @classmethod
    def darija_base(cls) -> "SBERTaConfig":
        return cls()

    small = darija_small
    medium = darija_medium
    base = darija_base
