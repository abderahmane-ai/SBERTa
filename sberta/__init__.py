"""
SBERTa: Switch-aware Bidirectional Encoder Representations from
        Transformers with Alternation.

A transformer encoder for code-switched text, designed for Algerian Darija.
See README.md for architecture description and quick-start guide.
"""

from .config import SBERTaConfig
from .model import (
    LanguagePrototypes,
    SBERTaEmbeddings,
    SBERTaForPreTraining,
    SBERTaLayer,
    SBERTaModel,
)
from .tokenizer import SBERTaTokenizer

__version__ = "0.1.0"

__all__ = [
    # Config
    "SBERTaConfig",
    # Model components
    "LanguagePrototypes",
    "SBERTaEmbeddings",
    "SBERTaLayer",
    "SBERTaModel",
    "SBERTaForPreTraining",
    # Classification
    "SBERTaForSequenceClassification",
    # Tokenizer
    "SBERTaTokenizer",
]
