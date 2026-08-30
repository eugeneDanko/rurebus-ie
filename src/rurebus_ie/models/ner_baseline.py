"""RuBERT token-classification model factory.

The model will use ``AutoModelForTokenClassification`` with the canonical
RuREBus BIO label mappings. Loading Transformers is deferred until the factory
is called, keeping data-only package imports lightweight.
"""

from __future__ import annotations

from typing import Any

from rurebus_ie.data.ner_labels import ID2LABEL, LABEL2ID


def build_rubert_token_classifier(
    pretrained_name: str,
    *,
    dropout: float | None = None,
) -> Any:
    """Build the pretrained RuBERT encoder and linear token-classification head."""
    try:
        from transformers import AutoConfig, AutoModelForTokenClassification
    except ImportError as error:
        raise RuntimeError(
            "transformers is required to build RuBERT. Install the project dependencies first."
        ) from error

    config = AutoConfig.from_pretrained(
        pretrained_name,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    if dropout is not None:
        config.hidden_dropout_prob = dropout
        config.classifier_dropout = dropout
    return AutoModelForTokenClassification.from_pretrained(
        pretrained_name,
        config=config,
        # The encoder checkpoint may contain a pretraining/task head with a
        # different output size. The RuREBus 17-label head must be initialized.
        ignore_mismatched_sizes=True,
    )


def build_rubert_tokenizer(pretrained_name: str) -> Any:
    """Load the fast tokenizer required for character-offset alignment."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required to load the RuBERT tokenizer."
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name,
        use_fast=True,
        # Supported by modern Transformers and ignored by tokenizers that do
        # not use the affected regex implementation.
        fix_mistral_regex=True,
    )
    if not tokenizer.is_fast:
        raise ValueError(f"{pretrained_name!r} does not provide a fast tokenizer")
    return tokenizer


__all__ = [
    "ID2LABEL",
    "LABEL2ID",
    "build_rubert_token_classifier",
    "build_rubert_tokenizer",
]
