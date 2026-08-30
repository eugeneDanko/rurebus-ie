"""Canonical label schema for span-based RuREBus NER."""

from __future__ import annotations

from .ner_labels import ENTITY_TYPE_ORDER, IGNORE_LABEL_ID


NONE_LABEL = "NONE"
SPAN_LABELS = (NONE_LABEL, *ENTITY_TYPE_ORDER)
SPAN_LABEL2ID = {label: index for index, label in enumerate(SPAN_LABELS)}
SPAN_ID2LABEL = {index: label for label, index in SPAN_LABEL2ID.items()}


__all__ = [
    "IGNORE_LABEL_ID",
    "NONE_LABEL",
    "SPAN_ID2LABEL",
    "SPAN_LABEL2ID",
    "SPAN_LABELS",
]
