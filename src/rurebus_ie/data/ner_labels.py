"""Canonical entity order and BIO label schema for RuREBus NER."""

from __future__ import annotations


ENTITY_TYPE_ORDER = ("MET", "ECO", "BIN", "CMP", "QUA", "ACT", "INST", "SOC")


def build_bio_labels(entity_types: tuple[str, ...] = ENTITY_TYPE_ORDER) -> tuple[str, ...]:
    """Build a stable O/B-/I- label order for token classification."""
    labels = ["O"]
    for entity_type in entity_types:
        labels.extend((f"B-{entity_type}", f"I-{entity_type}"))
    return tuple(labels)


BIO_LABELS = build_bio_labels()
LABEL2ID = {label: index for index, label in enumerate(BIO_LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
IGNORE_LABEL_ID = -100


__all__ = [
    "BIO_LABELS",
    "ENTITY_TYPE_ORDER",
    "ID2LABEL",
    "IGNORE_LABEL_ID",
    "LABEL2ID",
    "build_bio_labels",
]

