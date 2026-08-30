"""Canonical multi-label relation schema used by RuREBus."""

from __future__ import annotations


RELATION_LABELS = (
    "GOL",
    "TSK",
    "PNG",
    "PNT",
    "PPS",
    "NNG",
    "NNT",
    "NPS",
    "FNG",
    "FNT",
    "FPS",
)
RELATION_LABEL2ID = {label: index for index, label in enumerate(RELATION_LABELS)}
RELATION_ID2LABEL = {index: label for label, index in RELATION_LABEL2ID.items()}
NEGATIVE_RELATION_LABEL = "NO_RELATION"


__all__ = [
    "NEGATIVE_RELATION_LABEL",
    "RELATION_ID2LABEL",
    "RELATION_LABEL2ID",
    "RELATION_LABELS",
]
