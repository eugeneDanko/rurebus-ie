"""Internal superclass groups for hierarchical RuREBus Span NER.

The public/fine label vocabulary remains the original RuREBus schema.  Coarse
labels are training-only groups used by the curriculum and auxiliary head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .ner_labels import ENTITY_TYPE_ORDER
from .span_labels import NONE_LABEL, SPAN_LABEL2ID


DEFAULT_SUPERCLASS_GROUPS = (
    ("ACT", "BIN"),
    ("CMP", "QUA"),
    ("ECO", "SOC"),
    ("MET",),
    ("INST",),
)


@dataclass(frozen=True)
class SpanLabelHierarchy:
    """Validated mapping from original labels to internal superclass ids."""

    groups: tuple[tuple[str, ...], ...]
    coarse_labels: tuple[str, ...]
    fine_to_coarse_ids: tuple[int, ...]
    fine_to_coarse_labels: dict[str, str]

    @property
    def num_coarse_labels(self) -> int:
        return len(self.coarse_labels)


def build_span_label_hierarchy(
    groups: Sequence[Sequence[str]] = DEFAULT_SUPERCLASS_GROUPS,
) -> SpanLabelHierarchy:
    """Validate groups and build a stable mapping including ``NONE``.

    Every original RuREBus label must occur exactly once.  The generated
    superclass names (for example ``ACT+BIN``) never replace the fine labels in
    inference; they only make training diagnostics readable.
    """

    normalized = tuple(tuple(str(label) for label in group) for group in groups)
    if not normalized or any(not group for group in normalized):
        raise ValueError("superclass_groups must contain non-empty groups")

    flattened = tuple(label for group in normalized for label in group)
    expected = set(ENTITY_TYPE_ORDER)
    unknown = set(flattened) - expected
    missing = expected - set(flattened)
    duplicates = sorted({label for label in flattened if flattened.count(label) > 1})
    if unknown or missing or duplicates:
        raise ValueError(
            "superclass_groups must contain every RuREBus label exactly once; "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}, "
            f"duplicates={duplicates}"
        )

    coarse_labels = (NONE_LABEL, *("+".join(group) for group in normalized))
    label_to_group = {
        fine_label: coarse_index
        for coarse_index, group in enumerate(normalized, start=1)
        for fine_label in group
    }
    fine_to_coarse_ids = [0] * len(SPAN_LABEL2ID)
    fine_to_coarse_labels = {NONE_LABEL: NONE_LABEL}
    for fine_label, fine_id in SPAN_LABEL2ID.items():
        if fine_label == NONE_LABEL:
            continue
        coarse_id = label_to_group[fine_label]
        fine_to_coarse_ids[fine_id] = coarse_id
        fine_to_coarse_labels[fine_label] = coarse_labels[coarse_id]

    return SpanLabelHierarchy(
        groups=normalized,
        coarse_labels=coarse_labels,
        fine_to_coarse_ids=tuple(fine_to_coarse_ids),
        fine_to_coarse_labels=fine_to_coarse_labels,
    )


__all__ = [
    "DEFAULT_SUPERCLASS_GROUPS",
    "SpanLabelHierarchy",
    "build_span_label_hierarchy",
]
