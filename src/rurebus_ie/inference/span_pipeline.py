"""Span decoding and non-overlapping document-level merging."""

from __future__ import annotations

from typing import Mapping, Sequence

from rurebus_ie.data.span_labels import SPAN_ID2LABEL
from rurebus_ie.inference.ner_pipeline import PredictedEntity


def _overlaps(left: PredictedEntity, right: PredictedEntity) -> bool:
    return left.start < right.end and right.start < left.end


def decode_span_predictions(
    label_ids: Sequence[int],
    char_offsets: Sequence[Sequence[int]],
    text: str,
    *,
    confidences: Sequence[float],
    span_mask: Sequence[bool] | None = None,
    id2label: Mapping[int, str] = SPAN_ID2LABEL,
    confidence_threshold: float = 0.5,
) -> tuple[PredictedEntity, ...]:
    """Convert classified candidates into entities before global NMS."""
    if not (len(label_ids) == len(char_offsets) == len(confidences)):
        raise ValueError("Span labels, offsets and confidences must have equal length")
    if span_mask is not None and len(span_mask) != len(label_ids):
        raise ValueError("span_mask and label_ids must have equal length")
    entities: list[PredictedEntity] = []
    for index, (label_id, raw_offset, confidence) in enumerate(
        zip(label_ids, char_offsets, confidences)
    ):
        if span_mask is not None and not span_mask[index]:
            continue
        label = id2label.get(int(label_id))
        if label is None:
            raise ValueError(f"Unknown span label id: {label_id}")
        if label == "NONE" or float(confidence) < confidence_threshold:
            continue
        start, end = int(raw_offset[0]), int(raw_offset[1])
        if end <= start:
            continue
        entities.append(
            PredictedEntity(
                text=text[start:end],
                entity_type=label,
                start=start,
                end=end,
                confidence=float(confidence),
            )
        )
    return tuple(entities)


def merge_span_predictions(
    predictions: Sequence[PredictedEntity],
    *,
    allow_overlapping: bool = False,
) -> tuple[PredictedEntity, ...]:
    """Deduplicate windows and greedily suppress overlapping candidates."""
    exact: dict[tuple[str, int, int], PredictedEntity] = {}
    for entity in predictions:
        key = (entity.entity_type, entity.start, entity.end)
        if key not in exact or entity.confidence > exact[key].confidence:
            exact[key] = entity
    ranked = sorted(
        exact.values(),
        key=lambda item: (-item.confidence, -(item.end - item.start), item.start),
    )
    kept: list[PredictedEntity] = []
    for candidate in ranked:
        if allow_overlapping or not any(_overlaps(candidate, other) for other in kept):
            kept.append(candidate)
    return tuple(sorted(kept, key=lambda item: (item.start, item.end, item.entity_type)))


__all__ = ["decode_span_predictions", "merge_span_predictions"]
