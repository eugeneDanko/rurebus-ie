"""Strict document-level metrics for RuREBus named entities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from rurebus_ie.data.ner_labels import ENTITY_TYPE_ORDER


@dataclass(frozen=True)
class NerMetrics:
    precision: float
    recall: float
    micro_f1: float
    macro_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entity_key(entity: Any) -> tuple[str, int, int]:
    if isinstance(entity, (tuple, list)):
        if len(entity) < 3:
            raise ValueError(f"Entity tuple must contain type, start and end: {entity!r}")
        return str(entity[0]), int(entity[1]), int(entity[2])
    try:
        return str(entity.entity_type), int(entity.start), int(entity.end)
    except AttributeError as error:
        raise TypeError(f"Unsupported entity value: {entity!r}") from error


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _scores(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": true_positive + false_negative,
        "predicted": true_positive + false_positive,
    }


def compute_strict_ner_metrics(
    predictions: Mapping[str, Iterable[Any]],
    references: Mapping[str, Iterable[Any]],
    *,
    entity_types: Iterable[str] = ENTITY_TYPE_ORDER,
) -> NerMetrics:
    """Compare entities by exact ``(document, type, start, end)`` equality."""
    document_ids = set(predictions) | set(references)
    predicted = {
        (document_id, *_entity_key(entity))
        for document_id in document_ids
        for entity in predictions.get(document_id, ())
    }
    gold = {
        (document_id, *_entity_key(entity))
        for document_id in document_ids
        for entity in references.get(document_id, ())
    }

    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    micro = _scores(true_positive, false_positive, false_negative)

    labels = tuple(entity_types)
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        predicted_class = {entity for entity in predicted if entity[1] == label}
        gold_class = {entity for entity in gold if entity[1] == label}
        per_class[label] = _scores(
            len(predicted_class & gold_class),
            len(predicted_class - gold_class),
            len(gold_class - predicted_class),
        )
    macro_f1 = sum(values["f1"] for values in per_class.values()) / len(labels) if labels else 0.0

    return NerMetrics(
        precision=micro["precision"],
        recall=micro["recall"],
        micro_f1=micro["f1"],
        macro_f1=macro_f1,
        per_class=per_class,
    )


__all__ = ["NerMetrics", "compute_strict_ner_metrics"]
