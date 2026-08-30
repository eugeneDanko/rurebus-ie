"""Strict positive-relation metrics for component and end-to-end evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from rurebus_ie.data.brat_parser import BratDocument
from rurebus_ie.data.relation_labels import RELATION_LABELS
from rurebus_ie.inference.relation_pipeline import PredictedRelation


@dataclass(frozen=True)
class RelationMetrics:
    precision: float
    recall: float
    micro_f1: float
    macro_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": tp + fn,
        "predicted": tp + fp,
    }


def _metrics_from_sets(predicted: set[tuple], gold: set[tuple]) -> RelationMetrics:
    micro = _scores(len(predicted & gold), len(predicted - gold), len(gold - predicted))
    per_class: dict[str, dict[str, float]] = {}
    for label in RELATION_LABELS:
        predicted_class = {item for item in predicted if item[1] == label}
        gold_class = {item for item in gold if item[1] == label}
        per_class[label] = _scores(
            len(predicted_class & gold_class),
            len(predicted_class - gold_class),
            len(gold_class - predicted_class),
        )
    return RelationMetrics(
        precision=micro["precision"],
        recall=micro["recall"],
        micro_f1=micro["f1"],
        macro_f1=sum(row["f1"] for row in per_class.values()) / len(per_class),
        per_class=per_class,
    )


def compute_relation_metrics(
    predictions: Iterable[PredictedRelation],
    references: Iterable[tuple[str, str, str, str]],
) -> RelationMetrics:
    predicted = {
        (item.document_id, item.relation_type, item.arg1_id, item.arg2_id)
        for item in predictions
    }
    gold = {
        (str(document), str(label), str(arg1), str(arg2))
        for document, label, arg1, arg2 in references
    }
    return _metrics_from_sets(predicted, gold)


def compute_end_to_end_relation_metrics(
    predictions: Iterable[PredictedRelation],
    gold_documents: Sequence[BratDocument],
) -> RelationMetrics:
    """Match type plus exact typed boundaries of both relation arguments."""
    predicted: set[tuple] = set()
    for item in predictions:
        if item.arg1_signature is None or item.arg2_signature is None:
            raise ValueError("End-to-end predictions require argument signatures")
        predicted.add(
            (
                item.document_id,
                item.relation_type,
                *item.arg1_signature,
                *item.arg2_signature,
            )
        )
    gold: set[tuple] = set()
    for document in gold_documents:
        entities = document.entity_by_id
        for relation in document.relations:
            arg1, arg2 = entities[relation.arg1], entities[relation.arg2]
            gold.add(
                (
                    document.document_id,
                    relation.relation_type,
                    arg1.entity_type,
                    arg1.start,
                    arg1.end,
                    arg2.entity_type,
                    arg2.start,
                    arg2.end,
                )
            )
    return _metrics_from_sets(predicted, gold)


__all__ = [
    "RelationMetrics",
    "compute_end_to_end_relation_metrics",
    "compute_relation_metrics",
]
