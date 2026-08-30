"""Threshold decoding and NER-to-RE document conversion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rurebus_ie.data.brat_parser import BratDocument, Entity, TextSpan
from rurebus_ie.data.relation_labels import RELATION_LABELS


@dataclass(frozen=True)
class PredictedRelation:
    document_id: str
    relation_type: str
    arg1_id: str
    arg2_id: str
    confidence: float
    arg1_signature: tuple[str, int, int] | None = None
    arg2_signature: tuple[str, int, int] | None = None


def decode_relation_scores(
    document_ids: Sequence[str],
    arg1_ids: Sequence[str],
    arg2_ids: Sequence[str],
    scores: Sequence[Sequence[float]],
    *,
    threshold: float,
    arg1_signatures: Sequence[tuple[str, int, int]] | None = None,
    arg2_signatures: Sequence[tuple[str, int, int]] | None = None,
) -> tuple[PredictedRelation, ...]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be inside [0, 1]")
    size = len(document_ids)
    if not (len(arg1_ids) == len(arg2_ids) == len(scores) == size):
        raise ValueError("Relation metadata and scores must have equal lengths")
    if arg1_signatures is not None and len(arg1_signatures) != size:
        raise ValueError("arg1_signatures length mismatch")
    if arg2_signatures is not None and len(arg2_signatures) != size:
        raise ValueError("arg2_signatures length mismatch")
    predictions: list[PredictedRelation] = []
    for row_index, row_scores in enumerate(scores):
        if len(row_scores) != len(RELATION_LABELS):
            raise ValueError("Each score row must match RELATION_LABELS")
        for label, score in zip(RELATION_LABELS, row_scores):
            confidence = float(score)
            if confidence >= threshold:
                predictions.append(
                    PredictedRelation(
                        document_id=document_ids[row_index],
                        relation_type=label,
                        arg1_id=arg1_ids[row_index],
                        arg2_id=arg2_ids[row_index],
                        confidence=confidence,
                        arg1_signature=(
                            None if arg1_signatures is None else arg1_signatures[row_index]
                        ),
                        arg2_signature=(
                            None if arg2_signatures is None else arg2_signatures[row_index]
                        ),
                    )
                )
    return tuple(predictions)


def load_ner_prediction_documents(
    predictions_path: str | Path,
    gold_documents: Sequence[BratDocument],
) -> tuple[BratDocument, ...]:
    """Attach predicted entities to gold texts without exposing gold entities."""
    source = Path(predictions_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    predictions: dict[str, list[dict[str, object]]] = {}
    with source.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            predictions[str(row["document_id"])] = list(row["entities"])
    expected = {document.document_id for document in gold_documents}
    if set(predictions) != expected:
        missing = sorted(expected - set(predictions))[:10]
        extra = sorted(set(predictions) - expected)[:10]
        raise ValueError(f"NER prediction document mismatch; missing={missing}, extra={extra}")

    result: list[BratDocument] = []
    for document in gold_documents:
        entities: list[Entity] = []
        for index, row in enumerate(predictions[document.document_id], start=1):
            start, end = int(row["start"]), int(row["end"])
            text = str(row["text"])
            if document.text[start:end] != text:
                raise ValueError(
                    f"NER offset mismatch in {document.document_id}: {start}:{end} {text!r}"
                )
            entities.append(
                Entity(
                    entity_id=f"P{index}",
                    entity_type=str(row["type"]),
                    spans=(TextSpan(start, end),),
                    text=text,
                )
            )
        result.append(
            BratDocument(
                document_id=document.document_id,
                text=document.text,
                entities=tuple(entities),
                relations=(),
                txt_path=document.txt_path,
                ann_path=document.ann_path,
            )
        )
    return tuple(result)


__all__ = [
    "PredictedRelation",
    "decode_relation_scores",
    "load_ner_prediction_documents",
]
