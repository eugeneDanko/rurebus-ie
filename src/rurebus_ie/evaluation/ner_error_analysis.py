"""Detailed, validation-first error analysis for document-level NER."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rurebus_ie.data.ner_dataset import RuReBusNerDataset
from rurebus_ie.evaluation.ner_metrics import compute_strict_ner_metrics
from rurebus_ie.inference.ner_pipeline import PredictedEntity


ERROR_CATEGORIES = (
    "true_positive",
    "type_error",
    "boundary_error",
    "boundary_and_type_error",
    "false_positive",
    "false_negative",
)


@dataclass(frozen=True)
class _EntityView:
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class NerErrorAnalysis:
    summary: dict[str, Any]
    errors: tuple[dict[str, Any], ...]
    per_class: tuple[dict[str, Any], ...]
    confusion: tuple[dict[str, Any], ...]
    length_breakdown: tuple[dict[str, Any], ...]
    documents: tuple[dict[str, Any], ...]


def load_predictions_jsonl(
    path: str | Path,
) -> dict[str, tuple[PredictedEntity, ...]]:
    """Load predictions produced by ``ner_experiment``."""
    source = Path(path)
    predictions: dict[str, tuple[PredictedEntity, ...]] = {}
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            document_id = str(value["document_id"])
            if document_id in predictions:
                raise ValueError(f"{source}:{line_number}: duplicate document_id {document_id}")
            predictions[document_id] = tuple(
                PredictedEntity(
                    text=str(entity.get("text", "")),
                    entity_type=str(entity.get("type", entity.get("entity_type", ""))),
                    start=int(entity["start"]),
                    end=int(entity["end"]),
                    confidence=float(entity.get("confidence", 1.0)),
                )
                for entity in value.get("entities", ())
            )
    return predictions


def _intersection(left: _EntityView, right: _EntityView) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def _iou(left: _EntityView, right: _EntityView) -> float:
    intersection = _intersection(left, right)
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union else 0.0


def _greedy_pairs(
    gold: Sequence[_EntityView],
    predicted: Sequence[_EntityView],
    gold_indices: set[int],
    predicted_indices: set[int],
    predicate: Any,
) -> list[tuple[int, int]]:
    candidates = []
    for gold_index in gold_indices:
        for predicted_index in predicted_indices:
            gold_entity = gold[gold_index]
            predicted_entity = predicted[predicted_index]
            if predicate(gold_entity, predicted_entity):
                candidates.append(
                    (
                        _iou(gold_entity, predicted_entity),
                        predicted_entity.confidence or 0.0,
                        -abs(gold_entity.start - predicted_entity.start)
                        - abs(gold_entity.end - predicted_entity.end),
                        gold_index,
                        predicted_index,
                    )
                )
    pairs: list[tuple[int, int]] = []
    for _, _, _, gold_index, predicted_index in sorted(candidates, reverse=True):
        if gold_index in gold_indices and predicted_index in predicted_indices:
            gold_indices.remove(gold_index)
            predicted_indices.remove(predicted_index)
            pairs.append((gold_index, predicted_index))
    return pairs


def _length_bucket(text: str) -> str:
    token_count = max(1, len(text.split()))
    if token_count == 1:
        return "1"
    if token_count == 2:
        return "2"
    if token_count <= 4:
        return "3-4"
    if token_count <= 8:
        return "5-8"
    return "9+"


def _window_index(dataset: RuReBusNerDataset) -> dict[str, list[tuple[tuple[int, int], ...]]]:
    windows: dict[str, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    for example in dataset:
        real_offsets = tuple(offset for offset in example.offset_mapping if offset[1] > offset[0])
        windows[example.document_id].append(real_offsets)
    return windows


def _edge_features(
    windows: Sequence[Sequence[tuple[int, int]]],
    start: int,
    end: int,
    edge_token_count: int,
) -> tuple[int | None, bool | None, bool]:
    best_distance: int | None = None
    for offsets in windows:
        if not offsets or start < offsets[0][0] or end > offsets[-1][1]:
            continue
        positions = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start < end and token_end > start
        ]
        if not positions:
            continue
        distance = min(positions[0], len(offsets) - positions[-1] - 1)
        best_distance = distance if best_distance is None else max(best_distance, distance)
    if best_distance is None:
        return None, None, True
    return best_distance, best_distance < edge_token_count, False


def _error_row(
    document_id: str,
    category: str,
    gold: _EntityView | None,
    predicted: _EntityView | None,
    windows: Sequence[Sequence[tuple[int, int]]],
    edge_token_count: int,
) -> dict[str, Any]:
    compared_iou = _iou(gold, predicted) if gold is not None and predicted is not None else None
    reference = gold or predicted
    assert reference is not None
    edge_distance, near_edge, crosses_window = _edge_features(
        windows, reference.start, reference.end, edge_token_count
    )
    return {
        "document_id": document_id,
        "category": category,
        "gold_type": gold.entity_type if gold else None,
        "predicted_type": predicted.entity_type if predicted else None,
        "gold_start": gold.start if gold else None,
        "gold_end": gold.end if gold else None,
        "predicted_start": predicted.start if predicted else None,
        "predicted_end": predicted.end if predicted else None,
        "gold_text": gold.text if gold else None,
        "predicted_text": predicted.text if predicted else None,
        "confidence": predicted.confidence if predicted else None,
        "iou": compared_iou,
        "start_correct": (
            gold.start == predicted.start if gold is not None and predicted is not None else None
        ),
        "end_correct": (
            gold.end == predicted.end if gold is not None and predicted is not None else None
        ),
        "gold_char_length": gold.end - gold.start if gold else None,
        "predicted_char_length": predicted.end - predicted.start if predicted else None,
        "gold_token_bucket": _length_bucket(gold.text) if gold else None,
        "predicted_token_bucket": _length_bucket(predicted.text) if predicted else None,
        "best_window_edge_distance_tokens": edge_distance,
        "near_window_edge": near_edge,
        "crosses_every_window": crosses_window,
    }


def analyze_ner_predictions(
    dataset: RuReBusNerDataset,
    predictions: Mapping[str, Iterable[PredictedEntity]],
    *,
    edge_token_count: int = 16,
) -> NerErrorAnalysis:
    """Match predictions to gold entities and classify strict NER errors."""
    if edge_token_count < 0:
        raise ValueError("edge_token_count must be non-negative")
    unknown_documents = set(predictions) - set(dataset.document_texts)
    if unknown_documents:
        raise ValueError(f"Predictions contain unknown documents: {sorted(unknown_documents)[:5]}")

    windows = _window_index(dataset)
    all_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    strict_predictions: dict[str, tuple[PredictedEntity, ...]] = {}

    for document_id, text in dataset.document_texts.items():
        gold = tuple(
            _EntityView(entity_type, start, end, surface)
            for entity_type, start, end, surface in dataset.gold_entities[document_id]
        )
        source_predictions = tuple(predictions.get(document_id, ()))
        predicted = tuple(
            _EntityView(
                entity.entity_type,
                entity.start,
                entity.end,
                entity.text or text[entity.start : entity.end],
                entity.confidence,
            )
            for entity in source_predictions
        )
        for entity in predicted:
            if entity.start < 0 or entity.end <= entity.start or entity.end > len(text):
                raise ValueError(f"{document_id}: invalid predicted span [{entity.start}, {entity.end})")
        strict_predictions[document_id] = source_predictions
        remaining_gold = set(range(len(gold)))
        remaining_predicted = set(range(len(predicted)))
        matches: list[tuple[str, int, int]] = []

        stages = (
            (
                "true_positive",
                lambda left, right: left.entity_type == right.entity_type
                and left.start == right.start
                and left.end == right.end,
            ),
            (
                "type_error",
                lambda left, right: left.entity_type != right.entity_type
                and left.start == right.start
                and left.end == right.end,
            ),
            (
                "boundary_error",
                lambda left, right: left.entity_type == right.entity_type
                and _intersection(left, right) > 0,
            ),
            (
                "boundary_and_type_error",
                lambda left, right: left.entity_type != right.entity_type
                and _intersection(left, right) > 0,
            ),
        )
        for category, predicate in stages:
            matches.extend(
                (category, gold_index, predicted_index)
                for gold_index, predicted_index in _greedy_pairs(
                    gold, predicted, remaining_gold, remaining_predicted, predicate
                )
            )
        for category, gold_index, predicted_index in matches:
            all_rows.append(
                _error_row(
                    document_id,
                    category,
                    gold[gold_index],
                    predicted[predicted_index],
                    windows[document_id],
                    edge_token_count,
                )
            )
        for predicted_index in sorted(remaining_predicted):
            all_rows.append(
                _error_row(
                    document_id,
                    "false_positive",
                    None,
                    predicted[predicted_index],
                    windows[document_id],
                    edge_token_count,
                )
            )
        for gold_index in sorted(remaining_gold):
            all_rows.append(
                _error_row(
                    document_id,
                    "false_negative",
                    gold[gold_index],
                    None,
                    windows[document_id],
                    edge_token_count,
                )
            )

        exact_count = sum(category == "true_positive" for category, _, _ in matches)
        document_categories = Counter(category for category, _, _ in matches)
        document_rows.append(
            {
                "document_id": document_id,
                "characters": len(text),
                "gold_entities": len(gold),
                "predicted_entities": len(predicted),
                "true_positive": exact_count,
                "strict_false_positive": len(predicted) - exact_count,
                "strict_false_negative": len(gold) - exact_count,
                "type_error": document_categories["type_error"],
                "boundary_error": document_categories["boundary_error"],
                "boundary_and_type_error": document_categories["boundary_and_type_error"],
                "unmatched_false_positive": len(remaining_predicted),
                "unmatched_false_negative": len(remaining_gold),
            }
        )

    metrics = compute_strict_ner_metrics(strict_predictions, dataset.gold_entities)
    category_counts = Counter(row["category"] for row in all_rows)
    gold_total = sum(len(entities) for entities in dataset.gold_entities.values())
    predicted_total = sum(len(entities) for entities in strict_predictions.values())
    true_positive = category_counts["true_positive"]
    boundary_rows = [
        row
        for row in all_rows
        if row["category"] in {"boundary_error", "boundary_and_type_error"}
    ]
    summary = {
        "schema_version": 1,
        "documents": len(dataset.document_texts),
        "gold_entities": gold_total,
        "predicted_entities": predicted_total,
        "strict_counts": {
            "true_positive": true_positive,
            "false_positive": predicted_total - true_positive,
            "false_negative": gold_total - true_positive,
        },
        "strict_metrics": metrics.to_dict(),
        "diagnostic_categories": {category: category_counts[category] for category in ERROR_CATEGORIES},
        "boundary_diagnostics": {
            "matched_errors": len(boundary_rows),
            "mean_iou": (
                sum(float(row["iou"]) for row in boundary_rows) / len(boundary_rows)
                if boundary_rows
                else 0.0
            ),
            "start_correct_rate": (
                sum(bool(row["start_correct"]) for row in boundary_rows) / len(boundary_rows)
                if boundary_rows
                else 0.0
            ),
            "end_correct_rate": (
                sum(bool(row["end_correct"]) for row in boundary_rows) / len(boundary_rows)
                if boundary_rows
                else 0.0
            ),
            "near_window_edge": sum(row["near_window_edge"] is True for row in boundary_rows),
        },
        "window_edge_threshold_tokens": edge_token_count,
    }

    per_class = tuple(
        {"entity_type": entity_type, **values}
        for entity_type, values in metrics.per_class.items()
    )
    confusion_counter: Counter[tuple[str, str, str]] = Counter()
    for row in all_rows:
        if row["category"] in {"type_error", "boundary_and_type_error"}:
            confusion_counter[
                (row["gold_type"], row["predicted_type"], row["category"])
            ] += 1
    confusion = tuple(
        {
            "gold_type": gold_type,
            "predicted_type": predicted_type,
            "category": category,
            "count": count,
        }
        for (gold_type, predicted_type, category), count in sorted(confusion_counter.items())
    )

    exact_gold = Counter()
    total_gold = Counter()
    exact_predicted = Counter()
    total_predicted = Counter()
    for row in all_rows:
        if row["gold_token_bucket"] is not None:
            total_gold[row["gold_token_bucket"]] += 1
            if row["category"] == "true_positive":
                exact_gold[row["gold_token_bucket"]] += 1
        if row["predicted_token_bucket"] is not None:
            total_predicted[row["predicted_token_bucket"]] += 1
            if row["category"] == "true_positive":
                exact_predicted[row["predicted_token_bucket"]] += 1
    length_breakdown = tuple(
        {
            "token_bucket": bucket,
            "gold": total_gold[bucket],
            "exact_gold": exact_gold[bucket],
            "recall": exact_gold[bucket] / total_gold[bucket] if total_gold[bucket] else 0.0,
            "predicted": total_predicted[bucket],
            "exact_predicted": exact_predicted[bucket],
            "precision": (
                exact_predicted[bucket] / total_predicted[bucket]
                if total_predicted[bucket]
                else 0.0
            ),
        }
        for bucket in ("1", "2", "3-4", "5-8", "9+")
    )
    return NerErrorAnalysis(
        summary=summary,
        errors=tuple(all_rows),
        per_class=per_class,
        confusion=confusion,
        length_breakdown=length_breakdown,
        documents=tuple(document_rows),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_ner_error_analysis(
    analysis: NerErrorAnalysis,
    output_dir: str | Path,
) -> Path:
    """Write machine-readable and spreadsheet-friendly analysis artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(analysis.summary, stream, ensure_ascii=False, indent=2)
    with (output / "errors.jsonl").open("w", encoding="utf-8") as stream:
        for row in analysis.errors:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv(output / "errors.csv", analysis.errors)
    _write_csv(output / "per_class.csv", analysis.per_class)
    _write_csv(output / "type_confusion.csv", analysis.confusion)
    _write_csv(output / "length_breakdown.csv", analysis.length_breakdown)
    _write_csv(output / "documents.csv", analysis.documents)
    return output


__all__ = [
    "ERROR_CATEGORIES",
    "NerErrorAnalysis",
    "analyze_ner_predictions",
    "load_predictions_jsonl",
    "write_ner_error_analysis",
]
