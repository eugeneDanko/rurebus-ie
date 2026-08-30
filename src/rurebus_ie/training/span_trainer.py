"""Training and strict document evaluation for Span NER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rurebus_ie.data.ner_labels import ENTITY_TYPE_ORDER
from rurebus_ie.evaluation.ner_metrics import NerMetrics, compute_strict_ner_metrics
from rurebus_ie.inference.ner_pipeline import PredictedEntity
from rurebus_ie.inference.span_pipeline import (
    decode_span_predictions,
    merge_span_predictions,
)
from rurebus_ie.training.ner_trainer import (
    NerEvaluationResult,
    NerTrainer,
    _require_torch,
)


@dataclass(frozen=True)
class SpanThresholdCalibration:
    """Validation sweep result produced from one model inference pass."""

    best_threshold: float
    best_metrics: NerMetrics
    rows: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class SpanClassThresholdCalibration:
    """Class-specific validation thresholds selected by coordinate descent."""

    initial_threshold: float
    class_thresholds: dict[str, float]
    best_metrics: NerMetrics
    trace: tuple[dict[str, float | int | str], ...]


class SpanNerTrainer(NerTrainer):
    """Reuse the baseline loop with span inputs, decoding and differential LR."""

    @staticmethod
    def _model_inputs(
        batch: Mapping[str, Any], device: Any, *, include_labels: bool
    ) -> dict[str, Any]:
        keys = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "span_starts",
            "span_ends",
            "span_widths",
            "span_mask",
        }
        if include_labels:
            keys.add("labels")
        return {key: value.to(device) for key, value in batch.items() if key in keys}

    def _build_optimizer(self, torch: Any) -> Any:
        training = self.training_config
        default_lr = float(training.get("learning_rate", 2e-5))
        encoder_lr = float(training.get("encoder_learning_rate", default_lr))
        head_lr = float(training.get("head_learning_rate", default_lr))
        weight_decay = float(training.get("weight_decay", 0.01))
        encoder_parameters = list(self.model.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        head_parameters = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in encoder_ids
        ]
        return torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": encoder_lr},
                {"params": head_parameters, "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )

    def _collect_candidates(
        self,
        data_loader: Any,
        *,
        minimum_confidence: float,
        include_labels: bool = True,
    ) -> tuple[float, dict[str, list[PredictedEntity]]]:
        """Run the encoder once and retain decoded span candidates.

        Threshold calibration needs only fine-head logits. Passing gold labels
        there would unnecessarily execute fine/coarse/contrastive losses and
        their CUDA indexing path, so calibration explicitly disables labels.
        """
        torch = _require_torch()
        device = self._device(torch)
        self.model.to(device)
        self.model.eval()
        dataset = data_loader.dataset
        document_texts = getattr(dataset, "document_texts", None)
        gold_entities = getattr(dataset, "gold_entities", None)
        if document_texts is None or gold_entities is None:
            raise ValueError("Span dataset must expose document_texts and gold_entities")

        id2label = {
            int(key): value
            for key, value in getattr(self.model.config, "id2label", {}).items()
        }
        predictions: dict[str, list[PredictedEntity]] = {
            document_id: [] for document_id in document_texts
        }
        total_loss = 0.0
        loss_batch_count = 0
        with torch.inference_mode():
            for batch in data_loader:
                outputs = self.model(
                    **self._model_inputs(
                        batch, device, include_labels=include_labels
                    )
                )
                if outputs.loss is not None:
                    total_loss += float(outputs.loss.detach().cpu())
                    loss_batch_count += 1
                probabilities = torch.softmax(outputs.logits, dim=-1)
                confidence, label_ids = probabilities.max(dim=-1)
                masks = batch["span_mask"].detach().cpu().tolist()
                for index, document_id in enumerate(batch["document_id"]):
                    predictions[document_id].extend(
                        decode_span_predictions(
                            label_ids[index].detach().cpu().tolist(),
                            batch["span_char_offsets"][index],
                            document_texts[document_id],
                            id2label=id2label,
                            confidences=confidence[index].detach().cpu().tolist(),
                            span_mask=masks[index],
                            confidence_threshold=minimum_confidence,
                        )
                    )
        return (
            total_loss / loss_batch_count if loss_batch_count else 0.0,
            predictions,
        )

    @staticmethod
    def _predictions_at_threshold(
        candidates: Mapping[str, Sequence[PredictedEntity]],
        *,
        threshold: float,
        allow_overlapping: bool,
    ) -> dict[str, tuple[PredictedEntity, ...]]:
        return {
            document_id: merge_span_predictions(
                tuple(entity for entity in entities if entity.confidence >= threshold),
                allow_overlapping=allow_overlapping,
            )
            for document_id, entities in candidates.items()
        }

    @staticmethod
    def _predictions_at_class_thresholds(
        candidates: Mapping[str, Sequence[PredictedEntity]],
        *,
        class_thresholds: Mapping[str, float],
        default_threshold: float,
        allow_overlapping: bool,
    ) -> dict[str, tuple[PredictedEntity, ...]]:
        return {
            document_id: merge_span_predictions(
                tuple(
                    entity
                    for entity in entities
                    if entity.confidence
                    >= float(class_thresholds.get(entity.entity_type, default_threshold))
                ),
                allow_overlapping=allow_overlapping,
            )
            for document_id, entities in candidates.items()
        }

    def evaluate(self, data_loader: Any) -> NerEvaluationResult:
        dataset = data_loader.dataset
        gold_entities = getattr(dataset, "gold_entities", None)
        if gold_entities is None:
            raise ValueError("Span dataset must expose gold_entities")
        decoding = self.config.get("decoding", self.config.get("evaluation", {}))
        threshold = float(decoding.get("confidence_threshold", 0.5))
        class_thresholds = {
            str(label): float(value)
            for label, value in decoding.get("class_thresholds", {}).items()
        }
        allow_overlapping = bool(decoding.get("allow_overlapping", False))
        loss, candidates = self._collect_candidates(
            data_loader,
            minimum_confidence=min((threshold, *class_thresholds.values())),
        )
        if class_thresholds:
            merged = self._predictions_at_class_thresholds(
                candidates,
                class_thresholds=class_thresholds,
                default_threshold=threshold,
                allow_overlapping=allow_overlapping,
            )
        else:
            merged = self._predictions_at_threshold(
                candidates,
                threshold=threshold,
                allow_overlapping=allow_overlapping,
            )
        metrics = compute_strict_ner_metrics(merged, gold_entities)
        return NerEvaluationResult(loss=loss, metrics=metrics, predictions=merged)

    def calibrate_thresholds(
        self,
        data_loader: Any,
        thresholds: Sequence[float],
    ) -> SpanThresholdCalibration:
        """Select validation micro-F1 threshold without repeated encoder passes."""
        values = tuple(sorted({round(float(value), 10) for value in thresholds}))
        if not values:
            raise ValueError("At least one confidence threshold is required")
        if values[0] < 0.0 or values[-1] > 1.0:
            raise ValueError("Confidence thresholds must be in the [0, 1] interval")
        gold_entities = getattr(data_loader.dataset, "gold_entities", None)
        if gold_entities is None:
            raise ValueError("Span dataset must expose gold_entities")
        decoding = self.config.get("decoding", self.config.get("evaluation", {}))
        allow_overlapping = bool(decoding.get("allow_overlapping", False))
        _, candidates = self._collect_candidates(
            data_loader,
            minimum_confidence=values[0],
            include_labels=False,
        )
        rows: list[dict[str, float | int]] = []
        metrics_by_threshold: dict[float, NerMetrics] = {}
        for threshold in values:
            predictions = self._predictions_at_threshold(
                candidates,
                threshold=threshold,
                allow_overlapping=allow_overlapping,
            )
            metrics = compute_strict_ner_metrics(predictions, gold_entities)
            metrics_by_threshold[threshold] = metrics
            rows.append(
                {
                    "threshold": threshold,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "micro_f1": metrics.micro_f1,
                    "macro_f1": metrics.macro_f1,
                    "predicted_entities": sum(len(items) for items in predictions.values()),
                }
            )
        best_row = max(
            rows,
            key=lambda row: (
                float(row["micro_f1"]),
                float(row["precision"]),
                float(row["threshold"]),
            ),
        )
        best_threshold = float(best_row["threshold"])
        return SpanThresholdCalibration(
            best_threshold=best_threshold,
            best_metrics=metrics_by_threshold[best_threshold],
            rows=tuple(rows),
        )

    def calibrate_class_thresholds(
        self,
        data_loader: Any,
        thresholds: Sequence[float],
        *,
        initial_threshold: float,
        max_rounds: int = 2,
        entity_types: Sequence[str] = ENTITY_TYPE_ORDER,
    ) -> SpanClassThresholdCalibration:
        """Optimize per-type thresholds on validation by coordinate descent.

        Every trial is evaluated after the same cross-type non-overlap decoder,
        so the selected values reflect actual document-level micro-F1 rather
        than isolated token or candidate accuracy.
        """
        values = tuple(sorted({round(float(value), 10) for value in thresholds}))
        if not values:
            raise ValueError("At least one class confidence threshold is required")
        if values[0] < 0.0 or values[-1] > 1.0:
            raise ValueError("Class confidence thresholds must be in [0, 1]")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        tuned_types = tuple(dict.fromkeys(str(value) for value in entity_types))
        unknown_types = set(tuned_types) - set(ENTITY_TYPE_ORDER)
        if not tuned_types or unknown_types:
            raise ValueError(
                "entity_types must contain known RuREBus labels; "
                f"unknown={sorted(unknown_types)}"
            )
        initial = float(initial_threshold)
        if not 0.0 <= initial <= 1.0:
            raise ValueError("initial_threshold must be in [0, 1]")
        gold_entities = getattr(data_loader.dataset, "gold_entities", None)
        if gold_entities is None:
            raise ValueError("Span dataset must expose gold_entities")
        decoding = self.config.get("decoding", self.config.get("evaluation", {}))
        allow_overlapping = bool(decoding.get("allow_overlapping", False))
        _, candidates = self._collect_candidates(
            data_loader,
            minimum_confidence=min(values[0], initial),
            include_labels=False,
        )
        selected = {entity_type: initial for entity_type in tuned_types}

        def score(threshold_map: Mapping[str, float]) -> tuple[NerMetrics, int]:
            predictions = self._predictions_at_class_thresholds(
                candidates,
                class_thresholds=threshold_map,
                default_threshold=initial,
                allow_overlapping=allow_overlapping,
            )
            return (
                compute_strict_ner_metrics(predictions, gold_entities),
                sum(len(items) for items in predictions.values()),
            )

        current_metrics, current_count = score(selected)
        trace: list[dict[str, float | int | str]] = [
            {
                "round": 0,
                "entity_type": "GLOBAL_START",
                "threshold": initial,
                "precision": current_metrics.precision,
                "recall": current_metrics.recall,
                "micro_f1": current_metrics.micro_f1,
                "macro_f1": current_metrics.macro_f1,
                "predicted_entities": current_count,
            }
        ]
        for round_index in range(1, max_rounds + 1):
            changed = False
            for entity_type in tuned_types:
                best_value = selected[entity_type]
                best_metrics = current_metrics
                best_count = current_count
                for value in values:
                    trial = dict(selected)
                    trial[entity_type] = value
                    metrics, predicted_count = score(trial)
                    trial_key = (metrics.micro_f1, metrics.precision, value)
                    best_key = (
                        best_metrics.micro_f1,
                        best_metrics.precision,
                        best_value,
                    )
                    if trial_key > best_key:
                        best_value = value
                        best_metrics = metrics
                        best_count = predicted_count
                if best_value != selected[entity_type]:
                    changed = True
                selected[entity_type] = best_value
                current_metrics = best_metrics
                current_count = best_count
                trace.append(
                    {
                        "round": round_index,
                        "entity_type": entity_type,
                        "threshold": best_value,
                        "precision": current_metrics.precision,
                        "recall": current_metrics.recall,
                        "micro_f1": current_metrics.micro_f1,
                        "macro_f1": current_metrics.macro_f1,
                        "predicted_entities": current_count,
                    }
                )
            if not changed:
                break
        return SpanClassThresholdCalibration(
            initial_threshold=initial,
            class_thresholds=selected,
            best_metrics=current_metrics,
            trace=tuple(trace),
        )


__all__ = [
    "SpanClassThresholdCalibration",
    "SpanNerTrainer",
    "SpanThresholdCalibration",
]
