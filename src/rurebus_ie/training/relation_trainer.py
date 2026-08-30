"""Training, calibration and evaluation for the multi-label R-BERT head."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from rurebus_ie.evaluation.relation_metrics import (
    RelationMetrics,
    compute_relation_metrics,
)
from rurebus_ie.inference.relation_pipeline import (
    PredictedRelation,
    decode_relation_scores,
)
from rurebus_ie.training.ner_trainer import NerTrainer, _require_torch


@dataclass(frozen=True)
class RelationEvaluationResult:
    loss: float
    metrics: RelationMetrics
    predictions: tuple[PredictedRelation, ...] = field(default_factory=tuple)
    candidate_recall: float = 1.0
    candidate_pairs: int = 0


@dataclass(frozen=True)
class RelationThresholdCalibration:
    best_threshold: float
    best_metrics: RelationMetrics
    rows: tuple[dict[str, float], ...]


class RelationTrainer(NerTrainer):
    def _build_optimizer(self, torch: Any) -> Any:
        training = self.training_config
        encoder_parameters = list(self.model.encoder.parameters())
        head_parameters = list(self.model.classifier.parameters())
        return torch.optim.AdamW(
            (
                {
                    "params": encoder_parameters,
                    "lr": float(training.get("encoder_learning_rate", 2e-5)),
                },
                {
                    "params": head_parameters,
                    "lr": float(training.get("head_learning_rate", 1e-4)),
                },
            ),
            weight_decay=float(training.get("weight_decay", 0.01)),
        )

    def _save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        evaluation: RelationEvaluationResult,
    ) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
        payload = {
            "epoch": epoch,
            "validation_loss": evaluation.loss,
            **evaluation.metrics.to_dict(),
            "candidate_recall": evaluation.candidate_recall,
            "candidate_pairs": evaluation.candidate_pairs,
        }
        for path in (
            checkpoint_dir / "validation_metrics.json",
            self.output_dir / "validation_metrics.json",
        ):
            with path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)

    @staticmethod
    def _model_inputs(
        batch: Mapping[str, Any], device: Any, *, include_labels: bool
    ) -> dict[str, Any]:
        keys = {"input_ids", "attention_mask", "token_type_ids"}
        if include_labels:
            keys.add("labels")
        return {key: value.to(device) for key, value in batch.items() if key in keys}

    def _collect_scores(
        self, data_loader: Any, *, include_labels: bool = True
    ) -> tuple[float, dict[str, list[Any]]]:
        torch = _require_torch()
        device = self._device(torch)
        self.model.to(device)
        self.model.eval()
        collected: dict[str, list[Any]] = {
            "document_id": [],
            "arg1_id": [],
            "arg2_id": [],
            "arg1_signature": [],
            "arg2_signature": [],
            "scores": [],
        }
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(
                    **self._model_inputs(batch, device, include_labels=include_labels)
                )
                if outputs.loss is not None:
                    total_loss += float(outputs.loss.detach().cpu())
                    batch_count += 1
                collected["document_id"].extend(batch["document_id"])
                collected["arg1_id"].extend(batch["arg1_id"])
                collected["arg2_id"].extend(batch["arg2_id"])
                collected["arg1_signature"].extend(batch["arg1_signature"])
                collected["arg2_signature"].extend(batch["arg2_signature"])
                collected["scores"].extend(
                    torch.sigmoid(outputs.logits).detach().cpu().tolist()
                )
        return total_loss / batch_count if batch_count else 0.0, collected

    @staticmethod
    def _decode(collected: Mapping[str, list[Any]], threshold: float) -> tuple[PredictedRelation, ...]:
        return decode_relation_scores(
            collected["document_id"],
            collected["arg1_id"],
            collected["arg2_id"],
            collected["scores"],
            threshold=threshold,
            arg1_signatures=collected["arg1_signature"],
            arg2_signatures=collected["arg2_signature"],
        )

    def evaluate(
        self, data_loader: Any, *, threshold: float | None = None
    ) -> RelationEvaluationResult:
        dataset = data_loader.dataset
        references = getattr(dataset, "gold_relations", None)
        if references is None:
            raise ValueError("Relation dataset must expose gold_relations")
        if threshold is None:
            threshold = float(self.config.get("decoding", {}).get("confidence_threshold", 0.5))
        loss, collected = self._collect_scores(data_loader, include_labels=True)
        predictions = self._decode(collected, threshold)
        return RelationEvaluationResult(
            loss=loss,
            metrics=compute_relation_metrics(predictions, references),
            predictions=predictions,
            candidate_recall=float(getattr(dataset, "candidate_recall", 1.0)),
            candidate_pairs=len(dataset),
        )

    def predict(
        self, data_loader: Any, *, threshold: float
    ) -> tuple[PredictedRelation, ...]:
        _, collected = self._collect_scores(data_loader, include_labels=False)
        return self._decode(collected, threshold)

    def calibrate_thresholds(
        self, data_loader: Any, thresholds: list[float] | tuple[float, ...]
    ) -> RelationThresholdCalibration:
        values = sorted({float(value) for value in thresholds})
        if not values or any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("thresholds must be a non-empty sequence inside [0, 1]")
        references = data_loader.dataset.gold_relations
        _, collected = self._collect_scores(data_loader, include_labels=False)
        rows: list[dict[str, float]] = []
        scored: list[tuple[float, RelationMetrics]] = []
        for threshold in values:
            predictions = self._decode(collected, threshold)
            metrics = compute_relation_metrics(predictions, references)
            scored.append((threshold, metrics))
            rows.append(
                {
                    "threshold": threshold,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "micro_f1": metrics.micro_f1,
                    "macro_f1": metrics.macro_f1,
                    "predicted_relations": float(len(predictions)),
                    "candidate_recall": float(data_loader.dataset.candidate_recall),
                }
            )
        # Primary metric first; precision and then the higher threshold provide
        # deterministic conservative tie-breaking.
        best_threshold, best_metrics = max(
            scored,
            key=lambda item: (
                item[1].micro_f1,
                item[1].precision,
                item[0],
            ),
        )
        return RelationThresholdCalibration(
            best_threshold=best_threshold,
            best_metrics=best_metrics,
            rows=tuple(rows),
        )


__all__ = [
    "RelationEvaluationResult",
    "RelationThresholdCalibration",
    "RelationTrainer",
]
