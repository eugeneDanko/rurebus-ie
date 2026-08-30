"""Two-stage curriculum trainer for hierarchical Span NER."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from rurebus_ie.data.span_hierarchy import build_span_label_hierarchy
from rurebus_ie.evaluation.ner_metrics import compute_strict_ner_metrics
from rurebus_ie.inference.ner_pipeline import PredictedEntity
from rurebus_ie.inference.span_pipeline import (
    decode_span_predictions,
    merge_span_predictions,
)
from rurebus_ie.training.ner_trainer import (
    NerEvaluationResult,
    NerTrainingSummary,
    _require_torch,
    set_seed,
)
from rurebus_ie.training.span_trainer import SpanNerTrainer


class HierarchicalSpanNerTrainer(SpanNerTrainer):
    """Train coarse superclasses first and original RuREBus labels second."""

    def _build_stage_optimizer(self, torch: Any, stage: str) -> Any:
        training = self.training_config
        default_lr = float(training.get("learning_rate", 2e-5))
        encoder_lr = float(
            training.get(
                f"{stage}_encoder_learning_rate",
                training.get("encoder_learning_rate", default_lr),
            )
        )
        head_lr = float(
            training.get(
                f"{stage}_head_learning_rate",
                training.get("head_learning_rate", default_lr),
            )
        )
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

    def _collect_coarse_candidates(
        self,
        data_loader: Any,
        *,
        minimum_confidence: float,
    ) -> tuple[float, dict[str, list[PredictedEntity]]]:
        torch = _require_torch()
        device = self._device(torch)
        self.model.to(device)
        self.model.set_training_stage("coarse")
        self.model.eval()
        dataset = data_loader.dataset
        document_texts = getattr(dataset, "document_texts", None)
        if document_texts is None:
            raise ValueError("Span dataset must expose document_texts")
        id2label = {
            int(key): value
            for key, value in self.model.config.coarse_id2label.items()
        }
        predictions: dict[str, list[PredictedEntity]] = {
            document_id: [] for document_id in document_texts
        }
        total_loss = 0.0
        batch_count = 0
        with torch.inference_mode():
            for batch in data_loader:
                outputs = self.model(
                    **self._model_inputs(batch, device, include_labels=True)
                )
                total_loss += float(outputs.loss.detach().cpu())
                batch_count += 1
                probabilities = torch.softmax(outputs.coarse_logits, dim=-1)
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
        return total_loss / batch_count if batch_count else 0.0, predictions

    def evaluate_coarse(self, data_loader: Any) -> NerEvaluationResult:
        """Compute strict entity F1 after remapping gold to internal groups."""

        dataset = data_loader.dataset
        gold_entities = getattr(dataset, "gold_entities", None)
        if gold_entities is None:
            raise ValueError("Span dataset must expose gold_entities")
        hierarchy = build_span_label_hierarchy(self.model.config.superclass_groups)
        coarse_gold = {
            document_id: tuple(
                (
                    hierarchy.fine_to_coarse_labels[entity_type],
                    start,
                    end,
                    text,
                )
                for entity_type, start, end, text in entities
            )
            for document_id, entities in gold_entities.items()
        }
        decoding = self.config.get("decoding", self.config.get("evaluation", {}))
        threshold = float(decoding.get("confidence_threshold", 0.5))
        allow_overlapping = bool(decoding.get("allow_overlapping", False))
        loss, candidates = self._collect_coarse_candidates(
            data_loader, minimum_confidence=threshold
        )
        predictions = {
            document_id: merge_span_predictions(
                tuple(entity for entity in entities if entity.confidence >= threshold),
                allow_overlapping=allow_overlapping,
            )
            for document_id, entities in candidates.items()
        }
        return NerEvaluationResult(
            loss=loss,
            metrics=compute_strict_ner_metrics(predictions, coarse_gold),
            predictions=predictions,
        )

    @staticmethod
    def _loss_value(outputs: Any, name: str) -> float:
        value = getattr(outputs, name, None)
        return 0.0 if value is None else float(value.detach().cpu())

    def fit(self, train_loader: Any, validation_loader: Any) -> NerTrainingSummary:
        """Run independent coarse and fine optimizer/scheduler phases."""

        torch = _require_torch()
        try:
            from transformers import get_linear_schedule_with_warmup
        except ImportError as error:
            raise RuntimeError(
                "transformers is required for the learning-rate scheduler"
            ) from error

        training = self.training_config
        coarse_epochs = int(training.get("coarse_epochs", 5))
        fine_epochs = int(training.get("fine_epochs", 20))
        accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
        if coarse_epochs < 1 or fine_epochs < 1 or accumulation_steps < 1:
            raise ValueError(
                "coarse_epochs, fine_epochs and gradient_accumulation_steps "
                "must be positive"
            )
        fine_min_epochs = int(
            training.get(
                "fine_early_stopping_min_epochs",
                training.get("early_stopping_min_epochs", 0),
            )
        )
        if not 0 <= fine_min_epochs <= fine_epochs:
            raise ValueError(
                "fine_early_stopping_min_epochs must be between 0 and fine_epochs"
            )

        seed = int(self.config.get("experiment", {}).get("seed", 42))
        set_seed(seed)
        device = self._device(torch)
        self.model.to(device)
        amp_enabled = bool(training.get("mixed_precision", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        clip_norm = float(training.get("gradient_clip_norm", 1.0))
        patience = int(training.get("early_stopping_patience", fine_epochs))
        output_dir = self.output_dir.resolve()
        checkpoint_dir = output_dir / "checkpoints" / "best"
        output_dir.mkdir(parents=True, exist_ok=True)

        best_f1 = float("-inf")
        best_global_epoch = 0
        best_fine_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []
        stop_training = False

        stage_plan = (("coarse", coarse_epochs), ("fine", fine_epochs))
        global_epoch = 0
        for stage, stage_epochs in stage_plan:
            self.model.set_training_stage(stage)
            optimizer = self._build_stage_optimizer(torch, stage)
            updates_per_epoch = max(
                1, (len(train_loader) + accumulation_steps - 1) // accumulation_steps
            )
            total_updates = updates_per_epoch * stage_epochs
            warmup_ratio = float(
                training.get(f"{stage}_warmup_ratio", training.get("warmup_ratio", 0.0))
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=round(total_updates * warmup_ratio),
                num_training_steps=total_updates,
            )

            for stage_epoch in range(1, stage_epochs + 1):
                global_epoch += 1
                self.model.set_training_stage(stage)
                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                totals = {
                    "loss": 0.0,
                    "fine_loss": 0.0,
                    "coarse_loss": 0.0,
                    "contrastive_loss": 0.0,
                }
                for batch_index, batch in enumerate(train_loader, start=1):
                    model_inputs = self._model_inputs(
                        batch, device, include_labels=True
                    )
                    final_group_size = len(train_loader) % accumulation_steps
                    divisor = (
                        final_group_size
                        if final_group_size
                        and batch_index > len(train_loader) - final_group_size
                        else accumulation_steps
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        outputs = self.model(**model_inputs)
                        loss = outputs.loss / divisor
                    scaler.scale(loss).backward()
                    totals["loss"] += self._loss_value(outputs, "loss")
                    totals["fine_loss"] += self._loss_value(outputs, "fine_loss")
                    totals["coarse_loss"] += self._loss_value(outputs, "coarse_loss")
                    totals["contrastive_loss"] += self._loss_value(
                        outputs, "contrastive_loss"
                    )

                    should_update = (
                        batch_index % accumulation_steps == 0
                        or batch_index == len(train_loader)
                    )
                    if should_update:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                if stage == "coarse":
                    validation = self.evaluate_coarse(validation_loader)
                    coarse_f1 = validation.metrics.micro_f1
                    fine_f1 = 0.0
                else:
                    self.model.set_training_stage("fine")
                    validation = self.evaluate(validation_loader)
                    coarse_f1 = 0.0
                    fine_f1 = validation.metrics.micro_f1

                batch_count = max(1, len(train_loader))
                history.append(
                    {
                        "epoch": global_epoch,
                        "stage": stage,
                        "stage_epoch": stage_epoch,
                        "train_loss": totals["loss"] / batch_count,
                        "train_fine_loss": totals["fine_loss"] / batch_count,
                        "train_coarse_loss": totals["coarse_loss"] / batch_count,
                        "train_contrastive_loss": (
                            totals["contrastive_loss"] / batch_count
                        ),
                        "validation_loss": validation.loss,
                        "validation_precision": validation.metrics.precision,
                        "validation_recall": validation.metrics.recall,
                        "validation_micro_f1": validation.metrics.micro_f1,
                        "validation_macro_f1": validation.metrics.macro_f1,
                        "validation_coarse_micro_f1": coarse_f1,
                        "validation_fine_micro_f1": fine_f1,
                    }
                )

                if stage == "fine":
                    if fine_f1 > best_f1:
                        best_f1 = fine_f1
                        best_global_epoch = global_epoch
                        best_fine_epoch = stage_epoch
                        epochs_without_improvement = 0
                        self._save_checkpoint(
                            checkpoint_dir, global_epoch, validation
                        )
                    else:
                        epochs_without_improvement += 1
                        if (
                            stage_epoch >= fine_min_epochs
                            and epochs_without_improvement >= patience
                        ):
                            stop_training = True
                            break
            if stop_training:
                break

        with (output_dir / "history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        with (output_dir / "train_metrics.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "training_strategy": "coarse_to_fine_curriculum",
                    "coarse_epochs_completed": sum(
                        row["stage"] == "coarse" for row in history
                    ),
                    "fine_epochs_completed": sum(
                        row["stage"] == "fine" for row in history
                    ),
                    "best_epoch": best_global_epoch,
                    "best_fine_epoch": best_fine_epoch,
                    "best_validation_f1": best_f1,
                    "checkpoint_dir": str(checkpoint_dir),
                    "superclass_groups": self.model.config.superclass_groups,
                    "fine_labels": list(self.model.config.label2id),
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
        return NerTrainingSummary(
            best_epoch=best_global_epoch,
            best_validation_f1=best_f1,
            checkpoint_dir=checkpoint_dir,
            history=tuple(history),
        )


__all__ = ["HierarchicalSpanNerTrainer"]
