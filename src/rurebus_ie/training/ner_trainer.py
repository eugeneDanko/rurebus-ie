"""Training, validation and checkpointing for the RuBERT NER baseline."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Mapping

from rurebus_ie.evaluation.ner_metrics import NerMetrics, compute_strict_ner_metrics
from rurebus_ie.inference.ner_pipeline import (
    PredictedEntity,
    decode_bio_predictions,
    merge_window_predictions,
)


@dataclass(frozen=True)
class NerEvaluationResult:
    loss: float
    metrics: NerMetrics
    predictions: Mapping[str, tuple[PredictedEntity, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class NerTrainingSummary:
    best_epoch: int
    best_validation_f1: float
    checkpoint_dir: Path
    history: tuple[dict[str, float], ...] = ()


def _require_torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "A working PyTorch installation is required. In Colab, select a GPU "
            "runtime and reinstall the project before training."
        ) from error
    return torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy (when installed) and PyTorch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NerTrainer:
    """Own the training loop, validation loop and best-checkpoint selection."""

    def __init__(
        self,
        model: Any,
        config: dict[str, Any],
        *,
        tokenizer: Any | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self._device_name = device

    @property
    def training_config(self) -> dict[str, Any]:
        return self.config.get("training", self.config)

    @property
    def evaluation_config(self) -> dict[str, Any]:
        return self.config.get("evaluation", {})

    @property
    def output_dir(self) -> Path:
        experiment = self.config.get("experiment", {})
        return Path(experiment.get("output_dir", self.config.get("output_dir", "results/ner")))

    def _device(self, torch: Any) -> Any:
        return torch.device(
            self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _build_optimizer(self, torch: Any) -> Any:
        """Create the optimizer; specialized trainers may override parameter groups."""
        training = self.training_config
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training.get("learning_rate", 2e-5)),
            weight_decay=float(training.get("weight_decay", 0.01)),
        )

    @staticmethod
    def _model_inputs(batch: Mapping[str, Any], device: Any, *, include_labels: bool) -> dict:
        keys = {"input_ids", "attention_mask", "token_type_ids"}
        if include_labels:
            keys.add("labels")
        return {key: value.to(device) for key, value in batch.items() if key in keys}

    def evaluate(self, data_loader: Any) -> NerEvaluationResult:
        """Evaluate and merge all windows back to document entities."""
        torch = _require_torch()
        device = self._device(torch)
        self.model.to(device)
        self.model.eval()
        dataset = data_loader.dataset
        document_texts = getattr(dataset, "document_texts", None)
        gold_entities = getattr(dataset, "gold_entities", None)
        if document_texts is None or gold_entities is None:
            raise ValueError("Validation dataset must expose document_texts and gold_entities")

        id2label = {
            int(key): value
            for key, value in getattr(self.model.config, "id2label", {}).items()
        }
        predictions: dict[str, list[PredictedEntity]] = {
            document_id: [] for document_id in document_texts
        }
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(
                    **self._model_inputs(batch, device, include_labels=True)
                )
                total_loss += float(outputs.loss.detach().cpu())
                batch_count += 1
                probabilities = torch.softmax(outputs.logits, dim=-1)
                confidence, label_ids = probabilities.max(dim=-1)
                for index, document_id in enumerate(batch["document_id"]):
                    predictions[document_id].extend(
                        decode_bio_predictions(
                            label_ids[index].detach().cpu().tolist(),
                            batch["offset_mapping"][index],
                            document_texts[document_id],
                            id2label=id2label,
                            confidences=confidence[index].detach().cpu().tolist(),
                        )
                    )

        merged = {
            document_id: merge_window_predictions(entities)
            for document_id, entities in predictions.items()
        }
        metrics = compute_strict_ner_metrics(merged, gold_entities)
        return NerEvaluationResult(
            loss=total_loss / batch_count if batch_count else 0.0,
            metrics=metrics,
            predictions=merged,
        )

    def _save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        evaluation: NerEvaluationResult,
    ) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
        payload = {
            "epoch": epoch,
            "validation_loss": evaluation.loss,
            **evaluation.metrics.to_dict(),
        }
        with (checkpoint_dir / "validation_metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        with (self.output_dir / "validation_metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)

    def fit(self, train_loader: Any, validation_loader: Any) -> NerTrainingSummary:
        torch = _require_torch()
        try:
            from transformers import get_linear_schedule_with_warmup
        except ImportError as error:
            raise RuntimeError("transformers is required for the learning-rate scheduler") from error

        training = self.training_config
        seed = int(self.config.get("experiment", {}).get("seed", training.get("seed", 42)))
        set_seed(seed)
        device = self._device(torch)
        self.model.to(device)

        epochs = int(training.get("epochs", 3))
        accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
        if epochs < 1 or accumulation_steps < 1:
            raise ValueError("epochs and gradient_accumulation_steps must be positive")
        optimizer = self._build_optimizer(torch)
        updates_per_epoch = max(1, (len(train_loader) + accumulation_steps - 1) // accumulation_steps)
        total_updates = updates_per_epoch * epochs
        warmup_steps = round(total_updates * float(training.get("warmup_ratio", 0.0)))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )

        amp_enabled = bool(training.get("mixed_precision", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        clip_norm = float(training.get("gradient_clip_norm", 1.0))
        patience = int(training.get("early_stopping_patience", epochs))
        min_epochs = int(training.get("early_stopping_min_epochs", 0))
        if not 0 <= min_epochs <= epochs:
            raise ValueError("early_stopping_min_epochs must satisfy 0 <= value <= epochs")
        output_dir = self.output_dir.resolve()
        checkpoint_dir = output_dir / "checkpoints" / "best"
        output_dir.mkdir(parents=True, exist_ok=True)

        best_f1 = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            total_train_loss = 0.0
            for batch_index, batch in enumerate(train_loader, start=1):
                model_inputs = self._model_inputs(batch, device, include_labels=True)
                final_group_size = len(train_loader) % accumulation_steps
                divisor = (
                    final_group_size
                    if final_group_size and batch_index > len(train_loader) - final_group_size
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
                total_train_loss += float(outputs.loss.detach().cpu())

                should_update = (
                    batch_index % accumulation_steps == 0 or batch_index == len(train_loader)
                )
                if should_update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            validation = self.evaluate(validation_loader)
            epoch_row = {
                "epoch": float(epoch),
                "train_loss": total_train_loss / max(1, len(train_loader)),
                "validation_loss": validation.loss,
                "validation_precision": validation.metrics.precision,
                "validation_recall": validation.metrics.recall,
                "validation_micro_f1": validation.metrics.micro_f1,
                "validation_macro_f1": validation.metrics.macro_f1,
            }
            history.append(epoch_row)

            if validation.metrics.micro_f1 > best_f1:
                best_f1 = validation.metrics.micro_f1
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir, epoch, validation)
            else:
                epochs_without_improvement += 1
                if epoch >= min_epochs and epochs_without_improvement >= patience:
                    break

        with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_validation_f1": best_f1,
                    "checkpoint_dir": str(checkpoint_dir),
                    "epochs_completed": len(history),
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
        return NerTrainingSummary(
            best_epoch=best_epoch,
            best_validation_f1=best_f1,
            checkpoint_dir=checkpoint_dir,
            history=tuple(history),
        )


__all__ = [
    "NerEvaluationResult",
    "NerTrainer",
    "NerTrainingSummary",
    "set_seed",
]
