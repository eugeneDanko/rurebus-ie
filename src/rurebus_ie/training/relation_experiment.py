"""Configured R-BERT training, calibration, component and pipeline testing."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from rurebus_ie.configuration import load_experiment_bundle
from rurebus_ie.data.ner_dataset import load_documents_from_manifest
from rurebus_ie.data.relation_collator import RelationClassificationCollator
from rurebus_ie.data.relation_dataset import (
    E1_START,
    E2_START,
    ENTITY_MARKERS,
    RuReBusRelationDataset,
    build_relation_dataset_from_manifest,
    build_relation_examples,
)
from rurebus_ie.evaluation.relation_metrics import (
    RelationMetrics,
    compute_end_to_end_relation_metrics,
)
from rurebus_ie.inference.relation_pipeline import (
    PredictedRelation,
    load_ner_prediction_documents,
)
from rurebus_ie.models.ner_baseline import build_rubert_tokenizer
from rurebus_ie.models.relation_classifier import (
    RelationClassifierModel,
    build_rubert_relation_classifier,
)
from rurebus_ie.training.artifacts import assert_output_is_unlocked
from rurebus_ie.training.ner_experiment import (
    _checkpoint_path,
    _manifest_path,
    _require_torch_data_loader,
    _validate_configured_dataset,
    _write_config_snapshot,
)
from rurebus_ie.training.ner_trainer import NerTrainingSummary
from rurebus_ie.training.relation_trainer import (
    RelationEvaluationResult,
    RelationThresholdCalibration,
    RelationTrainer,
)


def _register_markers(tokenizer: Any) -> Any:
    tokenizer.add_special_tokens({"additional_special_tokens": list(ENTITY_MARKERS)})
    marker_ids = tokenizer.convert_tokens_to_ids(list(ENTITY_MARKERS))
    if len(set(marker_ids)) != len(ENTITY_MARKERS):
        raise ValueError("Failed to register distinct R-BERT entity markers")
    return tokenizer


def _relation_options(bundle: dict[str, Any], *, training: bool) -> dict[str, Any]:
    candidates = bundle["model_config"].get("candidate_generation", {})
    return {
        "max_pair_distance": int(candidates.get("max_pair_distance", 128)),
        "context_margin": int(candidates.get("context_margin", 128)),
        "negative_to_positive_ratio": (
            float(candidates.get("train_negative_to_positive_ratio", 4.0))
            if training
            else None
        ),
        "min_negative_pairs": (
            int(candidates.get("train_min_negative_pairs", 0)) if training else 0
        ),
        "seed": int(bundle["experiment"].get("seed", 42)),
    }


def build_relation_data_loader(
    bundle: dict[str, Any],
    split_key: str,
    tokenizer: Any,
    *,
    shuffle: bool,
) -> Any:
    DataLoader = _require_torch_data_loader()
    split_name = bundle["data_config"]["splits"][split_key]
    is_training = split_key == "train"
    dataset = build_relation_dataset_from_manifest(
        _manifest_path(bundle),
        split_name,
        **_relation_options(bundle, training=is_training),
    )
    training_config = bundle["training"]
    batch_size = int(
        training_config.get(
            "train_batch_size" if is_training else f"{split_key}_batch_size",
            training_config.get("validation_batch_size", 16),
        )
    )
    max_length = int(bundle["model_config"].get("tokenization", {}).get("max_length", 256))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=bool(training_config.get("pin_memory", False)),
        collate_fn=RelationClassificationCollator(tokenizer, max_length=max_length),
    )


def _positive_class_weights(
    dataset: RuReBusRelationDataset, *, cap: float
) -> list[float]:
    total = len(dataset)
    if total == 0:
        raise ValueError("Cannot compute class weights for an empty dataset")
    weights = []
    for positive in dataset.positive_label_counts:
        if positive == 0:
            raise ValueError("Every relation label must occur in the training split")
        # Square-root balancing is less volatile than the raw negative/positive
        # ratio for rare RuREBus relations.
        weights.append(min(cap, math.sqrt(max(1.0, (total - positive) / positive))))
    return weights


def _build_model(
    bundle: dict[str, Any], tokenizer: Any, train_dataset: RuReBusRelationDataset
) -> RelationClassifierModel:
    model_config = bundle["model_config"]["model"]
    marker_ids = tokenizer.convert_tokens_to_ids([E1_START, E2_START])
    cap = float(model_config.get("positive_class_weight_cap", 10.0))
    return build_rubert_relation_classifier(
        model_config["pretrained_name"],
        tokenizer_size=len(tokenizer),
        e1_start_token_id=int(marker_ids[0]),
        e2_start_token_id=int(marker_ids[1]),
        relation_hidden_size=int(model_config.get("relation_hidden_size", 512)),
        dropout=float(model_config.get("dropout", 0.1)),
        positive_class_weights=_positive_class_weights(train_dataset, cap=cap),
    )


def _dataset_stats(dataset: RuReBusRelationDataset) -> dict[str, Any]:
    from rurebus_ie.data.relation_labels import RELATION_LABELS

    return {
        "documents": len(dataset.documents),
        "candidate_pairs": len(dataset),
        "positive_pairs": dataset.positive_pair_count,
        "negative_pairs": dataset.negative_pair_count,
        "positive_labels": dict(zip(RELATION_LABELS, dataset.positive_label_counts)),
        "gold_relations": len(set(dataset.gold_relations)),
        "candidate_gold_relations": len(set(dataset.candidate_gold_relations)),
        "candidate_recall": dataset.candidate_recall,
    }


def train_relation_experiment(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    output_dir_override: str | Path | None = None,
) -> NerTrainingSummary:
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    if output_dir_override is not None:
        bundle["experiment"]["output_dir"] = str(Path(output_dir_override).resolve())
    _validate_configured_dataset(bundle)
    output_dir = Path(bundle["experiment"]["output_dir"])
    assert_output_is_unlocked(output_dir)
    tokenizer = _register_markers(
        build_rubert_tokenizer(bundle["model_config"]["model"]["pretrained_name"])
    )
    train_loader = build_relation_data_loader(bundle, "train", tokenizer, shuffle=True)
    validation_loader = build_relation_data_loader(
        bundle, "validation", tokenizer, shuffle=False
    )
    model = _build_model(bundle, tokenizer, train_loader.dataset)
    _write_config_snapshot(bundle, output_dir)
    with (output_dir / "relation_dataset_stats.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "train": _dataset_stats(train_loader.dataset),
                "validation": _dataset_stats(validation_loader.dataset),
                "multilabel_targets": True,
                "negative_label_representation": "all_zero_vector",
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return RelationTrainer(model, bundle, tokenizer=tokenizer).fit(
        train_loader, validation_loader
    )


def _load_relation_checkpoint(
    bundle: dict[str, Any], checkpoint_dir: str | Path | None
) -> tuple[RelationTrainer, Any]:
    _validate_configured_dataset(bundle)
    checkpoint = _checkpoint_path(bundle, checkpoint_dir)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required to load relation checkpoints") from error
    tokenizer = _register_markers(
        AutoTokenizer.from_pretrained(
            checkpoint, use_fast=True, fix_mistral_regex=True
        )
    )
    model = RelationClassifierModel.from_pretrained(checkpoint)
    return RelationTrainer(model, bundle, tokenizer=tokenizer), tokenizer


def _checkpoint_evaluator(
    bundle: dict[str, Any],
    split_key: str,
    checkpoint_dir: str | Path | None,
) -> tuple[RelationTrainer, Any]:
    trainer, tokenizer = _load_relation_checkpoint(bundle, checkpoint_dir)
    loader = build_relation_data_loader(bundle, split_key, tokenizer, shuffle=False)
    return trainer, loader


def _write_predictions(path: Path, predictions: Sequence[PredictedRelation]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for relation in predictions:
            stream.write(
                json.dumps(
                    {
                        "document_id": relation.document_id,
                        "type": relation.relation_type,
                        "arg1": relation.arg1_id,
                        "arg2": relation.arg2_id,
                        "confidence": relation.confidence,
                        "arg1_signature": relation.arg1_signature,
                        "arg2_signature": relation.arg2_signature,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_evaluation(
    output_dir: Path,
    prefix: str,
    result: RelationEvaluationResult,
    *,
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "loss": result.loss,
                **result.metrics.to_dict(),
                "candidate_recall": result.candidate_recall,
                "candidate_pairs": result.candidate_pairs,
                "confidence_threshold": threshold,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    _write_predictions(output_dir / f"{prefix}_predictions.jsonl", result.predictions)


def evaluate_relation_experiment(
    experiment_config_path: str | Path,
    *,
    split_key: str,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    confidence_threshold_override: float | None = None,
    artifact_prefix: str | None = None,
) -> RelationEvaluationResult:
    if split_key not in {"validation", "test"}:
        raise ValueError("Only validation and test relation evaluation is allowed")
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    threshold = float(
        confidence_threshold_override
        if confidence_threshold_override is not None
        else bundle.get("decoding", {}).get("confidence_threshold", 0.5)
    )
    trainer, loader = _checkpoint_evaluator(bundle, split_key, checkpoint_dir)
    result = trainer.evaluate(loader, threshold=threshold)
    prefix = artifact_prefix or ("test" if split_key == "test" else "validation_evaluation")
    _write_evaluation(Path(bundle["experiment"]["output_dir"]), prefix, result, threshold=threshold)
    return result


def calibrate_relation_threshold_experiment(
    experiment_config_path: str | Path,
    *,
    thresholds: Sequence[float],
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> RelationThresholdCalibration:
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    trainer, loader = _checkpoint_evaluator(bundle, "validation", checkpoint_dir)
    calibration = trainer.calibrate_thresholds(loader, tuple(thresholds))
    output_dir = Path(bundle["experiment"]["output_dir"])
    with (output_dir / "relation_threshold_calibration.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration.rows[0]))
        writer.writeheader()
        writer.writerows(calibration.rows)
    with (output_dir / "relation_threshold_calibration.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "selection_split": "validation",
                "selection_metric": "strict_positive_relation_micro_f1",
                "best_threshold": calibration.best_threshold,
                "best_metrics": calibration.best_metrics.to_dict(),
                "candidate_recall": loader.dataset.candidate_recall,
                "threshold_count": len(calibration.rows),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return calibration


def _predicted_entity_loader(
    bundle: dict[str, Any],
    split_key: str,
    checkpoint_dir: str | Path | None,
    ner_predictions_path: str | Path,
) -> tuple[RelationTrainer, Any, tuple[Any, ...], float]:
    trainer, tokenizer = _load_relation_checkpoint(bundle, checkpoint_dir)
    split_name = bundle["data_config"]["splits"][split_key]
    gold_documents = load_documents_from_manifest(_manifest_path(bundle), split_name)
    predicted_documents = load_ner_prediction_documents(
        ner_predictions_path, gold_documents
    )
    max_pair_distance = int(
        bundle["model_config"].get("candidate_generation", {}).get(
            "max_pair_distance", 128
        )
    )
    representable = 0
    gold_count = 0
    for gold_document, predicted_document in zip(gold_documents, predicted_documents):
        predicted_signatures = {
            (entity.entity_type, entity.start, entity.end)
            for entity in predicted_document.entities
        }
        gold_entities = gold_document.entity_by_id
        for relation in gold_document.relations:
            gold_count += 1
            arg1 = gold_entities[relation.arg1]
            arg2 = gold_entities[relation.arg2]
            gap = max(0, max(arg1.start, arg2.start) - min(arg1.end, arg2.end))
            if (
                (arg1.entity_type, arg1.start, arg1.end) in predicted_signatures
                and (arg2.entity_type, arg2.start, arg2.end) in predicted_signatures
                and gap <= max_pair_distance
            ):
                representable += 1
    pipeline_candidate_recall = representable / gold_count if gold_count else 1.0
    predicted_dataset = build_relation_examples(
        predicted_documents,
        **_relation_options(bundle, training=False),
    )
    DataLoader = _require_torch_data_loader()
    loader = DataLoader(
        predicted_dataset,
        batch_size=int(bundle["training"].get(f"{split_key}_batch_size", 16)),
        shuffle=False,
        num_workers=int(bundle["training"].get("num_workers", 0)),
        pin_memory=bool(bundle["training"].get("pin_memory", False)),
        collate_fn=RelationClassificationCollator(
            tokenizer,
            max_length=int(
                bundle["model_config"].get("tokenization", {}).get("max_length", 256)
            ),
        ),
    )
    return trainer, loader, gold_documents, pipeline_candidate_recall


def calibrate_pipeline_relation_threshold_experiment(
    experiment_config_path: str | Path,
    *,
    ner_validation_predictions_path: str | Path,
    thresholds: Sequence[float],
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> RelationThresholdCalibration:
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    trainer, loader, gold_documents, candidate_recall = _predicted_entity_loader(
        bundle,
        "validation",
        checkpoint_dir,
        ner_validation_predictions_path,
    )
    _, collected = trainer._collect_scores(loader, include_labels=False)
    rows: list[dict[str, float]] = []
    scored: list[tuple[float, RelationMetrics]] = []
    for threshold in sorted({float(value) for value in thresholds}):
        predictions = trainer._decode(collected, threshold)
        metrics = compute_end_to_end_relation_metrics(predictions, gold_documents)
        scored.append((threshold, metrics))
        rows.append(
            {
                "threshold": threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "micro_f1": metrics.micro_f1,
                "macro_f1": metrics.macro_f1,
                "predicted_relations": float(len(predictions)),
                "candidate_recall": candidate_recall,
            }
        )
    best_threshold, best_metrics = max(
        scored, key=lambda item: (item[1].micro_f1, item[1].precision, item[0])
    )
    calibration = RelationThresholdCalibration(
        best_threshold, best_metrics, tuple(rows)
    )
    output_dir = Path(bundle["experiment"]["output_dir"])
    with (output_dir / "pipeline_threshold_calibration.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "pipeline_threshold_calibration.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "selection_split": "validation",
                "uses_predicted_ner_entities": True,
                "best_threshold": best_threshold,
                "best_metrics": best_metrics.to_dict(),
                "pipeline_candidate_recall": candidate_recall,
                "threshold_count": len(rows),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return calibration


def test_end_to_end_pipeline_experiment(
    experiment_config_path: str | Path,
    *,
    ner_test_predictions_path: str | Path,
    confidence_threshold: float,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> RelationEvaluationResult:
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    trainer, loader, gold_documents, candidate_recall = _predicted_entity_loader(
        bundle, "test", checkpoint_dir, ner_test_predictions_path
    )
    predictions = trainer.predict(loader, threshold=confidence_threshold)
    result = RelationEvaluationResult(
        loss=0.0,
        metrics=compute_end_to_end_relation_metrics(predictions, gold_documents),
        predictions=predictions,
        candidate_recall=candidate_recall,
        candidate_pairs=len(loader.dataset),
    )
    _write_evaluation(
        Path(bundle["experiment"]["output_dir"]),
        "pipeline_test",
        result,
        threshold=confidence_threshold,
    )
    return result


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train RuREBus R-BERT relations")
    parser.add_argument("experiment_config")
    parser.add_argument("--project-root")
    args = parser.parse_args()
    result = train_relation_experiment(
        args.experiment_config, project_root=args.project_root
    )
    print(json.dumps({"best_epoch": result.best_epoch, "best_validation_f1": result.best_validation_f1}, indent=2))


def test_main() -> None:
    parser = argparse.ArgumentParser(description="Test RuREBus R-BERT relations")
    parser.add_argument("experiment_config")
    parser.add_argument("--project-root")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--confidence-threshold", type=float, required=True)
    args = parser.parse_args()
    result = evaluate_relation_experiment(
        args.experiment_config,
        split_key="test",
        project_root=args.project_root,
        checkpoint_dir=args.checkpoint_dir,
        confidence_threshold_override=args.confidence_threshold,
    )
    print(json.dumps(result.metrics.to_dict(), ensure_ascii=False, indent=2))


__all__ = [
    "build_relation_data_loader",
    "calibrate_pipeline_relation_threshold_experiment",
    "calibrate_relation_threshold_experiment",
    "evaluate_relation_experiment",
    "test_end_to_end_pipeline_experiment",
    "train_relation_experiment",
]


if __name__ == "__main__":
    train_main()
