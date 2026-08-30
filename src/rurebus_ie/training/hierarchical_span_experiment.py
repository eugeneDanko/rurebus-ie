"""Configured workflows for two-stage hierarchical Span NER."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rurebus_ie.configuration import load_experiment_bundle
from rurebus_ie.models.hierarchical_span_ner import (
    HierarchicalSpanNerModel,
    build_hierarchical_rubert_span_classifier,
)
from rurebus_ie.models.ner_baseline import build_rubert_tokenizer
from rurebus_ie.training.artifacts import assert_output_is_unlocked
from rurebus_ie.training.hierarchical_span_trainer import (
    HierarchicalSpanNerTrainer,
)
from rurebus_ie.training.ner_experiment import (
    _checkpoint_path,
    _validate_configured_dataset,
    _write_config_snapshot,
    _write_evaluation,
)
from rurebus_ie.training.ner_trainer import NerEvaluationResult, NerTrainingSummary
from rurebus_ie.training.span_experiment import (
    _apply_class_thresholds_override,
    _apply_output_override,
    _apply_threshold_override,
    build_span_data_loader,
)
from rurebus_ie.training.span_trainer import SpanThresholdCalibration


def _build_model(bundle: dict[str, Any]) -> HierarchicalSpanNerModel:
    model = bundle["model_config"]["model"]
    spans = bundle["model_config"].get("spans", {})
    hierarchy = bundle["model_config"].get("hierarchy", {})
    return build_hierarchical_rubert_span_classifier(
        model["pretrained_name"],
        max_span_width=int(spans.get("max_span_width", 32)),
        width_embedding_dim=int(model.get("width_embedding_dim", 32)),
        span_hidden_size=int(model.get("span_hidden_size", 512)),
        dropout=float(model.get("dropout", 0.1)),
        none_loss_weight=float(model.get("none_loss_weight", 1.0)),
        superclass_groups=hierarchy.get("superclass_groups"),
        contrastive_projection_size=int(
            hierarchy.get("contrastive_projection_size", 128)
        ),
        contrastive_temperature=float(
            hierarchy.get("contrastive_temperature", 0.1)
        ),
        contrastive_weight=float(hierarchy.get("contrastive_weight", 0.1)),
        coarse_aux_weight=float(hierarchy.get("coarse_aux_weight", 0.3)),
        contrastive_max_spans=int(hierarchy.get("contrastive_max_spans", 256)),
    )


def train_hierarchical_span_ner_experiment(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    output_dir_override: str | Path | None = None,
) -> NerTrainingSummary:
    """Build and train the coarse-to-fine Span NER curriculum."""

    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    _apply_output_override(bundle, output_dir_override)
    _validate_configured_dataset(bundle)
    output_dir = Path(bundle["experiment"]["output_dir"])
    assert_output_is_unlocked(output_dir)
    pretrained_name = bundle["model_config"]["model"]["pretrained_name"]
    tokenizer = build_rubert_tokenizer(pretrained_name)
    train_loader = build_span_data_loader(bundle, "train", tokenizer, shuffle=True)
    validation_loader = build_span_data_loader(
        bundle, "validation", tokenizer, shuffle=False
    )
    unrepresentable = train_loader.dataset.unrepresentable_entities
    if unrepresentable:
        print(
            f"Span alignment warning: {len(unrepresentable)} train entities exceed "
            "the configured width or do not match tokenizer boundaries."
        )
    model = _build_model(bundle)
    _write_config_snapshot(bundle, output_dir)
    return HierarchicalSpanNerTrainer(
        model, bundle, tokenizer=tokenizer
    ).fit(train_loader, validation_loader)


def _checkpoint_evaluator(
    bundle: dict[str, Any],
    split_key: str,
    checkpoint_dir: str | Path | None,
) -> tuple[HierarchicalSpanNerTrainer, Any]:
    _validate_configured_dataset(bundle)
    checkpoint = _checkpoint_path(bundle, checkpoint_dir)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required to load Span NER") from error
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, use_fast=True, fix_mistral_regex=True
    )
    model = HierarchicalSpanNerModel.from_pretrained(checkpoint)
    model.set_training_stage("fine")
    loader = build_span_data_loader(bundle, split_key, tokenizer, shuffle=False)
    return HierarchicalSpanNerTrainer(
        model, bundle, tokenizer=tokenizer
    ), loader


def evaluate_hierarchical_span_ner_experiment(
    experiment_config_path: str | Path,
    *,
    split_key: str,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    confidence_threshold_override: float | None = None,
    class_thresholds_override: dict[str, float] | None = None,
    artifact_prefix: str | None = None,
) -> NerEvaluationResult:
    """Evaluate original RuREBus labels from the fine prediction head."""

    if split_key not in {"validation", "test"}:
        raise ValueError("Only validation or test evaluation is allowed")
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    _apply_output_override(bundle, output_dir_override)
    _apply_threshold_override(bundle, confidence_threshold_override)
    _apply_class_thresholds_override(bundle, class_thresholds_override)
    trainer, loader = _checkpoint_evaluator(bundle, split_key, checkpoint_dir)
    result = trainer.evaluate(loader)
    output_dir = Path(bundle["experiment"]["output_dir"])
    _write_evaluation(
        output_dir, split_key, result, artifact_prefix=artifact_prefix
    )
    decoding_name = (
        f"{artifact_prefix}_decoding_config.json"
        if artifact_prefix
        else f"{split_key}_decoding_config.json"
    )
    with (output_dir / decoding_name).open("w", encoding="utf-8") as stream:
        json.dump(bundle.get("decoding", {}), stream, ensure_ascii=False, indent=2)
    return result


def calibrate_hierarchical_span_threshold_experiment(
    experiment_config_path: str | Path,
    *,
    thresholds: list[float] | tuple[float, ...],
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    output_dir_override: str | Path | None = None,
) -> SpanThresholdCalibration:
    """Select a global threshold for the fine head using validation only.

    The encoder is evaluated once at the smallest requested threshold. All
    other thresholds are scored from the retained span candidates.
    """

    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    _apply_output_override(bundle, output_dir_override)
    trainer, loader = _checkpoint_evaluator(bundle, "validation", checkpoint_dir)
    calibration = trainer.calibrate_thresholds(loader, thresholds)
    output_dir = Path(bundle["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "threshold_calibration.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration.rows[0]))
        writer.writeheader()
        writer.writerows(calibration.rows)
    with (output_dir / "threshold_calibration.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "selection_split": "validation",
                "selection_metric": "strict_entity_micro_f1",
                "model_architecture": "hierarchical_rubert_span_classification",
                "prediction_head": "fine",
                "best_threshold": calibration.best_threshold,
                "best_metrics": calibration.best_metrics.to_dict(),
                "threshold_count": len(calibration.rows),
                "checkpoint_dir": str(_checkpoint_path(bundle, checkpoint_dir)),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return calibration


def test_hierarchical_span_ner_experiment(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    confidence_threshold_override: float | None = None,
    class_thresholds_override: dict[str, float] | None = None,
    artifact_prefix: str | None = None,
) -> NerEvaluationResult:
    return evaluate_hierarchical_span_ner_experiment(
        experiment_config_path,
        split_key="test",
        project_root=project_root,
        checkpoint_dir=checkpoint_dir,
        output_dir_override=output_dir_override,
        confidence_threshold_override=confidence_threshold_override,
        class_thresholds_override=class_thresholds_override,
        artifact_prefix=artifact_prefix,
    )


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical RuREBus Span NER")
    parser.add_argument("experiment_config")
    parser.add_argument("--project-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = train_hierarchical_span_ner_experiment(
        args.experiment_config,
        project_root=args.project_root,
        output_dir_override=args.output_dir,
    )
    print(
        json.dumps(
            {
                "best_epoch": summary.best_epoch,
                "best_validation_f1": summary.best_validation_f1,
                "checkpoint_dir": str(summary.checkpoint_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def test_main() -> None:
    parser = argparse.ArgumentParser(description="Test hierarchical RuREBus Span NER")
    parser.add_argument("experiment_config")
    parser.add_argument("--project-root")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--confidence-threshold", type=float)
    args = parser.parse_args()
    result = test_hierarchical_span_ner_experiment(
        args.experiment_config,
        project_root=args.project_root,
        checkpoint_dir=args.checkpoint_dir,
        output_dir_override=args.output_dir,
        confidence_threshold_override=args.confidence_threshold,
    )
    print(
        json.dumps(
            {"test_loss": result.loss, **result.metrics.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )


__all__ = [
    "calibrate_hierarchical_span_threshold_experiment",
    "evaluate_hierarchical_span_ner_experiment",
    "test_hierarchical_span_ner_experiment",
    "train_hierarchical_span_ner_experiment",
]
