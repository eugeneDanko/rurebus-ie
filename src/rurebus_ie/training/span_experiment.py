"""Configured train/test workflows for the RuREBus Span NER model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rurebus_ie.configuration import load_experiment_bundle
from rurebus_ie.data.span_collator import SpanClassificationCollator
from rurebus_ie.data.span_dataset import build_span_dataset_from_manifest
from rurebus_ie.evaluation.ner_error_analysis import (
    NerErrorAnalysis,
    analyze_ner_predictions,
    write_ner_error_analysis,
)
from rurebus_ie.models.ner_baseline import build_rubert_tokenizer
from rurebus_ie.models.span_ner import SpanNerModel, build_rubert_span_classifier
from rurebus_ie.training.artifacts import assert_output_is_unlocked
from rurebus_ie.training.ner_experiment import (
    _checkpoint_path,
    _manifest_path,
    _require_torch_data_loader,
    _validate_configured_dataset,
    _write_config_snapshot,
    _write_evaluation,
)
from rurebus_ie.training.ner_trainer import NerEvaluationResult, NerTrainingSummary
from rurebus_ie.training.span_trainer import (
    SpanClassThresholdCalibration,
    SpanNerTrainer,
    SpanThresholdCalibration,
)


def _apply_output_override(bundle: dict[str, Any], output_dir: str | Path | None) -> None:
    if output_dir is not None:
        bundle["experiment"]["output_dir"] = str(Path(output_dir).expanduser().resolve())


def _apply_threshold_override(bundle: dict[str, Any], threshold: float | None) -> None:
    if threshold is None:
        return
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence_threshold_override must be in the [0, 1] interval")
    bundle.setdefault("decoding", {})["confidence_threshold"] = value


def _apply_class_thresholds_override(
    bundle: dict[str, Any], thresholds: dict[str, float] | None
) -> None:
    if thresholds is None:
        return
    entity_types = set(bundle["data_config"]["entities"]["labels"])
    unknown = set(thresholds) - entity_types
    if unknown:
        raise ValueError(f"Unknown class thresholds: {sorted(unknown)}")
    values = {str(label): float(value) for label, value in thresholds.items()}
    if any(not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("Every class threshold must be in the [0, 1] interval")
    bundle.setdefault("decoding", {})["class_thresholds"] = values


def _span_options(bundle: dict[str, Any]) -> dict[str, Any]:
    tokenization = bundle["model_config"].get("tokenization", {})
    spans = bundle["model_config"].get("spans", {})
    return {
        "max_length": int(tokenization.get("max_length", 512)),
        "stride": int(tokenization.get("stride", 128)),
        "max_span_width": int(spans.get("max_span_width", 32)),
        "strict_alignment": bool(spans.get("strict_alignment", False)),
    }


def build_span_data_loader(
    bundle: dict[str, Any],
    split_key: str,
    tokenizer: Any,
    *,
    shuffle: bool,
) -> Any:
    DataLoader = _require_torch_data_loader()
    split_name = bundle["data_config"]["splits"][split_key]
    dataset = build_span_dataset_from_manifest(
        _manifest_path(bundle),
        split_name,
        tokenizer,
        **_span_options(bundle),
    )
    training = bundle["training"]
    spans = bundle["model_config"].get("spans", {})
    is_training = split_key == "train"
    batch_size = int(
        training.get(
            "train_batch_size" if is_training else f"{split_key}_batch_size",
            training.get("validation_batch_size", 1),
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", False)),
        collate_fn=SpanClassificationCollator(
            tokenizer,
            max_span_width=int(spans.get("max_span_width", 32)),
            training=is_training,
            negative_to_positive_ratio=int(spans.get("negative_to_positive_ratio", 20)),
            min_negative_spans=int(spans.get("min_negative_spans", 128)),
            max_negative_spans=int(spans.get("max_negative_spans", 1024)),
            seed=int(bundle["experiment"].get("seed", 42)),
        ),
    )
    return loader


def _build_model(bundle: dict[str, Any]) -> SpanNerModel:
    model = bundle["model_config"]["model"]
    spans = bundle["model_config"].get("spans", {})
    return build_rubert_span_classifier(
        model["pretrained_name"],
        max_span_width=int(spans.get("max_span_width", 32)),
        width_embedding_dim=int(model.get("width_embedding_dim", 32)),
        span_hidden_size=int(model.get("span_hidden_size", 512)),
        dropout=float(model.get("dropout", 0.1)),
        none_loss_weight=float(model.get("none_loss_weight", 1.0)),
    )


def train_span_ner_experiment(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    output_dir_override: str | Path | None = None,
) -> NerTrainingSummary:
    """Build configured components and train Span NER."""
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
    return SpanNerTrainer(model, bundle, tokenizer=tokenizer).fit(
        train_loader, validation_loader
    )


def _checkpoint_evaluator(
    bundle: dict[str, Any],
    split_key: str,
    checkpoint_dir: str | Path | None,
) -> tuple[SpanNerTrainer, Any]:
    _validate_configured_dataset(bundle)
    checkpoint = _checkpoint_path(bundle, checkpoint_dir)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required to load Span NER") from error
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, use_fast=True, fix_mistral_regex=True
    )
    model = SpanNerModel.from_pretrained(checkpoint)
    loader = build_span_data_loader(bundle, split_key, tokenizer, shuffle=False)
    return SpanNerTrainer(model, bundle, tokenizer=tokenizer), loader


def evaluate_span_ner_experiment(
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
        output_dir,
        split_key,
        result,
        artifact_prefix=artifact_prefix,
    )
    decoding_name = (
        f"{artifact_prefix}_decoding_config.json"
        if artifact_prefix
        else f"{split_key}_decoding_config.json"
    )
    with (output_dir / decoding_name).open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(bundle.get("decoding", {}), stream, ensure_ascii=False, indent=2)
    return result


def test_span_ner_experiment(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    confidence_threshold_override: float | None = None,
    class_thresholds_override: dict[str, float] | None = None,
    artifact_prefix: str | None = None,
) -> NerEvaluationResult:
    return evaluate_span_ner_experiment(
        experiment_config_path,
        split_key="test",
        project_root=project_root,
        checkpoint_dir=checkpoint_dir,
        output_dir_override=output_dir_override,
        confidence_threshold_override=confidence_threshold_override,
        class_thresholds_override=class_thresholds_override,
        artifact_prefix=artifact_prefix,
    )


def calibrate_span_threshold_experiment(
    experiment_config_path: str | Path,
    *,
    thresholds: list[float] | tuple[float, ...],
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    output_dir_override: str | Path | None = None,
) -> SpanThresholdCalibration:
    """Choose a validation threshold and persist a small reproducibility report."""
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


def calibrate_span_class_thresholds_experiment(
    experiment_config_path: str | Path,
    *,
    thresholds: list[float] | tuple[float, ...],
    initial_threshold: float,
    max_rounds: int = 2,
    entity_types: list[str] | tuple[str, ...] | None = None,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    output_dir_override: str | Path | None = None,
) -> SpanClassThresholdCalibration:
    """Select class-specific thresholds using validation only."""
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    _apply_output_override(bundle, output_dir_override)
    trainer, loader = _checkpoint_evaluator(bundle, "validation", checkpoint_dir)
    calibration = trainer.calibrate_class_thresholds(
        loader,
        thresholds,
        initial_threshold=initial_threshold,
        max_rounds=max_rounds,
        entity_types=(entity_types if entity_types is not None else tuple(
            bundle["data_config"]["entities"]["labels"]
        )),
    )
    output_dir = Path(bundle["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "class_threshold_calibration.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration.trace[0]))
        writer.writeheader()
        writer.writerows(calibration.trace)
    with (output_dir / "class_threshold_calibration.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "selection_split": "validation",
                "selection_metric": "strict_entity_micro_f1",
                "method": "coordinate_descent",
                "initial_global_threshold": calibration.initial_threshold,
                "class_thresholds": calibration.class_thresholds,
                "best_metrics": calibration.best_metrics.to_dict(),
                "max_rounds": max_rounds,
                "tuned_entity_types": list(
                    entity_types
                    if entity_types is not None
                    else bundle["data_config"]["entities"]["labels"]
                ),
                "trace_steps": len(calibration.trace),
                "checkpoint_dir": str(_checkpoint_path(bundle, checkpoint_dir)),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return calibration


def run_span_validation_error_analysis(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    confidence_threshold: float | None = None,
    class_thresholds: dict[str, float] | None = None,
    output_dir_override: str | Path | None = None,
    edge_token_count: int = 16,
    artifact_name: str = "global_threshold",
) -> NerErrorAnalysis:
    """Evaluate validation and write generic plus Span-compatible error tables."""
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    _apply_output_override(bundle, output_dir_override)
    _apply_threshold_override(bundle, confidence_threshold)
    _apply_class_thresholds_override(bundle, class_thresholds)
    trainer, loader = _checkpoint_evaluator(bundle, "validation", checkpoint_dir)
    result = trainer.evaluate(loader)
    output_dir = Path(bundle["experiment"]["output_dir"])
    _write_evaluation(
        output_dir,
        "validation",
        result,
        artifact_prefix=f"{artifact_name}_validation",
    )
    analysis = analyze_ner_predictions(
        loader.dataset,
        result.predictions,
        edge_token_count=edge_token_count,
    )
    write_ner_error_analysis(
        analysis,
        output_dir / "error_analysis" / "validation" / artifact_name,
    )
    return analysis


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train RuREBus Span NER")
    parser.add_argument("experiment_config")
    parser.add_argument("--project-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = train_span_ner_experiment(
        args.experiment_config,
        project_root=args.project_root,
        output_dir_override=args.output_dir,
    )
    print(json.dumps({
        "best_epoch": summary.best_epoch,
        "best_validation_f1": summary.best_validation_f1,
        "checkpoint_dir": str(summary.checkpoint_dir),
    }, ensure_ascii=False, indent=2))


def test_main() -> None:
    parser = argparse.ArgumentParser(description="Test RuREBus Span NER")
    parser.add_argument("experiment_config")
    parser.add_argument("--project-root")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--confidence-threshold", type=float)
    args = parser.parse_args()
    result = test_span_ner_experiment(
        args.experiment_config,
        project_root=args.project_root,
        checkpoint_dir=args.checkpoint_dir,
        output_dir_override=args.output_dir,
        confidence_threshold_override=args.confidence_threshold,
    )
    print(json.dumps({"test_loss": result.loss, **result.metrics.to_dict()},
                     ensure_ascii=False, indent=2))


__all__ = [
    "build_span_data_loader",
    "calibrate_span_class_thresholds_experiment",
    "calibrate_span_threshold_experiment",
    "evaluate_span_ner_experiment",
    "run_span_validation_error_analysis",
    "test_span_ner_experiment",
    "train_span_ner_experiment",
]
