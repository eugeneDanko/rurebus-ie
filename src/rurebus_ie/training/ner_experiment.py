"""High-level train/test workflows called by the project notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rurebus_ie.configuration import load_experiment_bundle, load_yaml
from rurebus_ie.data.collators import TokenClassificationCollator
from rurebus_ie.data.ner_dataset import build_ner_dataset_from_manifest
from rurebus_ie.data.dataset_versioning import validate_versioned_dataset
from rurebus_ie.data.preprocessing import file_sha256
from rurebus_ie.evaluation.ner_error_analysis import (
    NerErrorAnalysis,
    analyze_ner_predictions,
    write_ner_error_analysis,
)
from rurebus_ie.models.ner_baseline import (
    build_rubert_token_classifier,
    build_rubert_tokenizer,
)
from rurebus_ie.training.ner_trainer import (
    NerEvaluationResult,
    NerTrainer,
    NerTrainingSummary,
)
from rurebus_ie.training.artifacts import (
    assert_output_is_unlocked,
    register_baseline_run,
)


def _require_torch_data_loader() -> Any:
    try:
        from torch.utils.data import DataLoader
    except (ImportError, OSError) as error:
        raise RuntimeError("A working PyTorch installation is required") from error
    return DataLoader


def _manifest_path(bundle: dict[str, Any]) -> Path:
    root = Path(bundle["paths"]["project_root"])
    configured = Path(bundle["data_config"]["dataset"]["manifest_path"])
    return configured if configured.is_absolute() else root / configured


def _configured_path(bundle: dict[str, Any], key: str) -> Path | None:
    configured = bundle["data_config"]["dataset"].get(key)
    if not configured:
        return None
    path = Path(configured)
    return path if path.is_absolute() else Path(bundle["paths"]["project_root"]) / path


def _validate_configured_dataset(bundle: dict[str, Any]) -> None:
    """Fail before model download if a versioned dataset is stale or incomplete."""
    manifest = _manifest_path(bundle)
    if not manifest.is_file():
        raise FileNotFoundError(f"Configured dataset manifest not found: {manifest}")
    dataset_config = bundle["data_config"]["dataset"]
    version = dataset_config.get("version")
    if not version:
        return
    validation = validate_versioned_dataset(manifest.parent)
    report_path = _configured_path(bundle, "dataset_report_path")
    if report_path is None or not report_path.is_file():
        raise FileNotFoundError(f"Dataset report not found: {report_path}")
    with report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("dataset_version") != version:
        raise ValueError(
            f"Dataset version mismatch: config={version!r}, "
            f"report={report.get('dataset_version')!r}"
        )
    if validation["corpus_fingerprint"] != report.get("output_corpus_fingerprint"):
        raise ValueError("Dataset corpus fingerprint differs from dataset_report.json")
    corrections_path = _configured_path(bundle, "corrections_path")
    if corrections_path is not None:
        if not corrections_path.is_file():
            raise FileNotFoundError(corrections_path)
        if file_sha256(corrections_path) != report.get("correction_manifest_sha256"):
            raise ValueError(
                "Correction decisions changed after this immutable dataset version was built. "
                "Create a new dataset version."
            )
    protocol_report_path = _configured_path(bundle, "protocol_report_path")
    if protocol_report_path is not None:
        from rurebus_ie.data.protocol_split import validate_global_test_protocol

        validate_global_test_protocol(manifest, protocol_report_path)


def _tokenization_options(bundle: dict[str, Any]) -> dict[str, Any]:
    config = bundle["model_config"]["tokenization"]
    return {
        "max_length": int(config.get("max_length", 512)),
        "stride": int(config.get("stride", 128)),
        "label_all_subtokens": bool(config.get("label_all_subtokens", True)),
        "ignore_label_id": int(config.get("ignore_label_id", -100)),
    }


def build_ner_data_loader(
    bundle: dict[str, Any],
    split_key: str,
    tokenizer: Any,
    *,
    shuffle: bool,
) -> Any:
    """Build one configured split and its dynamic-padding DataLoader."""
    DataLoader = _require_torch_data_loader()
    split_name = bundle["data_config"]["splits"][split_key]
    dataset = build_ner_dataset_from_manifest(
        _manifest_path(bundle),
        split_name,
        tokenizer,
        **_tokenization_options(bundle),
    )
    training = bundle["training"]
    if split_key == "train":
        batch_size = int(training.get("train_batch_size", 8))
    else:
        batch_size = int(
            training.get(f"{split_key}_batch_size", training.get("validation_batch_size", 8))
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", False)),
        collate_fn=TokenClassificationCollator(tokenizer),
    )


def _write_config_snapshot(bundle: dict[str, Any], output_dir: Path) -> None:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to save the experiment snapshot") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(bundle, stream, allow_unicode=True, sort_keys=False)


def _checkpoint_path(
    bundle: dict[str, Any], checkpoint_dir: str | Path | None
) -> Path:
    output_dir = Path(bundle["experiment"]["output_dir"])
    checkpoint = (
        Path(checkpoint_dir).expanduser().resolve()
        if checkpoint_dir
        else output_dir / "checkpoints" / "best"
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
    return checkpoint


def _checkpoint_evaluator(
    bundle: dict[str, Any],
    split_key: str,
    checkpoint_dir: str | Path | None,
) -> tuple[NerTrainer, Any]:
    _validate_configured_dataset(bundle)
    checkpoint = _checkpoint_path(bundle, checkpoint_dir)
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required to load the checkpoint") from error
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, use_fast=True, fix_mistral_regex=True
    )
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    data_loader = build_ner_data_loader(bundle, split_key, tokenizer, shuffle=False)
    return NerTrainer(model, bundle, tokenizer=tokenizer), data_loader


def _write_predictions(path: Path, result: NerEvaluationResult) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for document_id, entities in sorted(result.predictions.items()):
            stream.write(
                json.dumps(
                    {
                        "document_id": document_id,
                        "entities": [
                            {
                                "text": entity.text,
                                "type": entity.entity_type,
                                "start": entity.start,
                                "end": entity.end,
                                "confidence": entity.confidence,
                            }
                            for entity in entities
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_evaluation(
    output_dir: Path,
    split_key: str,
    result: NerEvaluationResult,
    *,
    artifact_prefix: str | None = None,
) -> None:
    if artifact_prefix:
        metrics_name = f"{artifact_prefix}_metrics.json"
        predictions_name = f"{artifact_prefix}_predictions.jsonl"
    else:
        metrics_name = (
            "test_metrics.json"
            if split_key == "test"
            else f"{split_key}_evaluation_metrics.json"
        )
        predictions_name = f"{split_key}_predictions.jsonl"
    metrics_payload = {f"{split_key}_loss": result.loss, **result.metrics.to_dict()}
    with (output_dir / metrics_name).open("w", encoding="utf-8") as stream:
        json.dump(metrics_payload, stream, ensure_ascii=False, indent=2)
    predictions_path = output_dir / predictions_name
    _write_predictions(predictions_path, result)
    if split_key == "test" and not artifact_prefix:
        # Backward-compatible canonical end-to-end predictions filename.
        _write_predictions(output_dir / "predictions.jsonl", result)


def train_ner_experiment(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> NerTrainingSummary:
    """Build every component from YAML and train the NER baseline."""
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    _validate_configured_dataset(bundle)
    output_dir = Path(bundle["experiment"]["output_dir"])
    assert_output_is_unlocked(output_dir)
    model_config = bundle["model_config"]["model"]
    pretrained_name = model_config["pretrained_name"]
    tokenizer = build_rubert_tokenizer(pretrained_name)
    train_loader = build_ner_data_loader(bundle, "train", tokenizer, shuffle=True)
    validation_loader = build_ner_data_loader(bundle, "validation", tokenizer, shuffle=False)
    model = build_rubert_token_classifier(
        pretrained_name,
        dropout=model_config.get("dropout"),
    )
    _write_config_snapshot(bundle, output_dir)
    trainer = NerTrainer(model, bundle, tokenizer=tokenizer)
    return trainer.fit(train_loader, validation_loader)


def test_ner_experiment(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> NerEvaluationResult:
    """Load the best checkpoint, evaluate test and save metrics/predictions."""
    return evaluate_ner_experiment(
        experiment_config_path,
        split_key="test",
        project_root=project_root,
        checkpoint_dir=checkpoint_dir,
    )


def evaluate_ner_experiment(
    experiment_config_path: str | Path,
    *,
    split_key: str,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    evaluation_data_config_path: str | Path | None = None,
    artifact_prefix: str | None = None,
) -> NerEvaluationResult:
    """Evaluate a configured split and persist its metrics and predictions."""
    if split_key not in {"validation", "test"}:
        raise ValueError("Only validation or test evaluation is allowed")
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    if evaluation_data_config_path is not None:
        data_path = Path(evaluation_data_config_path).expanduser().resolve()
        bundle["data_config"] = load_yaml(data_path)
        bundle["paths"]["evaluation_data_config"] = str(data_path)
    if split_key == "test":
        assert_output_is_unlocked(bundle["experiment"]["output_dir"])
    trainer, data_loader = _checkpoint_evaluator(bundle, split_key, checkpoint_dir)
    result = trainer.evaluate(data_loader)
    _write_evaluation(
        Path(bundle["experiment"]["output_dir"]),
        split_key,
        result,
        artifact_prefix=artifact_prefix,
    )
    return result


def run_validation_error_analysis(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    edge_token_count: int = 16,
) -> NerErrorAnalysis:
    """Re-evaluate the best checkpoint on validation and write error tables."""
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    trainer, data_loader = _checkpoint_evaluator(bundle, "validation", checkpoint_dir)
    result = trainer.evaluate(data_loader)
    output_dir = Path(bundle["experiment"]["output_dir"])
    _write_evaluation(output_dir, "validation", result)
    analysis = analyze_ner_predictions(
        data_loader.dataset,
        result.predictions,
        edge_token_count=edge_token_count,
    )
    write_ner_error_analysis(analysis, output_dir / "error_analysis" / "validation")
    return analysis


def register_ner_baseline(
    experiment_config_path: str | Path,
    *,
    alias: str = "B0",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register the completed NER run and prevent accidental overwrites."""
    bundle = load_experiment_bundle(experiment_config_path, project_root=project_root)
    manifest = _manifest_path(bundle)
    report_path = _configured_path(bundle, "dataset_report_path")
    return register_baseline_run(
        bundle["experiment"]["output_dir"],
        alias=alias,
        manifest_path=manifest,
        preprocessing_report_path=report_path or manifest.parent / "preprocessing_report.json",
        require_test_artifacts=True,
    )


__all__ = [
    "build_ner_data_loader",
    "evaluate_ner_experiment",
    "register_ner_baseline",
    "run_validation_error_analysis",
    "test_ner_experiment",
    "train_ner_experiment",
]


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train the RuREBus RuBERT NER baseline")
    parser.add_argument("experiment_config")
    parser.add_argument("--project-root")
    args = parser.parse_args()
    summary = train_ner_experiment(
        args.experiment_config, project_root=args.project_root
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
    parser = argparse.ArgumentParser(description="Test the best RuREBus NER checkpoint")
    parser.add_argument("experiment_config")
    parser.add_argument("--project-root")
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    result = test_ner_experiment(
        args.experiment_config,
        project_root=args.project_root,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(
        json.dumps(
            {"test_loss": result.loss, **result.metrics.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )


def analyze_main() -> None:
    parser = argparse.ArgumentParser(
        description="Register B0 and analyze NER errors on validation"
    )
    parser.add_argument("experiment_config")
    parser.add_argument("--project-root")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--alias", default="B0")
    parser.add_argument("--edge-token-count", type=int, default=16)
    args = parser.parse_args()
    record = register_ner_baseline(
        args.experiment_config,
        alias=args.alias,
        project_root=args.project_root,
    )
    analysis = run_validation_error_analysis(
        args.experiment_config,
        project_root=args.project_root,
        checkpoint_dir=args.checkpoint_dir,
        edge_token_count=args.edge_token_count,
    )
    print(
        json.dumps(
            {
                "baseline": record["alias"],
                "summary": analysis.summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
