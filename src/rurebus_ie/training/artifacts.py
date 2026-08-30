"""Reproducible baseline registration and immutable run markers."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable


LOCK_FILENAME = ".baseline_locked"
RECORD_FILENAME = "baseline_record.json"


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except ImportError:
        return None
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return value if isinstance(value, dict) else None


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def assert_output_is_unlocked(output_dir: str | Path) -> None:
    """Prevent accidental overwriting of a registered baseline."""
    output = Path(output_dir)
    lock = output / LOCK_FILENAME
    if lock.is_file():
        raise FileExistsError(
            f"Baseline output is locked: {output}. Create a new experiment name/output_dir "
            "instead of overwriting the registered baseline."
        )


def register_baseline_run(
    output_dir: str | Path,
    *,
    alias: str,
    manifest_path: str | Path,
    preprocessing_report_path: str | Path | None = None,
    require_test_artifacts: bool = True,
) -> dict[str, Any]:
    """Hash a completed run, write its record and lock it against overwrites."""
    output = Path(output_dir).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    existing_record_path = output / RECORD_FILENAME
    existing_lock = output / LOCK_FILENAME
    if existing_lock.is_file() and existing_record_path.is_file():
        existing = _read_json(existing_record_path)
        if existing is None or existing.get("alias") != alias:
            raise RuntimeError(f"Output is already registered under another alias: {output}")
        for artifact in existing.get("artifacts", ()):
            artifact_path = output / artifact["path"]
            if not artifact_path.is_file() or file_sha256(artifact_path) != artifact["sha256"]:
                raise RuntimeError(
                    f"Registered baseline artifact changed after locking: {artifact_path}"
                )
        return existing
    checkpoint = output / "checkpoints" / "best"
    required = [
        output / "config.yaml",
        output / "history.csv",
        output / "train_metrics.json",
        output / "validation_metrics.json",
        checkpoint / "config.json",
        manifest,
    ]
    if require_test_artifacts:
        required.extend([output / "test_metrics.json", output / "predictions.jsonl"])
    model_weights = sorted(checkpoint.glob("*.safetensors")) + sorted(
        checkpoint.glob("pytorch_model*.bin")
    )
    if not model_weights:
        required.append(checkpoint / "model.safetensors")
    missing = [path for path in required if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Cannot register incomplete baseline; missing:\n{details}")

    artifact_paths = [path for path in required if output in path.parents]
    artifact_paths.extend(path for path in model_weights if path not in artifact_paths)
    for optional_name in (
        "test_predictions.jsonl",
        "validation_predictions.jsonl",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        candidate = checkpoint / optional_name if optional_name.startswith("tokenizer") else output / optional_name
        if candidate.is_file():
            artifact_paths.append(candidate)
    artifact_paths.extend(output.glob("*_metrics.json"))
    artifact_paths.extend(output.glob("*_predictions.jsonl"))
    artifacts = [_artifact(path, output) for path in sorted(set(artifact_paths))]

    preprocessing_report = (
        Path(preprocessing_report_path).expanduser().resolve()
        if preprocessing_report_path is not None
        else manifest.parent / "preprocessing_report.json"
    )
    dataset = {
        "root": str(manifest.parent),
        "manifest": _artifact(manifest, manifest.parent),
    }
    if preprocessing_report.is_file():
        dataset["preprocessing_report"] = _artifact(
            preprocessing_report, preprocessing_report.parent
        )

    record: dict[str, Any] = {
        "schema_version": 1,
        "alias": alias,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output),
        "immutable": True,
        "configuration": _read_yaml(output / "config.yaml"),
        "metrics": {
            "train": _read_json(output / "train_metrics.json"),
            "validation": _read_json(output / "validation_metrics.json"),
            "test": _read_json(output / "test_metrics.json"),
        },
        "dataset": dataset,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(
                ("rurebus-ie", "torch", "transformers", "tokenizers", "numpy", "PyYAML")
            ),
        },
        "artifacts": artifacts,
    }
    record_path = output / RECORD_FILENAME
    with record_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    record_hash = file_sha256(record_path)
    with (output / LOCK_FILENAME).open("w", encoding="utf-8") as stream:
        stream.write(
            f"alias={alias}\nrecord={RECORD_FILENAME}\nrecord_sha256={record_hash}\n"
        )
    return record


__all__ = [
    "LOCK_FILENAME",
    "RECORD_FILENAME",
    "assert_output_is_unlocked",
    "file_sha256",
    "register_baseline_run",
]
