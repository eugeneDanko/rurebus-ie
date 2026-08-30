"""Persist only compact run metadata and the single best checkpoint."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Iterable


DEFAULT_METADATA = (
    "config.yaml",
    "history.csv",
    "train_metrics.json",
    "validation_metrics.json",
    "relation_dataset_stats.json",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persist_best_run(
    local_run_dir: str | Path,
    persistent_run_dir: str | Path,
    *,
    metadata_names: Iterable[str] = DEFAULT_METADATA,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy one best checkpoint from ephemeral storage to persistent storage."""
    source = Path(local_run_dir).expanduser().resolve()
    destination = Path(persistent_run_dir).expanduser().resolve()
    checkpoint = source / "checkpoints" / "best"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Persistent run already exists: {destination}. Use a new run name; "
            "do not silently overwrite trained artifacts."
        )
    destination.mkdir(parents=True, exist_ok=True)
    destination_checkpoint = destination / "checkpoints" / "best"
    if destination_checkpoint.exists():
        if not overwrite:
            raise FileExistsError(destination_checkpoint)
        shutil.rmtree(destination_checkpoint)
    destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint, destination_checkpoint)
    copied: list[Path] = list(destination_checkpoint.rglob("*"))
    for name in metadata_names:
        path = source / name
        if path.is_file():
            target = destination / name
            shutil.copy2(path, target)
            copied.append(target)
    files = [path for path in copied if path.is_file()]
    manifest = {
        "source_was_ephemeral": True,
        "best_checkpoint_only": True,
        "files": [
            {
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(files)
        ],
    }
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"])
    with (destination / "storage_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


__all__ = ["persist_best_run"]
