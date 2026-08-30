"""Loading and resolving project YAML configurations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load project configurations") from error
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {source}")
    return value


def load_experiment_bundle(
    experiment_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load experiment YAML and its referenced data/model configurations."""
    experiment_path = Path(experiment_config_path).expanduser().resolve()
    if project_root is None:
        try:
            root = experiment_path.parents[2]
        except IndexError as error:
            raise ValueError("project_root is required for this config location") from error
    else:
        root = Path(project_root).expanduser().resolve()

    experiment_config = load_yaml(experiment_path)
    experiment = experiment_config.get("experiment", {})
    for required in ("data_config", "model_config", "output_dir"):
        if required not in experiment:
            raise ValueError(f"Missing experiment.{required} in {experiment_path}")

    data_path = root / experiment["data_config"]
    model_path = root / experiment["model_config"]
    bundle = deepcopy(experiment_config)
    bundle["data_config"] = load_yaml(data_path)
    bundle["model_config"] = load_yaml(model_path)
    bundle["experiment"]["output_dir"] = str((root / experiment["output_dir"]).resolve())
    bundle["paths"] = {
        "project_root": str(root),
        "experiment_config": str(experiment_path),
        "data_config": str(data_path.resolve()),
        "model_config": str(model_path.resolve()),
    }
    return bundle


__all__ = ["load_experiment_bundle", "load_yaml"]
