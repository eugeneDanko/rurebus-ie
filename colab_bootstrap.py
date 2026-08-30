"""Reliable package bootstrap for notebooks running in an ephemeral kernel."""

from __future__ import annotations

import importlib
from pathlib import Path
import site
import subprocess
import sys
from types import ModuleType
from typing import Iterable


DEFAULT_REQUIRED_MODULES = (
    "rurebus_ie.data.brat_parser",
    "rurebus_ie.data.conflict_resolution",
    "rurebus_ie.data.dataset_versioning",
    "rurebus_ie.training.ner_experiment",
    "rurebus_ie.training.span_experiment",
    "rurebus_ie.training.hierarchical_span_experiment",
    "rurebus_ie.data.protocol_split",
    "rurebus_ie.training.relation_experiment",
)


def _module_source_path(root: Path, module_name: str) -> Path:
    relative = Path(*module_name.split("."))
    module_file = root / "src" / relative.with_suffix(".py")
    package_file = root / "src" / relative / "__init__.py"
    if module_file.is_file():
        return module_file
    if package_file.is_file():
        return package_file
    # Most required entries are regular modules. Reporting the .py candidate
    # gives an actionable path instead of a misleading nested __init__.py.
    return module_file


def _purge_project_modules() -> None:
    """Remove stale package objects retained by a long-running notebook kernel."""
    names = [
        name
        for name in sys.modules
        if name == "rurebus_ie" or name.startswith("rurebus_ie.")
    ]
    for name in sorted(names, key=lambda value: value.count("."), reverse=True):
        sys.modules.pop(name, None)


def bootstrap_project(
    project_dir: str | Path,
    *,
    install: bool = True,
    required_modules: Iterable[str] = DEFAULT_REQUIRED_MODULES,
) -> ModuleType:
    """Install the project and make its editable package visible immediately.

    Editable installs create a ``.pth`` file. A running notebook kernel does
    not automatically process a newly created ``.pth`` file, so relying only
    on a pip subprocess can still lead to ``ModuleNotFoundError`` until the
    runtime is restarted. Re-processing site directories and registering the
    project's ``src`` directory makes the same kernel ready immediately.
    """
    root = Path(project_dir).expanduser().resolve()
    pyproject = root / "pyproject.toml"
    package_init = root / "src" / "rurebus_ie" / "__init__.py"
    required = tuple(dict.fromkeys(required_modules))
    required_sources = [_module_source_path(root, name) for name in required]
    missing = [
        path
        for path in (pyproject, package_init, *required_sources)
        if not path.is_file()
    ]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "PROJECT_DIR points to an incomplete or outdated project. "
            f"Missing files:\n{details}\nCurrent PROJECT_DIR: {root}"
            "\nCopy the current project code into Google Drive before running notebooks."
        )

    if install:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                str(root),
                "--no-build-isolation",
            ],
            check=True,
        )

    # Process editable-install .pth files without restarting the notebook.
    site_directories = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        site_directories.append(user_site)
    for directory in site_directories:
        if Path(directory).is_dir():
            site.addsitedir(directory)

    # Deterministic fallback for notebook kernels whose site module ignores a
    # newly written editable-install hook. This does not change package imports:
    # callers still use ``import rurebus_ie``, never ``import src...``.
    source_dir = str(root / "src")
    # Move the expected source directory to the first position even if an old
    # editable-install path with the same package name is already present.
    normalized_source = str(Path(source_dir).resolve())
    cleaned_path = []
    for entry in sys.path:
        try:
            if str(Path(entry).resolve()) == normalized_source:
                continue
        except (OSError, TypeError):
            pass
        cleaned_path.append(entry)
    sys.path[:] = [source_dir, *cleaned_path]
    _purge_project_modules()
    importlib.invalidate_caches()

    module = importlib.import_module("rurebus_ie")
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        raise RuntimeError(
            "Imported rurebus_ie from another project: "
            f"{module_path}. Expected a package under {root}. Restart the runtime "
            "once to clear the previously imported module."
        )
    imported_modules = []
    for module_name in required:
        try:
            imported_modules.append(importlib.import_module(module_name))
        except ModuleNotFoundError as error:
            expected = _module_source_path(root, module_name)
            raise ModuleNotFoundError(
                f"Required project module {module_name!r} is unavailable. "
                f"Expected source file: {expected}. PROJECT_DIR: {root}. "
                "Update the project files in Google Drive and rerun this cell."
            ) from error
    wrong_modules = [
        imported
        for imported in imported_modules
        if root not in Path(imported.__file__).resolve().parents
    ]
    if wrong_modules:
        locations = "\n".join(
            f"- {item.__name__}: {Path(item.__file__).resolve()}" for item in wrong_modules
        )
        raise RuntimeError(f"Imported required modules from another project:\n{locations}")
    print(f"rurebus_ie imported successfully: {module_path}")
    print(f"Required modules verified: {len(required)}")
    return module


__all__ = ["DEFAULT_REQUIRED_MODULES", "bootstrap_project"]
