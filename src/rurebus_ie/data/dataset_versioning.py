"""Build immutable, traceable RuREBus dataset versions from reviewed corrections."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Sequence

from rurebus_ie.data.brat_parser import load_brat_document, validate_round_trip
from rurebus_ie.data.conflict_resolution import (
    Correction,
    apply_corrections,
    load_dataset,
    read_corrections,
)
from rurebus_ie.data.preprocessing import file_sha256, source_document_id


def corpus_fingerprint(dataset_root: str | Path) -> str:
    """Return a stable digest of split BRAT files, independent of absolute paths."""
    root = Path(dataset_root)
    digest = sha256()
    for path in sorted(
        candidate
        for split in ("train", "validation", "test")
        for candidate in (root / split).glob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".txt", ".ann"}
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def verify_source_integrity(
    source_root: str | Path,
    integrity_manifest: str | Path,
) -> int:
    """Verify source split files against a previously recorded SHA-256 table."""
    source = Path(source_root)
    checked = 0
    with Path(integrity_manifest).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_path", "source_sha256"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Integrity manifest must contain {sorted(required)}")
        for row in reader:
            relative = Path(row["relative_path"])
            if relative.suffix.lower() not in {".txt", ".ann"}:
                continue
            path = source / relative
            if not path.is_file():
                raise FileNotFoundError(f"Source file from integrity manifest is missing: {path}")
            actual = file_sha256(path)
            if actual != row["source_sha256"]:
                raise ValueError(
                    f"Source checksum mismatch for {relative}: "
                    f"expected {row['source_sha256']}, got {actual}"
                )
            checked += 1
    return checked


def _read_base_manifest(source_root: Path) -> tuple[list[str], list[dict[str, str]]]:
    manifest = source_root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Parent dataset manifest not found: {manifest}. "
            "Build corrected versions from rurebus_data/processed."
        )
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"Parent manifest is empty: {manifest}")
    return fields, rows


def _write_version_manifest(
    source_root: Path,
    staged_root: Path,
    final_root: Path,
    *,
    dataset_version: str,
    parent_version: str,
    correction_manifest_sha256: str,
) -> Path:
    base_fields, rows = _read_base_manifest(source_root)
    new_fields = [
        "dataset_version",
        "parent_dataset_version",
        "correction_manifest_sha256",
    ]
    managed_fields = [
        "document_id",
        "source_id",
        "split",
        "processed_txt_path",
        "processed_ann_path",
        "text_sha256",
        "ann_sha256",
        "characters",
        "whitespace_tokens",
        "entity_count",
        "relation_count",
        "entity_types",
        "relation_types",
    ]
    fields = new_fields + [field for field in base_fields if field not in new_fields]
    fields.extend(field for field in managed_fields if field not in fields)
    loader_prefix = final_root.name

    for row in rows:
        split = row["split"]
        document_id = row["document_id"]
        txt_path = staged_root / split / f"{document_id}.txt"
        ann_path = staged_root / split / f"{document_id}.ann"
        document = load_brat_document(txt_path, ann_path)
        validate_round_trip(document)
        entity_counts = Counter(entity.entity_type for entity in document.entities)
        relation_counts = Counter(relation.relation_type for relation in document.relations)
        row.update(
            {
                "dataset_version": dataset_version,
                "parent_dataset_version": parent_version,
                "correction_manifest_sha256": correction_manifest_sha256,
                "source_id": row.get("source_id") or source_document_id(document_id),
                "processed_txt_path": f"{loader_prefix}/{split}/{document_id}.txt",
                "processed_ann_path": f"{loader_prefix}/{split}/{document_id}.ann",
                "text_sha256": file_sha256(txt_path),
                "ann_sha256": file_sha256(ann_path),
                "characters": str(len(document.text)),
                "whitespace_tokens": str(len(document.text.split())),
                "entity_count": str(len(document.entities)),
                "relation_count": str(len(document.relations)),
                "entity_types": json.dumps(
                    dict(sorted(entity_counts.items())), ensure_ascii=False
                ),
                "relation_types": json.dumps(
                    dict(sorted(relation_counts.items())), ensure_ascii=False
                ),
            }
        )

    destination = staged_root / "manifest.csv"
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _split_summary(dataset_root: Path) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        documents = [document for current, document in load_dataset(dataset_root) if current == split]
        entity_counts = sum(
            (Counter(entity.entity_type for entity in document.entities) for document in documents),
            Counter(),
        )
        relation_counts = sum(
            (
                Counter(relation.relation_type for relation in document.relations)
                for document in documents
            ),
            Counter(),
        )
        summary[split] = {
            "documents": len(documents),
            "entities": sum(entity_counts.values()),
            "relations": sum(relation_counts.values()),
            "entity_types": dict(sorted(entity_counts.items())),
            "relation_types": dict(sorted(relation_counts.items())),
        }
    return summary


def validate_versioned_dataset(dataset_root: str | Path) -> dict[str, object]:
    root = Path(dataset_root).expanduser().resolve()
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    documents = load_dataset(root)
    if len(rows) != len(documents):
        raise ValueError(
            f"Manifest/document count mismatch: {len(rows)} rows vs {len(documents)} documents"
        )
    manifest_root = manifest.parent.parent
    for row in rows:
        txt_path = manifest_root / row["processed_txt_path"]
        ann_path = manifest_root / row["processed_ann_path"]
        if file_sha256(txt_path) != row["text_sha256"]:
            raise ValueError(f"TXT checksum mismatch: {txt_path}")
        if file_sha256(ann_path) != row["ann_sha256"]:
            raise ValueError(f"ANN checksum mismatch: {ann_path}")
        load_brat_document(txt_path, ann_path)
    return {
        "documents": len(documents),
        "entities": sum(len(document.entities) for _, document in documents),
        "relations": sum(len(document.relations) for _, document in documents),
        "manifest_rows": len(rows),
        "corpus_fingerprint": corpus_fingerprint(root),
        "checks": "passed",
    }


def build_versioned_dataset(
    source_root: str | Path,
    output_root: str | Path,
    corrections: Sequence[Correction],
    *,
    correction_manifest_path: str | Path,
    dataset_version: str,
    parent_version: str,
    source_integrity_manifest: str | Path | None = None,
    allowed_statuses: Iterable[str] = ("ACCEPTED",),
) -> dict[str, object]:
    """Atomically build an immutable corrected dataset with provenance metadata."""
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    correction_path = Path(correction_manifest_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Dataset version already exists and is immutable: {output}. "
            "Create a new version name instead of overwriting it."
        )
    if source == output or source in output.parents:
        raise ValueError("Output version must not be inside the parent dataset directory")
    if not dataset_version.strip() or not parent_version.strip():
        raise ValueError("dataset_version and parent_version must be non-empty")

    checked_source_files = None
    if source_integrity_manifest is not None:
        checked_source_files = verify_source_integrity(source, source_integrity_manifest)

    allowed = frozenset(allowed_statuses)
    selected = [correction for correction in corrections if correction.decision_status in allowed]
    status_counts = Counter(correction.decision_status for correction in corrections)
    correction_hash = file_sha256(correction_path)
    source_fingerprint = corpus_fingerprint(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    # Keep the final directory name inside staging so manifest-relative paths
    # resolve identically before and after the atomic rename.
    staged = temporary_root / output.name
    try:
        applied = apply_corrections(
            source,
            staged,
            corrections,
            allowed_statuses=allowed,
        )
        if applied != len(selected):
            raise AssertionError(f"Applied {applied} corrections, expected {len(selected)}")
        _write_version_manifest(
            source,
            staged,
            output,
            dataset_version=dataset_version,
            parent_version=parent_version,
            correction_manifest_sha256=correction_hash,
        )
        validation = validate_versioned_dataset(staged)
        output_fingerprint = corpus_fingerprint(staged)
        transition_counts = Counter(
            (correction.old_type, correction.new_type) for correction in selected
        )
        report: dict[str, object] = {
            "schema_version": 2,
            "dataset_version": dataset_version,
            "parent_dataset_version": parent_version,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source),
            "output_root": str(output),
            "source_corpus_fingerprint": source_fingerprint,
            "output_corpus_fingerprint": output_fingerprint,
            "correction_manifest": str(correction_path),
            "correction_manifest_sha256": correction_hash,
            "allowed_statuses": sorted(allowed),
            "decision_status_counts": dict(sorted(status_counts.items())),
            "applied_corrections": applied,
            "applied_by_split": dict(
                sorted(Counter(correction.split for correction in selected).items())
            ),
            "applied_transitions": {
                f"{old}->{new}": count
                for (old, new), count in sorted(transition_counts.items())
            },
            "source_integrity_files_checked": checked_source_files,
            "split": _split_summary(staged),
            "validation": validation,
            "benchmark_policy": {
                "role": "internal_audited",
                "official_test_unchanged": False,
                "comparison_warning": (
                    "Test metrics use corrected gold labels and must not be compared "
                    "directly with official/original test metrics."
                ),
            },
        }
        with (staged / "dataset_report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        # Training registration currently expects this conventional filename.
        shutil.copy2(staged / "dataset_report.json", staged / "preprocessing_report.json")
        staged.replace(output)
        temporary_root.rmdir()
        return report
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or validate a RuREBus dataset version")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-root", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--corrections", required=True)
    build.add_argument("--source-integrity-manifest")
    build.add_argument("--dataset-version", required=True)
    build.add_argument("--parent-version", required=True)
    build.add_argument("--allowed-status", action="append", default=["ACCEPTED"])
    validate = subparsers.add_parser("validate")
    validate.add_argument("dataset_root")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.command == "validate":
        result = validate_versioned_dataset(args.dataset_root)
    else:
        correction_path = Path(args.corrections)
        result = build_versioned_dataset(
            args.source_root,
            args.output_root,
            read_corrections(correction_path),
            correction_manifest_path=correction_path,
            dataset_version=args.dataset_version,
            parent_version=args.parent_version,
            source_integrity_manifest=args.source_integrity_manifest,
            allowed_statuses=args.allowed_status,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "build_versioned_dataset",
    "corpus_fingerprint",
    "validate_versioned_dataset",
    "verify_source_integrity",
]
