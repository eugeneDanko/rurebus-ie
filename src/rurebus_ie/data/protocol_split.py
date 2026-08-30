"""Build and validate a locked train/validation/global-test protocol.

The original RuREBus test split has already been used by earlier experiments
in this project.  A new global test is therefore selected only from documents
whose *parent* split was train.  The old validation may return to the
development pool, while the old test is retained as ``legacy_test`` for
historical comparisons only.

The protocol is manifest-only: annotated files are not copied, which keeps the
Google Drive footprint small.  Selection happens at ``source_id`` level and is
balanced against entity and relation label counts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_SPLITS = ("train", "validation", "global_test")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_counts(value: str) -> Counter[str]:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object with counts, got {value!r}")
    return Counter({str(key): int(count) for key, count in parsed.items()})


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def _group_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)
    return dict(groups)


def _aggregate(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    materialized = list(rows)
    entities: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    for row in materialized:
        entities.update(_json_counts(row.get("entity_types", "{}")))
        relations.update(_json_counts(row.get("relation_types", "{}")))
    return {
        "documents": len(materialized),
        "source_groups": len({row["source_id"] for row in materialized}),
        "entities": sum(entities.values()),
        "relations": sum(relations.values()),
        "entity_types": dict(sorted(entities.items())),
        "relation_types": dict(sorted(relations.items())),
    }


def _selection_score(
    selected_rows: Sequence[Mapping[str, str]],
    pool_summary: Mapping[str, Any],
    target_fraction: float,
) -> float:
    summary = _aggregate(selected_rows)
    target_documents = pool_summary["documents"] * target_fraction
    score = abs(summary["documents"] - target_documents) / max(1.0, target_documents)
    for field in ("entity_types", "relation_types"):
        relative_errors = []
        for label, total in pool_summary[field].items():
            target = float(total) * target_fraction
            observed = float(summary[field].get(label, 0))
            relative_errors.append(abs(observed - target) / max(1.0, target))
            if observed == 0:
                score += 5.0
        if relative_errors:
            score += sum(relative_errors) / len(relative_errors)
    return score


def _balanced_group_selection(
    groups: Mapping[str, Sequence[dict[str, str]]],
    *,
    pool_summary: Mapping[str, Any],
    target_fraction: float,
    seed: int,
    search_trials: int,
) -> set[str]:
    """Choose whole source groups using reproducible randomized search."""
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be inside (0, 1)")
    if search_trials < 1:
        raise ValueError("search_trials must be positive")
    target_documents = pool_summary["documents"] * target_fraction
    group_ids = sorted(groups)
    if len(group_ids) < 3:
        raise ValueError("At least three source groups are required for a split")

    best_ids: set[str] | None = None
    best_score = float("inf")
    for trial in range(search_trials):
        order = list(group_ids)
        random.Random(seed + trial * 104729).shuffle(order)
        selected: list[str] = []
        document_count = 0
        for group_id in order:
            group_size = len(groups[group_id])
            before = abs(document_count - target_documents)
            after = abs(document_count + group_size - target_documents)
            if after <= before or document_count < target_documents * 0.85:
                selected.append(group_id)
                document_count += group_size
        selected_rows = [row for group_id in selected for row in groups[group_id]]
        score = _selection_score(selected_rows, pool_summary, target_fraction)
        candidate = set(selected)
        if score < best_score or (
            score == best_score and sorted(candidate) < sorted(best_ids or set())
        ):
            best_score = score
            best_ids = candidate
    if not best_ids:
        raise RuntimeError("Failed to select a non-empty group split")
    return best_ids


def _assert_disjoint(rows: Sequence[Mapping[str, str]]) -> None:
    active = [row for row in rows if row["split"] in PROTOCOL_SPLITS]
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in active:
        group_splits[row["source_id"]].add(row["split"])
        hash_splits[row["text_sha256"]].add(row["split"])
    leaking_groups = {key: value for key, value in group_splits.items() if len(value) > 1}
    leaking_hashes = {key: value for key, value in hash_splits.items() if len(value) > 1}
    if leaking_groups:
        raise ValueError(f"source_id leakage between splits: {leaking_groups}")
    if leaking_hashes:
        raise ValueError(f"duplicate text leakage between splits: {leaking_hashes}")


def build_global_test_protocol(
    source_manifest: str | Path,
    output_manifest: str | Path,
    output_report: str | Path,
    *,
    protocol_version: str = "global_v1",
    global_test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    seed: int = 20260826,
    search_trials: int = 5000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a locked manifest without copying or modifying BRAT files."""
    source = Path(source_manifest).expanduser().resolve()
    destination = Path(output_manifest).expanduser().resolve()
    report_path = Path(output_report).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not overwrite and (destination.exists() or report_path.exists()):
        raise FileExistsError(
            "The protocol is already reserved. Validate it instead of overwriting it: "
            f"{destination}"
        )
    if global_test_fraction + validation_fraction >= 1.0:
        raise ValueError("global_test_fraction + validation_fraction must be below 1")

    fieldnames, source_rows = _read_manifest(source)
    development_rows = [row for row in source_rows if row["split"] in {"train", "validation"}]
    legacy_rows = [row for row in source_rows if row["split"] == "test"]
    if not development_rows or not legacy_rows:
        raise ValueError("Source manifest must contain train, validation and test rows")
    pool_summary = _aggregate(development_rows)

    # Previously tuned validation rows are deliberately ineligible for the new
    # global test.  They may be used again only inside train/validation.
    test_candidates = [row for row in development_rows if row["split"] == "train"]
    test_groups = _group_rows(test_candidates)
    global_ids = _balanced_group_selection(
        test_groups,
        pool_summary=pool_summary,
        target_fraction=global_test_fraction,
        seed=seed,
        search_trials=search_trials,
    )

    remaining = [row for row in development_rows if row["source_id"] not in global_ids]
    validation_groups = _group_rows(remaining)
    validation_ids = _balanced_group_selection(
        validation_groups,
        pool_summary=pool_summary,
        target_fraction=validation_fraction,
        seed=seed + 1,
        search_trials=search_trials,
    )

    output_rows: list[dict[str, str]] = []
    for source_row in development_rows:
        row = dict(source_row)
        row["parent_split"] = source_row["split"]
        row["protocol_version"] = protocol_version
        if row["source_id"] in global_ids:
            row["split"] = "global_test"
        elif row["source_id"] in validation_ids:
            row["split"] = "validation"
        else:
            row["split"] = "train"
        output_rows.append(row)
    for source_row in legacy_rows:
        row = dict(source_row)
        row["parent_split"] = source_row["split"]
        row["protocol_version"] = protocol_version
        row["split"] = "legacy_test"
        output_rows.append(row)

    _assert_disjoint(output_rows)
    active_rows = [row for row in output_rows if row["split"] in PROTOCOL_SPLITS]
    for split in PROTOCOL_SPLITS:
        summary = _aggregate(row for row in active_rows if row["split"] == split)
        if summary["documents"] == 0:
            raise ValueError(f"Protocol split {split!r} is empty")
        missing_entities = set(pool_summary["entity_types"]) - set(summary["entity_types"])
        missing_relations = set(pool_summary["relation_types"]) - set(summary["relation_types"])
        if missing_entities:
            raise ValueError(
                f"Protocol split {split!r} misses entity types {sorted(missing_entities)}"
            )
        if missing_relations:
            raise ValueError(
                f"Protocol split {split!r} misses relation types {sorted(missing_relations)}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    output_fieldnames = [*fieldnames]
    for extra in ("parent_split", "protocol_version"):
        if extra not in output_fieldnames:
            output_fieldnames.append(extra)
    split_order = {"train": 0, "validation": 1, "global_test": 2, "legacy_test": 3}
    output_rows.sort(key=lambda row: (split_order[row["split"]], row["document_id"]))
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": protocol_version,
        "source_manifest": str(source),
        "source_manifest_sha256": _file_sha256(source),
        "manifest": str(destination),
        "manifest_sha256": _file_sha256(destination),
        "seed": seed,
        "search_trials": search_trials,
        "requested_fractions": {
            "train": 1.0 - global_test_fraction - validation_fraction,
            "validation": validation_fraction,
            "global_test": global_test_fraction,
        },
        "split": {
            split: _aggregate(row for row in output_rows if row["split"] == split)
            for split in (*PROTOCOL_SPLITS, "legacy_test")
        },
        "safeguards": {
            "group_key": "source_id",
            "source_groups_disjoint": True,
            "text_sha256_disjoint": True,
            "global_test_parent_split": "train_only",
            "previous_validation_excluded_from_global_test": True,
            "legacy_test_excluded_from_model_selection": True,
            "manifest_only_no_data_copy": True,
        },
    }
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def validate_global_test_protocol(
    manifest_path: str | Path, report_path: str | Path
) -> dict[str, Any]:
    manifest = Path(manifest_path).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    if not manifest.is_file() or not report_file.is_file():
        raise FileNotFoundError(f"Protocol manifest/report missing: {manifest}, {report_file}")
    with report_file.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if _file_sha256(manifest) != report.get("manifest_sha256"):
        raise ValueError("Locked protocol manifest differs from its recorded SHA-256")
    _, rows = _read_manifest(manifest)
    _assert_disjoint(rows)
    global_rows = [row for row in rows if row["split"] == "global_test"]
    if not global_rows or any(row.get("parent_split") != "train" for row in global_rows):
        raise ValueError("global_test must be non-empty and originate only from parent train")
    legacy_rows = [row for row in rows if row["split"] == "legacy_test"]
    if any(row.get("parent_split") != "test" for row in legacy_rows):
        raise ValueError("legacy_test contains a row not originating from parent test")
    observed = {
        split: _aggregate(row for row in rows if row["split"] == split)
        for split in (*PROTOCOL_SPLITS, "legacy_test")
    }
    if observed != report.get("split"):
        raise ValueError("Protocol split statistics differ from the locked report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or validate a locked data protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("source_manifest")
    build.add_argument("output_manifest")
    build.add_argument("output_report")
    build.add_argument("--protocol-version", default="global_v1")
    build.add_argument("--global-test-fraction", type=float, default=0.15)
    build.add_argument("--validation-fraction", type=float, default=0.15)
    build.add_argument("--seed", type=int, default=20260826)
    build.add_argument("--search-trials", type=int, default=5000)
    build.add_argument("--overwrite", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest")
    validate.add_argument("report")
    args = parser.parse_args()
    if args.command == "build":
        result = build_global_test_protocol(
            args.source_manifest,
            args.output_manifest,
            args.output_report,
            protocol_version=args.protocol_version,
            global_test_fraction=args.global_test_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            search_trials=args.search_trials,
            overwrite=args.overwrite,
        )
    else:
        result = validate_global_test_protocol(args.manifest, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


__all__ = [
    "PROTOCOL_SPLITS",
    "build_global_test_protocol",
    "validate_global_test_protocol",
]


if __name__ == "__main__":
    main()
