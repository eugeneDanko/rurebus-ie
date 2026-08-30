"""Build a canonical, deduplicated RuREBus BRAT dataset and split manifest.

This stage is model-independent. It does not create BIO labels, tokenize text,
split documents into model windows, or generate NO_RELATION examples.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
from typing import Iterable

from .brat_parser import (
    BratDocument,
    ENTITY_TYPES,
    RELATION_TYPES,
    load_brat_document,
    read_text_exact,
    validate_round_trip,
)


TRAIN_PART_PRIORITY = {"train_1": 1, "train_2": 2, "train_3": 3}
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
GENERATED_FILENAMES = {
    "manifest.csv",
    "duplicates.csv",
    "preprocessing_report.json",
}


@dataclass(frozen=True)
class SourceRecord:
    group: str
    priority: int
    txt_path: Path
    ann_path: Path
    document: BratDocument
    text_sha256: str
    ann_sha256: str
    source_id: str
    entity_counts: Counter
    relation_counts: Counter

    @property
    def document_id(self) -> str:
        return self.txt_path.stem


def source_document_id(stem: str) -> str:
    """Collapse document fragments such as *_part_0 into one source group."""
    import re

    return re.sub(r"_part_\d+_?$", "", stem)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(path: Path) -> str:
    return sha256(read_text_exact(path).encode("utf-8")).hexdigest()


def locate_data_directory(raw_root: Path, relative_candidates: Iterable[Path]) -> Path:
    for relative_path in relative_candidates:
        candidate = raw_root / relative_path
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    choices = ", ".join(str(raw_root / path) for path in relative_candidates)
    raise FileNotFoundError(f"No extracted BRAT directory found. Checked: {choices}")


def locate_groups(raw_root: Path) -> dict[str, Path]:
    return {
        "train_1": locate_data_directory(
            raw_root,
            [Path("train_data/train_part_1/train_part_1"), Path("train_data/train_part_1")],
        ),
        "train_2": locate_data_directory(
            raw_root,
            [Path("train_data/train_part_2/train_part_2"), Path("train_data/train_part_2")],
        ),
        "train_3": locate_data_directory(
            raw_root,
            [Path("train_data/train_part_3/train_part_3"), Path("train_data/train_part_3")],
        ),
        "test_full": locate_data_directory(
            raw_root,
            [Path("test_data/test_full"), Path("test_data/test_full/test_full")],
        ),
    }


def scan_group(group: str, directory: Path, priority: int) -> list[SourceRecord]:
    txt_files = sorted(directory.glob("*.txt"))
    ann_files = {path.stem: path for path in directory.glob("*.ann")}
    txt_stems = {path.stem for path in txt_files}
    missing_ann = sorted(path.stem for path in txt_files if path.stem not in ann_files)
    missing_txt = sorted(set(ann_files) - txt_stems)
    if missing_ann or missing_txt:
        raise ValueError(
            f"{group}: unmatched BRAT pairs; missing ANN={missing_ann[:5]}, "
            f"missing TXT={missing_txt[:5]}"
        )

    records: list[SourceRecord] = []
    for txt_path in txt_files:
        ann_path = ann_files[txt_path.stem]
        document = load_brat_document(txt_path, ann_path)
        validate_round_trip(document)
        entity_counts = Counter(entity.entity_type for entity in document.entities)
        relation_counts = Counter(relation.relation_type for relation in document.relations)
        records.append(
            SourceRecord(
                group=group,
                priority=priority,
                txt_path=txt_path,
                ann_path=ann_path,
                document=document,
                text_sha256=text_sha256(txt_path),
                ann_sha256=file_sha256(ann_path),
                source_id=source_document_id(txt_path.stem),
                entity_counts=entity_counts,
                relation_counts=relation_counts,
            )
        )
    return records


def select_deduplicated_train(records: list[SourceRecord]):
    by_text_hash: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        by_text_hash[record.text_sha256].append(record)

    selected: list[SourceRecord] = []
    duplicate_rows: list[dict] = []
    for digest, copies in sorted(by_text_hash.items()):
        ordered = sorted(
            copies,
            key=lambda item: (-item.priority, item.group, item.document_id),
        )
        winner = ordered[0]
        selected.append(winner)
        if len(ordered) > 1:
            for record in ordered:
                duplicate_rows.append(
                    {
                        "text_sha256": digest,
                        "document_id": record.document_id,
                        "source_id": record.source_id,
                        "original_part": record.group,
                        "txt_path": str(record.txt_path),
                        "ann_path": str(record.ann_path),
                        "ann_sha256": record.ann_sha256,
                        "entity_count": len(record.document.entities),
                        "relation_count": len(record.document.relations),
                        "selected": record is winner,
                        "annotation_equals_selected": record.ann_sha256 == winner.ann_sha256,
                    }
                )

    selected.sort(key=lambda item: (item.source_id, item.document_id))
    return selected, duplicate_rows


def _counter_distance(actual: Counter, target: Counter) -> float:
    keys = sorted(set(actual) | set(target))
    if not keys:
        return 0.0
    return sum(abs(actual[key] - target[key]) / (target[key] + 1.0) for key in keys) / len(keys)


def stratified_group_split(
    records: list[SourceRecord],
    *,
    validation_size: float,
    seed: int,
    search_trials: int,
) -> tuple[set[str], dict]:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    by_source: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_id].append(record)
    source_ids = sorted(by_source)
    if len(source_ids) < 2:
        raise ValueError("At least two source groups are required for a split")

    validation_group_count = max(1, min(len(source_ids) - 1, round(len(source_ids) * validation_size)))
    source_entity_counts: dict[str, Counter] = {}
    source_relation_counts: dict[str, Counter] = {}
    for source_id, source_records in by_source.items():
        source_entity_counts[source_id] = sum(
            (record.entity_counts for record in source_records), Counter()
        )
        source_relation_counts[source_id] = sum(
            (record.relation_counts for record in source_records), Counter()
        )

    total_entities = sum(source_entity_counts.values(), Counter())
    total_relations = sum(source_relation_counts.values(), Counter())
    target_entities = Counter(
        {key: value * validation_size for key, value in total_entities.items()}
    )
    target_relations = Counter(
        {key: value * validation_size for key, value in total_relations.items()}
    )

    rng = random.Random(seed)
    best_sources: set[str] | None = None
    best_score = float("inf")
    for _ in range(search_trials):
        candidate = set(rng.sample(source_ids, validation_group_count))
        validation_entities = sum(
            (source_entity_counts[source_id] for source_id in candidate), Counter()
        )
        validation_relations = sum(
            (source_relation_counts[source_id] for source_id in candidate), Counter()
        )
        train_relations = total_relations - validation_relations

        relation_error = _counter_distance(validation_relations, target_relations)
        entity_error = _counter_distance(validation_entities, target_entities)
        missing_validation = sum(
            total_relations[label] > 0 and validation_relations[label] == 0
            for label in RELATION_TYPES
        )
        missing_train = sum(
            total_relations[label] > 0 and train_relations[label] == 0
            for label in RELATION_TYPES
        )
        score = relation_error + 0.25 * entity_error + 2.0 * missing_validation + 5.0 * missing_train
        if score < best_score:
            best_score = score
            best_sources = candidate

    if best_sources is None:
        raise RuntimeError("Unable to produce a validation split")
    return best_sources, {
        "score": best_score,
        "validation_source_groups": len(best_sources),
        "total_source_groups": len(source_ids),
    }


def _assert_safe_output(raw_root: Path, output_root: Path) -> None:
    raw_resolved = raw_root.resolve()
    output_resolved = output_root.resolve()
    if output_resolved == raw_resolved or raw_resolved in output_resolved.parents:
        raise ValueError("processed output must not be the raw RuREBus directory or its child")
    if output_resolved.parent == output_resolved:
        raise ValueError("refusing to use a filesystem root as processed output")


def prepare_output(output_root: Path, *, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [output_root / split for split in ("train", "validation", "test")]
    existing.extend(output_root / name for name in GENERATED_FILENAMES)
    present = [path for path in existing if path.exists()]
    if present and not overwrite:
        raise FileExistsError(
            f"Processed output already exists: {present[0]}. Use --overwrite to rebuild it."
        )
    if overwrite:
        for split in ("train", "validation", "test"):
            split_path = (output_root / split).resolve()
            if split_path.parent != output_root.resolve():
                raise ValueError(f"Unsafe split output path: {split_path}")
            if split_path.is_dir():
                shutil.rmtree(split_path)
        for filename in GENERATED_FILENAMES:
            generated_path = output_root / filename
            if generated_path.is_file():
                generated_path.unlink()
    for split in ("train", "validation", "test"):
        (output_root / split).mkdir(parents=True, exist_ok=True)


def relative_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def copy_record(
    record: SourceRecord,
    split: str,
    output_root: Path,
    data_root: Path,
) -> dict:
    destination = output_root / split
    txt_destination = destination / record.txt_path.name
    ann_destination = destination / record.ann_path.name
    if txt_destination.exists() or ann_destination.exists():
        raise FileExistsError(f"Output name collision for {record.document_id}")
    shutil.copy2(record.txt_path, txt_destination)
    shutil.copy2(record.ann_path, ann_destination)

    if file_sha256(txt_destination) != file_sha256(record.txt_path):
        raise IOError(f"TXT checksum mismatch after copying {record.document_id}")
    if file_sha256(ann_destination) != record.ann_sha256:
        raise IOError(f"ANN checksum mismatch after copying {record.document_id}")

    return {
        "document_id": record.document_id,
        "source_id": record.source_id,
        "split": split,
        "original_part": record.group,
        "original_txt_path": relative_path(record.txt_path, data_root),
        "original_ann_path": relative_path(record.ann_path, data_root),
        "processed_txt_path": relative_path(txt_destination, data_root),
        "processed_ann_path": relative_path(ann_destination, data_root),
        "text_sha256": record.text_sha256,
        "ann_sha256": record.ann_sha256,
        "characters": len(record.document.text),
        "whitespace_tokens": len(record.document.text.split()),
        "entity_count": len(record.document.entities),
        "relation_count": len(record.document.relations),
        "entity_types": json.dumps(dict(sorted(record.entity_counts.items())), ensure_ascii=False),
        "relation_types": json.dumps(dict(sorted(record.relation_counts.items())), ensure_ascii=False),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames are required for empty CSV {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def run_preprocessing(args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root).expanduser().resolve()
    raw_root = data_root / "RuREBus"
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else data_root / "processed"
    )
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw RuREBus directory not found: {raw_root}")
    _assert_safe_output(raw_root, output_root)

    groups = locate_groups(raw_root)
    all_train_records: list[SourceRecord] = []
    for group in ("train_1", "train_2", "train_3"):
        all_train_records.extend(
            scan_group(group, groups[group], TRAIN_PART_PRIORITY[group])
        )
    test_records = scan_group("test_full", groups["test_full"], 0)

    selected_train, duplicate_rows = select_deduplicated_train(all_train_records)
    validation_sources, split_details = stratified_group_split(
        selected_train,
        validation_size=args.validation_size,
        seed=args.seed,
        search_trials=args.search_trials,
    )

    train_sources = {record.source_id for record in selected_train} - validation_sources
    if train_sources & validation_sources:
        raise AssertionError("Source-group leakage between train and validation")
    train_hashes = {
        record.text_sha256
        for record in selected_train
        if record.source_id not in validation_sources
    }
    validation_hashes = {
        record.text_sha256
        for record in selected_train
        if record.source_id in validation_sources
    }
    if train_hashes & validation_hashes:
        raise AssertionError("Exact-text leakage between train and validation")
    test_hashes = {record.text_sha256 for record in test_records}
    if (train_hashes | validation_hashes) & test_hashes:
        raise AssertionError("Exact-text leakage from train/validation into test_full")

    prepare_output(output_root, overwrite=args.overwrite)
    manifest_rows: list[dict] = []
    for record in selected_train:
        split = "validation" if record.source_id in validation_sources else "train"
        manifest_rows.append(copy_record(record, split, output_root, data_root))
    for record in test_records:
        manifest_rows.append(copy_record(record, "test", output_root, data_root))
    manifest_rows.sort(key=lambda row: (SPLIT_ORDER[row["split"]], row["source_id"], row["document_id"]))

    manifest_fields = [
        "document_id",
        "source_id",
        "split",
        "original_part",
        "original_txt_path",
        "original_ann_path",
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
    write_csv(output_root / "manifest.csv", manifest_rows, manifest_fields)
    duplicate_fields = [
        "text_sha256",
        "document_id",
        "source_id",
        "original_part",
        "txt_path",
        "ann_path",
        "ann_sha256",
        "entity_count",
        "relation_count",
        "selected",
        "annotation_equals_selected",
    ]
    write_csv(output_root / "duplicates.csv", duplicate_rows, duplicate_fields)

    split_summary = {}
    for split in ("train", "validation", "test"):
        split_records = [row for row in manifest_rows if row["split"] == split]
        split_summary[split] = {
            "documents": len(split_records),
            "source_groups": len({row["source_id"] for row in split_records}),
            "entities": sum(int(row["entity_count"]) for row in split_records),
            "relations": sum(int(row["relation_count"]) for row in split_records),
        }

    selected_entity_counts = sum(
        (record.entity_counts for record in selected_train), Counter()
    )
    selected_relation_counts = sum(
        (record.relation_counts for record in selected_train), Counter()
    )
    report = {
        "schema_version": 1,
        "data_root": str(data_root),
        "raw_root": str(raw_root),
        "processed_root": str(output_root),
        "configuration": {
            "validation_size": args.validation_size,
            "seed": args.seed,
            "search_trials": args.search_trials,
            "duplicate_priority": ["train_3", "train_2", "train_1"],
        },
        "input": {
            "train_documents": len(all_train_records),
            "test_full_documents": len(test_records),
        },
        "deduplication": {
            "selected_train_documents": len(selected_train),
            "duplicate_groups": len({row["text_sha256"] for row in duplicate_rows}),
            "discarded_copies": sum(not row["selected"] for row in duplicate_rows),
            "duplicates_with_annotation_differences": len(
                {
                    row["text_sha256"]
                    for row in duplicate_rows
                    if not row["annotation_equals_selected"]
                }
            ),
        },
        "split": split_summary,
        "split_search": split_details,
        "train_entity_types": counter_to_dict(selected_entity_counts),
        "train_relation_types": counter_to_dict(selected_relation_counts),
        "checks": {
            "brat_validation": "passed",
            "round_trip": "passed",
            "train_validation_source_overlap": 0,
            "train_validation_text_hash_overlap": 0,
            "train_validation_test_text_hash_overlap": 0,
            "copied_file_checksums": "passed",
        },
        "excluded_transformations": [
            "text normalization",
            "BIO/BILOU conversion",
            "model tokenization",
            "windowing",
            "NO_RELATION generation",
            "negative sampling",
        ],
    }
    with (output_root / "preprocessing_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, deduplicate and split the RuREBus BRAT corpus."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to rurebus_data containing the raw RuREBus directory.",
    )
    parser.add_argument(
        "--output-dir",
        help="Processed output directory. Defaults to <data-root>/processed.",
    )
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=5000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild known generated files under processed/.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    report = run_preprocessing(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
