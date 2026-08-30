"""Audit and apply reviewed RuREBus annotation corrections.

The module intentionally separates detection from adjudication.  A surface form
having several labels is only a review candidate: RuREBus labels depend on the
entity's semantic role and relation context, so majority-vote relabelling would
silently corrupt valid examples.
"""

from __future__ import annotations

import csv
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rurebus_ie.data.brat_parser import (
    BratDocument,
    Entity,
    load_brat_document,
    validate_round_trip,
    write_brat_document,
)


SPLITS = ("train", "validation", "test")
BASE_CORRECTION_FIELDS = (
    "split",
    "document_id",
    "entity_id",
    "surface",
    "old_type",
    "new_type",
    "confidence",
    "reason",
    "guideline_basis",
)
REVIEW_FIELDS = (
    "decision_status",
    "reviewer",
    "review_comment",
    "reviewed_at",
)
CORRECTION_FIELDS = BASE_CORRECTION_FIELDS + REVIEW_FIELDS
DECISION_STATUSES = frozenset({"ACCEPTED", "REJECTED", "REVIEW_REQUIRED"})


@dataclass(frozen=True)
class EntityOccurrence:
    split: str
    document_id: str
    entity_id: str
    entity_type: str
    surface: str
    normalized_surface: str
    start: int
    end: int
    context: str


@dataclass(frozen=True)
class Correction:
    split: str
    document_id: str
    entity_id: str
    surface: str
    old_type: str
    new_type: str
    confidence: str
    reason: str
    guideline_basis: str
    decision_status: str = "ACCEPTED"
    reviewer: str = ""
    review_comment: str = ""
    reviewed_at: str = ""


def normalize_surface(text: str) -> str:
    """Normalize only casing and whitespace; do not stem or lemmatize."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _context(text: str, start: int, end: int, radius: int = 120) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def load_dataset(dataset_root: str | Path) -> list[tuple[str, BratDocument]]:
    root = Path(dataset_root)
    documents: list[tuple[str, BratDocument]] = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for txt_path in sorted(split_dir.glob("*.txt")):
            document = load_brat_document(txt_path)
            validate_round_trip(document)
            documents.append((split, document))
    return documents


def collect_occurrences(
    documents: Iterable[tuple[str, BratDocument]],
) -> list[EntityOccurrence]:
    occurrences: list[EntityOccurrence] = []
    for split, document in documents:
        for entity in document.entities:
            occurrences.append(
                EntityOccurrence(
                    split=split,
                    document_id=document.document_id,
                    entity_id=entity.entity_id,
                    entity_type=entity.entity_type,
                    surface=entity.text,
                    normalized_surface=normalize_surface(entity.text),
                    start=entity.start,
                    end=entity.end,
                    context=_context(document.text, entity.start, entity.end),
                )
            )
    return occurrences


def find_surface_conflicts(
    occurrences: Iterable[EntityOccurrence],
) -> dict[str, list[EntityOccurrence]]:
    grouped: dict[str, list[EntityOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[occurrence.normalized_surface].append(occurrence)
    return {
        surface: rows
        for surface, rows in grouped.items()
        if len({row.entity_type for row in rows}) > 1
    }


def write_conflict_registry(
    path: str | Path,
    conflicts: Mapping[str, Sequence[EntityOccurrence]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "normalized_surface",
        "total_occurrences",
        "type_counts",
        "split",
        "document_id",
        "entity_id",
        "entity_type",
        "surface",
        "start",
        "end",
        "context",
    )
    ordered_groups = sorted(
        conflicts.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for normalized_surface, rows in ordered_groups:
            counts = Counter(row.entity_type for row in rows)
            counts_text = ";".join(
                f"{label}:{count}" for label, count in counts.most_common()
            )
            for row in sorted(
                rows,
                key=lambda value: (value.split, value.document_id, value.start),
            ):
                writer.writerow(
                    {
                        "normalized_surface": normalized_surface,
                        "total_occurrences": len(rows),
                        "type_counts": counts_text,
                        "split": row.split,
                        "document_id": row.document_id,
                        "entity_id": row.entity_id,
                        "entity_type": row.entity_type,
                        "surface": row.surface,
                        "start": row.start,
                        "end": row.end,
                        "context": row.context,
                    }
                )


def read_corrections(path: str | Path) -> list[Correction]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(BASE_CORRECTION_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Correction manifest misses columns: {sorted(missing)}")
        corrections = []
        for row in reader:
            values = {field: row[field] for field in BASE_CORRECTION_FIELDS}
            values.update({field: row.get(field, "") for field in REVIEW_FIELDS})
            values["decision_status"] = values["decision_status"] or "ACCEPTED"
            if values["decision_status"] not in DECISION_STATUSES:
                raise ValueError(
                    f"Unsupported decision status {values['decision_status']!r} for "
                    f"{values['document_id']}/{values['entity_id']}"
                )
            corrections.append(Correction(**values))
        return corrections


def write_corrections(path: str | Path, corrections: Sequence[Correction]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CORRECTION_FIELDS)
        writer.writeheader()
        for correction in corrections:
            writer.writerow(
                {field: getattr(correction, field) for field in CORRECTION_FIELDS}
            )


def _apply_document_corrections(
    document: BratDocument,
    corrections: Mapping[str, Correction],
) -> BratDocument:
    entities: list[Entity] = []
    seen: set[str] = set()
    for entity in document.entities:
        correction = corrections.get(entity.entity_id)
        if correction is None:
            entities.append(entity)
            continue
        if entity.entity_type != correction.old_type or entity.text != correction.surface:
            raise ValueError(
                f"Stale correction for {document.document_id}/{entity.entity_id}: "
                f"expected {correction.old_type} {correction.surface!r}, got "
                f"{entity.entity_type} {entity.text!r}"
            )
        entities.append(replace(entity, entity_type=correction.new_type))
        seen.add(entity.entity_id)
    missing = set(corrections) - seen
    if missing:
        raise ValueError(f"Unknown entity ids in {document.document_id}: {sorted(missing)}")
    corrected = replace(document, entities=tuple(entities))
    validate_round_trip(corrected)
    return corrected


def apply_corrections(
    source_root: str | Path,
    output_root: str | Path,
    corrections: Sequence[Correction],
    *,
    allowed_statuses: Iterable[str] = ("ACCEPTED",),
) -> int:
    """Create a corrected copy of the corpus; never edit the source in place."""
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if source == output:
        raise ValueError("Output root must differ from source root")
    if output.exists():
        raise FileExistsError(output)

    allowed = frozenset(allowed_statuses)
    unknown_statuses = allowed - DECISION_STATUSES
    if unknown_statuses:
        raise ValueError(f"Unsupported allowed statuses: {sorted(unknown_statuses)}")

    selected = [
        correction
        for correction in corrections
        if correction.decision_status in allowed
    ]
    by_document: dict[tuple[str, str], dict[str, Correction]] = defaultdict(dict)
    for correction in selected:
        key = (correction.split, correction.document_id)
        if correction.entity_id in by_document[key]:
            raise ValueError(f"Duplicate correction: {key}/{correction.entity_id}")
        by_document[key][correction.entity_id] = correction

    documents = load_dataset(source)
    known_documents = {(split, document.document_id) for split, document in documents}
    unknown_documents = set(by_document) - known_documents
    if unknown_documents:
        raise ValueError(f"Unknown documents in corrections: {sorted(unknown_documents)}")

    output.mkdir(parents=True)
    applied = 0
    for split, document in documents:
        document_corrections = by_document.get((split, document.document_id), {})
        corrected = _apply_document_corrections(document, document_corrections)
        write_brat_document(corrected, output / split)
        applied += len(document_corrections)

    # Preserve any non-BRAT files in split directories without overwriting output.
    for split in SPLITS:
        for source_path in (source / split).iterdir():
            if source_path.suffix.lower() in {".txt", ".ann"}:
                continue
            destination = output / split / source_path.name
            if source_path.is_dir():
                shutil.copytree(source_path, destination)
            else:
                shutil.copy2(source_path, destination)
    return applied
