"""Minimal, strict BRAT standoff parser for the RuREBus corpus.

The parser preserves source text exactly, validates character offsets and
relation references, and can serialize the supported BRAT subset back to ANN.
It deliberately has no model/tokenizer dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


ENTITY_TYPES = frozenset({"MET", "ECO", "BIN", "CMP", "QUA", "ACT", "INST", "SOC"})
RELATION_TYPES = frozenset(
    {"NNG", "NNT", "NPS", "PNG", "PNT", "PPS", "FNG", "FNT", "FPS", "GOL", "TSK"}
)


class BratFormatError(ValueError):
    """Raised when BRAT data is malformed or inconsistent with its text."""


@dataclass(frozen=True, order=True)
class TextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise BratFormatError(f"Invalid span [{self.start}, {self.end})")


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    spans: tuple[TextSpan, ...]
    text: str

    @property
    def start(self) -> int:
        return self.spans[0].start

    @property
    def end(self) -> int:
        return self.spans[-1].end


@dataclass(frozen=True)
class Relation:
    relation_id: str
    relation_type: str
    arg1: str
    arg2: str


@dataclass(frozen=True)
class BratDocument:
    document_id: str
    text: str
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    txt_path: Path | None = None
    ann_path: Path | None = None

    @property
    def entity_by_id(self) -> Mapping[str, Entity]:
        return {entity.entity_id: entity for entity in self.entities}

    def signature(self) -> tuple:
        """Return a stable semantic signature used for round-trip checks."""
        entity_signature = tuple(
            sorted(
                (
                    entity.entity_id,
                    entity.entity_type,
                    tuple((span.start, span.end) for span in entity.spans),
                    entity.text,
                )
                for entity in self.entities
            )
        )
        relation_signature = tuple(
            sorted(
                (relation.relation_id, relation.relation_type, relation.arg1, relation.arg2)
                for relation in self.relations
            )
        )
        return self.document_id, self.text, entity_signature, relation_signature


def read_text_exact(path: str | Path) -> str:
    """Read UTF-8 text without universal-newline conversion.

    BRAT offsets are character offsets in the original text. Preserving line
    endings prevents silent offset shifts on platforms that translate CRLF.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        return stream.read()


def _parse_entity(line: str, line_number: int, ann_path: Path) -> Entity:
    parts = line.split("\t", 2)
    if len(parts) != 3:
        raise BratFormatError(f"{ann_path}:{line_number}: malformed entity line")

    entity_id, specification, surface_text = parts
    if not entity_id.startswith("T"):
        raise BratFormatError(f"{ann_path}:{line_number}: invalid entity id {entity_id!r}")

    try:
        entity_type, span_specification = specification.split(" ", 1)
        spans = tuple(
            TextSpan(*map(int, segment.split()))
            for segment in span_specification.split(";")
        )
    except (TypeError, ValueError) as error:
        raise BratFormatError(
            f"{ann_path}:{line_number}: malformed entity specification {specification!r}"
        ) from error

    if not spans:
        raise BratFormatError(f"{ann_path}:{line_number}: entity has no spans")
    if any(left.end > right.start for left, right in zip(spans, spans[1:])):
        raise BratFormatError(f"{ann_path}:{line_number}: spans overlap or are not ordered")

    return Entity(entity_id, entity_type, spans, surface_text)


def _parse_relation(line: str, line_number: int, ann_path: Path) -> Relation:
    try:
        relation_id, specification = line.split("\t", 1)
        fields = specification.split()
        relation_type = fields[0]
        arguments = dict(field.split(":", 1) for field in fields[1:])
        arg1 = arguments["Arg1"]
        arg2 = arguments["Arg2"]
    except (IndexError, KeyError, ValueError) as error:
        raise BratFormatError(f"{ann_path}:{line_number}: malformed relation line") from error

    if not relation_id.startswith("R"):
        raise BratFormatError(f"{ann_path}:{line_number}: invalid relation id {relation_id!r}")
    return Relation(relation_id, relation_type, arg1, arg2)


def parse_ann(
    ann_path: str | Path,
    text: str,
    *,
    document_id: str | None = None,
    txt_path: str | Path | None = None,
) -> BratDocument:
    """Parse and validate the RuREBus subset of BRAT standoff annotations."""
    annotation_path = Path(ann_path)
    entities: list[Entity] = []
    relations: list[Relation] = []

    for line_number, raw_line in enumerate(read_text_exact(annotation_path).splitlines(), start=1):
        if not raw_line.strip():
            continue
        if raw_line.startswith("T"):
            entities.append(_parse_entity(raw_line, line_number, annotation_path))
        elif raw_line.startswith("R"):
            relations.append(_parse_relation(raw_line, line_number, annotation_path))
        else:
            raise BratFormatError(
                f"{annotation_path}:{line_number}: unsupported BRAT record {raw_line[:1]!r}"
            )

    document = BratDocument(
        document_id=document_id or annotation_path.stem,
        text=text,
        entities=tuple(entities),
        relations=tuple(relations),
        txt_path=Path(txt_path) if txt_path is not None else None,
        ann_path=annotation_path,
    )
    validate_document(document)
    return document


def load_brat_document(
    txt_path: str | Path,
    ann_path: str | Path | None = None,
) -> BratDocument:
    """Load a TXT/ANN pair and return a validated document."""
    text_path = Path(txt_path)
    annotation_path = Path(ann_path) if ann_path is not None else text_path.with_suffix(".ann")
    if not text_path.is_file():
        raise FileNotFoundError(text_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    return parse_ann(
        annotation_path,
        read_text_exact(text_path),
        document_id=text_path.stem,
        txt_path=text_path,
    )


def validate_document(document: BratDocument) -> None:
    entity_ids: set[str] = set()
    for entity in document.entities:
        if entity.entity_id in entity_ids:
            raise BratFormatError(f"{document.document_id}: duplicate entity id {entity.entity_id}")
        entity_ids.add(entity.entity_id)
        if entity.entity_type not in ENTITY_TYPES:
            raise BratFormatError(
                f"{document.document_id}: unsupported entity type {entity.entity_type!r}"
            )
        if any(span.end > len(document.text) for span in entity.spans):
            raise BratFormatError(
                f"{document.document_id}: entity {entity.entity_id} exceeds text length"
            )
        actual_text = " ".join(document.text[span.start : span.end] for span in entity.spans)
        if actual_text != entity.text:
            raise BratFormatError(
                f"{document.document_id}: entity {entity.entity_id} text mismatch: "
                f"ANN={entity.text!r}, TXT={actual_text!r}"
            )

    relation_ids: set[str] = set()
    for relation in document.relations:
        if relation.relation_id in relation_ids:
            raise BratFormatError(
                f"{document.document_id}: duplicate relation id {relation.relation_id}"
            )
        relation_ids.add(relation.relation_id)
        if relation.relation_type not in RELATION_TYPES:
            raise BratFormatError(
                f"{document.document_id}: unsupported relation type {relation.relation_type!r}"
            )
        missing = {relation.arg1, relation.arg2} - entity_ids
        if missing:
            raise BratFormatError(
                f"{document.document_id}: relation {relation.relation_id} refers to missing "
                f"entities {sorted(missing)}"
            )


def serialize_ann(document: BratDocument) -> str:
    """Serialize the supported annotations without modifying source text."""
    lines: list[str] = []
    for entity in document.entities:
        spans = ";".join(f"{span.start} {span.end}" for span in entity.spans)
        lines.append(f"{entity.entity_id}\t{entity.entity_type} {spans}\t{entity.text}")
    for relation in document.relations:
        lines.append(
            f"{relation.relation_id}\t{relation.relation_type} "
            f"Arg1:{relation.arg1} Arg2:{relation.arg2}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def write_brat_document(
    document: BratDocument,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    txt_path = destination / f"{document.document_id}.txt"
    ann_path = destination / f"{document.document_id}.ann"
    if not overwrite and (txt_path.exists() or ann_path.exists()):
        raise FileExistsError(f"Output already exists for {document.document_id}")

    with txt_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(document.text)
    with ann_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(serialize_ann(document))
    return txt_path, ann_path


def validate_round_trip(document: BratDocument) -> None:
    """Verify that ANN serialization can be parsed without semantic changes."""
    serialized = serialize_ann(document)
    entities: list[Entity] = []
    relations: list[Relation] = []
    virtual_path = Path(f"{document.document_id}.ann")
    for line_number, raw_line in enumerate(serialized.splitlines(), start=1):
        if raw_line.startswith("T"):
            entities.append(_parse_entity(raw_line, line_number, virtual_path))
        elif raw_line.startswith("R"):
            relations.append(_parse_relation(raw_line, line_number, virtual_path))
    restored = BratDocument(
        document_id=document.document_id,
        text=document.text,
        entities=tuple(entities),
        relations=tuple(relations),
    )
    validate_document(restored)
    if restored.signature() != document.signature():
        raise BratFormatError(f"{document.document_id}: BRAT round-trip changed annotations")


def validate_pairs(txt_paths: Iterable[str | Path]) -> list[BratDocument]:
    """Convenience function for validating a sequence of TXT/ANN pairs."""
    documents = []
    for txt_path in txt_paths:
        document = load_brat_document(txt_path)
        validate_round_trip(document)
        documents.append(document)
    return documents
