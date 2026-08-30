"""Pair generation and marker contexts for RuREBus relation extraction.

RuREBus contains one genuinely multi-label entity pair, so targets are
multi-hot vectors over positive relation types. ``NO_RELATION`` is represented
by an all-zero vector instead of a mutually exclusive twelfth logit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from .brat_parser import BratDocument, Entity
from .ner_dataset import load_documents_from_manifest
from .relation_labels import RELATION_LABEL2ID, RELATION_LABELS


E1_START = "[E1]"
E1_END = "[/E1]"
E2_START = "[E2]"
E2_END = "[/E2]"
ENTITY_MARKERS = (E1_START, E1_END, E2_START, E2_END)


@dataclass(frozen=True)
class RelationTextExample:
    document_id: str
    arg1_id: str
    arg2_id: str
    arg1_type: str
    arg2_type: str
    arg1_start: int
    arg1_end: int
    arg2_start: int
    arg2_end: int
    marked_text: str
    labels: tuple[float, ...]

    @property
    def is_positive(self) -> bool:
        return any(self.labels)


class RuReBusRelationDataset(Sequence[RelationTextExample]):
    def __init__(
        self,
        examples: Sequence[RelationTextExample],
        documents: Sequence[BratDocument],
        *,
        candidate_gold_relations: Sequence[tuple[str, str, str, str]] = (),
    ) -> None:
        self._examples = tuple(examples)
        self._documents = tuple(documents)
        self._document_by_id = {document.document_id: document for document in documents}
        self._gold_relations = tuple(
            (
                document.document_id,
                relation.relation_type,
                relation.arg1,
                relation.arg2,
            )
            for document in documents
            for relation in document.relations
        )
        self._candidate_gold_relations = tuple(candidate_gold_relations)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> RelationTextExample:
        return self._examples[index]

    def __iter__(self) -> Iterator[RelationTextExample]:
        return iter(self._examples)

    @property
    def documents(self) -> tuple[BratDocument, ...]:
        return self._documents

    @property
    def document_by_id(self) -> Mapping[str, BratDocument]:
        return self._document_by_id

    @property
    def gold_relations(self) -> tuple[tuple[str, str, str, str], ...]:
        return self._gold_relations

    @property
    def candidate_gold_relations(self) -> tuple[tuple[str, str, str, str], ...]:
        return self._candidate_gold_relations

    @property
    def candidate_recall(self) -> float:
        gold = set(self._gold_relations)
        return len(set(self._candidate_gold_relations)) / len(gold) if gold else 1.0

    @property
    def positive_pair_count(self) -> int:
        return sum(example.is_positive for example in self._examples)

    @property
    def negative_pair_count(self) -> int:
        return len(self) - self.positive_pair_count

    @property
    def positive_label_counts(self) -> tuple[int, ...]:
        return tuple(
            sum(int(example.labels[index] > 0.0) for example in self._examples)
            for index in range(len(RELATION_LABELS))
        )


def _non_overlapping(left: Entity, right: Entity) -> bool:
    return left.end <= right.start or right.end <= left.start


def _gap(left: Entity, right: Entity) -> int:
    return max(0, max(left.start, right.start) - min(left.end, right.end))


def marked_relation_context(
    document: BratDocument,
    arg1: Entity,
    arg2: Entity,
    *,
    context_margin: int = 128,
) -> str:
    """Return local text with role markers while preserving argument direction."""
    if not _non_overlapping(arg1, arg2):
        raise ValueError(
            f"Overlapping pair cannot use flat R-BERT markers: {arg1.entity_id}, {arg2.entity_id}"
        )
    if context_margin < 0:
        raise ValueError("context_margin must be non-negative")
    context_start = max(0, min(arg1.start, arg2.start) - context_margin)
    context_end = min(len(document.text), max(arg1.end, arg2.end) + context_margin)
    role = {
        arg1.entity_id: (E1_START, E1_END),
        arg2.entity_id: (E2_START, E2_END),
    }
    ordered = sorted((arg1, arg2), key=lambda entity: (entity.start, entity.end))
    pieces: list[str] = []
    cursor = context_start
    for entity in ordered:
        start_marker, end_marker = role[entity.entity_id]
        pieces.extend(
            (
                document.text[cursor : entity.start],
                start_marker,
                document.text[entity.start : entity.end],
                end_marker,
            )
        )
        cursor = entity.end
    pieces.append(document.text[cursor:context_end])
    return "".join(pieces)


def _candidate_pairs(
    document: BratDocument,
    *,
    max_pair_distance: int,
) -> set[tuple[str, str]]:
    if max_pair_distance < 0:
        raise ValueError("max_pair_distance must be non-negative")
    entities = sorted(document.entities, key=lambda item: (item.start, item.end, item.entity_id))
    candidates: set[tuple[str, str]] = set()
    for left_index, left in enumerate(entities):
        for right in entities[left_index + 1 :]:
            if right.start - left.end > max_pair_distance:
                break
            if not _non_overlapping(left, right):
                continue
            candidates.add((left.entity_id, right.entity_id))
            candidates.add((right.entity_id, left.entity_id))
    return candidates


def build_relation_examples(
    documents: Sequence[BratDocument],
    *,
    max_pair_distance: int = 128,
    context_margin: int = 128,
    negative_to_positive_ratio: float | None = None,
    min_negative_pairs: int = 0,
    seed: int = 42,
) -> RuReBusRelationDataset:
    """Generate directed relation candidates and deterministic negatives.

    If ``negative_to_positive_ratio`` is ``None``, every candidate negative is
    retained.  This mode is intended for calibration and final evaluation.
    Gold relations outside the candidate radius remain in ``gold_relations``
    and therefore count as false negatives in extraction metrics.
    """
    if negative_to_positive_ratio is not None and negative_to_positive_ratio < 0:
        raise ValueError("negative_to_positive_ratio must be non-negative or None")
    if min_negative_pairs < 0:
        raise ValueError("min_negative_pairs must be non-negative")

    examples: list[RelationTextExample] = []
    candidate_gold: list[tuple[str, str, str, str]] = []
    for document_index, document in enumerate(documents):
        entity_by_id = document.entity_by_id
        pair_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
        for relation in document.relations:
            pair_labels[(relation.arg1, relation.arg2)].add(relation.relation_type)
        candidate_pairs = _candidate_pairs(
            document, max_pair_distance=max_pair_distance
        )
        positive_pairs = sorted(pair for pair in candidate_pairs if pair in pair_labels)
        negative_pairs = sorted(pair for pair in candidate_pairs if pair not in pair_labels)
        if negative_to_positive_ratio is not None:
            budget = max(
                min_negative_pairs,
                round(len(positive_pairs) * negative_to_positive_ratio),
            )
            if budget < len(negative_pairs):
                generator = random.Random(seed + document_index * 1009)
                negative_pairs = sorted(generator.sample(negative_pairs, budget))

        for arg1_id, arg2_id in (*positive_pairs, *negative_pairs):
            arg1 = entity_by_id[arg1_id]
            arg2 = entity_by_id[arg2_id]
            labels = tuple(
                float(label in pair_labels.get((arg1_id, arg2_id), set()))
                for label in RELATION_LABELS
            )
            examples.append(
                RelationTextExample(
                    document_id=document.document_id,
                    arg1_id=arg1_id,
                    arg2_id=arg2_id,
                    arg1_type=arg1.entity_type,
                    arg2_type=arg2.entity_type,
                    arg1_start=arg1.start,
                    arg1_end=arg1.end,
                    arg2_start=arg2.start,
                    arg2_end=arg2.end,
                    marked_text=marked_relation_context(
                        document, arg1, arg2, context_margin=context_margin
                    ),
                    labels=labels,
                )
            )
            for label in pair_labels.get((arg1_id, arg2_id), set()):
                candidate_gold.append((document.document_id, label, arg1_id, arg2_id))
    return RuReBusRelationDataset(
        examples,
        documents,
        candidate_gold_relations=candidate_gold,
    )


def build_relation_dataset_from_manifest(
    manifest_path: str | Path,
    split: str,
    **options: object,
) -> RuReBusRelationDataset:
    documents = load_documents_from_manifest(manifest_path, split)
    return build_relation_examples(documents, **options)


__all__ = [
    "E1_END",
    "E1_START",
    "E2_END",
    "E2_START",
    "ENTITY_MARKERS",
    "RelationTextExample",
    "RuReBusRelationDataset",
    "build_relation_dataset_from_manifest",
    "build_relation_examples",
    "marked_relation_context",
]
