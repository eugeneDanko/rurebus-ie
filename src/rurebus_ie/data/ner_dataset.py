"""RuREBus datasets for RuBERT token classification.

The module keeps BRAT character offsets as the source of truth. A fast
Hugging Face tokenizer supplies offsets aligned to the canonical BIO schema.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .brat_parser import BratDocument, Entity, load_brat_document
from .ner_labels import IGNORE_LABEL_ID, LABEL2ID


@dataclass(frozen=True)
class TokenizedNerExample:
    """One tokenized document window with labels and source character offsets."""

    document_id: str
    window_id: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    token_type_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.attention_mask),
            len(self.labels),
            len(self.offset_mapping),
        }
        if self.token_type_ids is not None:
            lengths.add(len(self.token_type_ids))
        if len(lengths) != 1:
            raise ValueError("All token-level fields must have equal length")


class RuReBusNerDataset(Sequence[TokenizedNerExample]):
    """A sequence compatible with a PyTorch ``DataLoader``."""

    def __init__(
        self,
        examples: Sequence[TokenizedNerExample],
        documents: Sequence[BratDocument] = (),
    ) -> None:
        self._examples = tuple(examples)
        self._document_texts = {document.document_id: document.text for document in documents}
        self._gold_entities = {
            document.document_id: tuple(
                (entity.entity_type, entity.start, entity.end, entity.text)
                for entity in document.entities
            )
            for document in documents
        }

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TokenizedNerExample:
        return self._examples[index]

    def __iter__(self) -> Iterator[TokenizedNerExample]:
        return iter(self._examples)

    @property
    def document_texts(self) -> Mapping[str, str]:
        return self._document_texts

    @property
    def gold_entities(self) -> Mapping[str, tuple[tuple[str, int, int, str], ...]]:
        return self._gold_entities


def _validate_entities(document: BratDocument) -> tuple[Entity, ...]:
    entities = tuple(sorted(document.entities, key=lambda item: (item.start, item.end)))
    for entity in entities:
        if len(entity.spans) != 1:
            raise ValueError(
                f"{document.document_id}: discontinuous entity {entity.entity_id} "
                "cannot be represented by BIO labels"
            )
    for left, right in zip(entities, entities[1:]):
        if left.end > right.start:
            raise ValueError(
                f"{document.document_id}: overlapping entities {left.entity_id} and "
                f"{right.entity_id} cannot be represented by one BIO sequence"
            )
    return entities


def _labels_for_offsets(
    document: BratDocument,
    offsets: Sequence[Sequence[int]],
    *,
    entities: Sequence[Entity] | None = None,
    label_all_subtokens: bool,
    ignore_label_id: int,
) -> tuple[int, ...]:
    ordered_entities = tuple(entities) if entities is not None else _validate_entities(document)
    labels: list[int] = []
    previous_entity_id: str | None = None
    entity_index = 0

    for raw_start, raw_end in offsets:
        start, end = int(raw_start), int(raw_end)
        if start == end:
            labels.append(ignore_label_id)
            previous_entity_id = None
            continue

        while (
            entity_index < len(ordered_entities)
            and ordered_entities[entity_index].end <= start
        ):
            entity_index += 1
        entity = (
            ordered_entities[entity_index]
            if entity_index < len(ordered_entities)
            and start < ordered_entities[entity_index].end
            and end > ordered_entities[entity_index].start
            else None
        )
        if entity is None:
            labels.append(LABEL2ID["O"])
            previous_entity_id = None
            continue

        if start <= entity.start < end:
            prefix = "B"
        elif label_all_subtokens:
            prefix = "I"
        elif previous_entity_id == entity.entity_id:
            labels.append(ignore_label_id)
            continue
        else:
            labels.append(ignore_label_id)
            previous_entity_id = entity.entity_id
            continue

        labels.append(LABEL2ID[f"{prefix}-{entity.entity_type}"])
        previous_entity_id = entity.entity_id

    return tuple(labels)


def build_ner_examples(
    documents: Sequence[BratDocument],
    tokenizer: Any,
    *,
    max_length: int = 512,
    stride: int = 128,
    label_all_subtokens: bool = True,
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> RuReBusNerDataset:
    """Convert processed BRAT documents into labeled tokenizer windows."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required to obtain offset_mapping")
    if max_length < 3:
        raise ValueError("max_length must leave room for text and special tokens")
    if not 0 <= stride < max_length:
        raise ValueError("stride must satisfy 0 <= stride < max_length")

    examples: list[TokenizedNerExample] = []
    for document in documents:
        entities = _validate_entities(document)
        encoded = tokenizer(
            document.text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
            padding=False,
        )
        for window_id in range(len(encoded["input_ids"])):
            offsets = tuple(
                (int(start), int(end)) for start, end in encoded["offset_mapping"][window_id]
            )
            token_type_ids = encoded.get("token_type_ids")
            examples.append(
                TokenizedNerExample(
                    document_id=document.document_id,
                    window_id=window_id,
                    input_ids=tuple(int(value) for value in encoded["input_ids"][window_id]),
                    attention_mask=tuple(
                        int(value) for value in encoded["attention_mask"][window_id]
                    ),
                    labels=_labels_for_offsets(
                        document,
                        offsets,
                        entities=entities,
                        label_all_subtokens=label_all_subtokens,
                        ignore_label_id=ignore_label_id,
                    ),
                    offset_mapping=offsets,
                    token_type_ids=(
                        tuple(int(value) for value in token_type_ids[window_id])
                        if token_type_ids is not None
                        else None
                    ),
                )
            )
    return RuReBusNerDataset(examples, documents)


def load_documents_from_manifest(
    manifest_path: str | Path,
    split: str,
    *,
    data_root: str | Path | None = None,
) -> tuple[BratDocument, ...]:
    """Load one processed split using paths stored in ``manifest.csv``."""
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Processed manifest not found: {manifest}. Run rurebus-preprocess first."
        )
    root = Path(data_root).expanduser().resolve() if data_root else manifest.parent.parent
    documents: list[BratDocument] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != split:
                continue
            txt_path = Path(row["processed_txt_path"])
            ann_path = Path(row["processed_ann_path"])
            if not txt_path.is_absolute():
                txt_path = root / txt_path
            if not ann_path.is_absolute():
                ann_path = root / ann_path
            documents.append(load_brat_document(txt_path, ann_path))
    if not documents:
        raise ValueError(f"Split {split!r} is empty or absent in {manifest}")
    return tuple(documents)


def build_ner_dataset_from_manifest(
    manifest_path: str | Path,
    split: str,
    tokenizer: Any,
    **tokenization_options: Any,
) -> RuReBusNerDataset:
    """Load a manifest split and create model-ready token windows."""
    documents = load_documents_from_manifest(manifest_path, split)
    return build_ner_examples(documents, tokenizer, **tokenization_options)


__all__ = [
    "RuReBusNerDataset",
    "TokenizedNerExample",
    "build_ner_dataset_from_manifest",
    "build_ner_examples",
    "load_documents_from_manifest",
]
