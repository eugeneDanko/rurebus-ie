"""Dynamic tokenization and padding for marked relation pairs."""

from __future__ import annotations

from typing import Any, Sequence

from .relation_dataset import E1_START, E2_START, ENTITY_MARKERS, RelationTextExample


class RelationClassificationCollator:
    def __init__(self, tokenizer: Any, *, max_length: int = 256) -> None:
        if max_length < 16:
            raise ValueError("max_length must be at least 16 tokens")
        self.tokenizer = tokenizer
        self.max_length = max_length
        marker_ids = tokenizer.convert_tokens_to_ids(list(ENTITY_MARKERS))
        if len(set(marker_ids)) != len(ENTITY_MARKERS):
            raise ValueError("Entity markers must have distinct tokenizer ids")
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        if unknown_id is not None and unknown_id in marker_ids:
            raise ValueError("Entity markers were not registered as special tokens")
        self.e1_start_token_id = int(marker_ids[ENTITY_MARKERS.index(E1_START)])
        self.e2_start_token_id = int(marker_ids[ENTITY_MARKERS.index(E2_START)])

    def __call__(self, examples: Sequence[RelationTextExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty relation batch")
        try:
            import torch
        except (ImportError, OSError) as error:
            raise RuntimeError("PyTorch is required for relation batching") from error
        encoded = self.tokenizer(
            [example.marked_text for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        for marker_id, marker_name in (
            (self.e1_start_token_id, E1_START),
            (self.e2_start_token_id, E2_START),
        ):
            counts = input_ids.eq(marker_id).sum(dim=1)
            invalid = (counts != 1).nonzero(as_tuple=False).flatten().tolist()
            if invalid:
                documents = [examples[index].document_id for index in invalid[:5]]
                raise ValueError(
                    f"Marker {marker_name} was truncated or duplicated for rows {invalid[:5]} "
                    f"(documents={documents}). Increase max_length or reduce context_margin."
                )
        encoded["labels"] = torch.tensor(
            [example.labels for example in examples], dtype=torch.float
        )
        encoded["document_id"] = [example.document_id for example in examples]
        encoded["arg1_id"] = [example.arg1_id for example in examples]
        encoded["arg2_id"] = [example.arg2_id for example in examples]
        encoded["arg1_signature"] = [
            (example.arg1_type, example.arg1_start, example.arg1_end)
            for example in examples
        ]
        encoded["arg2_signature"] = [
            (example.arg2_type, example.arg2_start, example.arg2_end)
            for example in examples
        ]
        return encoded


__all__ = ["RelationClassificationCollator"]
