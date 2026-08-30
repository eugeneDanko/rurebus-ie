"""Dynamic batching for token-classification examples."""

from __future__ import annotations

from typing import Any, Sequence

from .ner_dataset import TokenizedNerExample
from .ner_labels import IGNORE_LABEL_ID


class TokenClassificationCollator:
    """Pad token fields and labels into a model batch.

    The implementation will delegate token padding to the selected Hugging Face
    tokenizer and use ``IGNORE_LABEL_ID`` for padded label positions.
    """

    def __init__(self, tokenizer: Any, label_pad_token_id: int = IGNORE_LABEL_ID) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, examples: Sequence[TokenizedNerExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        try:
            import torch
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "PyTorch is required for batching. Install a wheel compatible with "
                "your Python version and operating system."
            ) from error

        features: list[dict[str, list[int]]] = []
        for example in examples:
            feature = {
                "input_ids": list(example.input_ids),
                "attention_mask": list(example.attention_mask),
            }
            if example.token_type_ids is not None:
                feature["token_type_ids"] = list(example.token_type_ids)
            features.append(feature)

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        sequence_length = int(batch["input_ids"].shape[1])
        padded_labels: list[list[int]] = []
        for example in examples:
            padding_length = sequence_length - len(example.labels)
            padding = [self.label_pad_token_id] * padding_length
            labels = list(example.labels)
            padded_labels.append(
                padding + labels if self.tokenizer.padding_side == "left" else labels + padding
            )
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        # Metadata stays outside model inputs and is used for document-level decoding.
        batch["document_id"] = [example.document_id for example in examples]
        batch["window_id"] = [example.window_id for example in examples]
        padded_offsets = []
        for example in examples:
            padding = ((0, 0),) * (sequence_length - len(example.offset_mapping))
            padded_offsets.append(
                padding + example.offset_mapping
                if self.tokenizer.padding_side == "left"
                else example.offset_mapping + padding
            )
        batch["offset_mapping"] = padded_offsets
        return batch


__all__ = ["TokenClassificationCollator"]
