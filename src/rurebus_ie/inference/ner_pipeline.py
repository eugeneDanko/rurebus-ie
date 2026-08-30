"""BIO decoding and document-level inference for the RuBERT NER baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rurebus_ie.data.ner_labels import ID2LABEL, IGNORE_LABEL_ID


@dataclass(frozen=True)
class PredictedEntity:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float


def decode_bio_predictions(
    label_ids: Sequence[int],
    offset_mapping: Sequence[Sequence[int]],
    text: str,
    *,
    id2label: Mapping[int, str] = ID2LABEL,
    confidences: Sequence[float] | None = None,
) -> tuple[PredictedEntity, ...]:
    """Convert one token window into entities with source-character offsets.

    An invalid ``I-X`` after ``O`` or another entity type is repaired as
    ``B-X``. This is the standard deterministic fallback for unconstrained
    token classifiers.
    """
    if len(label_ids) != len(offset_mapping):
        raise ValueError("label_ids and offset_mapping must have equal length")
    if confidences is not None and len(confidences) != len(label_ids):
        raise ValueError("confidences and label_ids must have equal length")

    entities: list[PredictedEntity] = []
    current_type: str | None = None
    current_start = 0
    current_end = 0
    current_scores: list[float] = []

    def close_current() -> None:
        nonlocal current_type, current_start, current_end, current_scores
        if current_type is not None and current_end > current_start:
            entities.append(
                PredictedEntity(
                    text=text[current_start:current_end],
                    entity_type=current_type,
                    start=current_start,
                    end=current_end,
                    confidence=(
                        sum(current_scores) / len(current_scores) if current_scores else 1.0
                    ),
                )
            )
        current_type = None
        current_scores = []

    for index, (label_id, raw_offset) in enumerate(zip(label_ids, offset_mapping)):
        start, end = int(raw_offset[0]), int(raw_offset[1])
        if int(label_id) == IGNORE_LABEL_ID or start == end:
            close_current()
            continue
        label = id2label.get(int(label_id))
        if label is None:
            raise ValueError(f"Unknown label id: {label_id}")
        score = float(confidences[index]) if confidences is not None else 1.0
        if label == "O":
            close_current()
            continue

        try:
            prefix, entity_type = label.split("-", 1)
        except ValueError as error:
            raise ValueError(f"Invalid BIO label: {label!r}") from error
        if prefix not in {"B", "I"}:
            raise ValueError(f"Invalid BIO prefix in {label!r}")

        if prefix == "B" or current_type != entity_type:
            close_current()
            current_type = entity_type
            current_start = start
            current_end = end
            current_scores = [score]
        else:
            current_end = max(current_end, end)
            current_scores.append(score)
    close_current()
    return tuple(entities)


def merge_window_predictions(
    predictions: Sequence[PredictedEntity],
) -> tuple[PredictedEntity, ...]:
    """Deduplicate overlapping-window output and remove contained fragments."""
    exact: dict[tuple[str, int, int], PredictedEntity] = {}
    for entity in predictions:
        key = (entity.entity_type, entity.start, entity.end)
        if key not in exact or entity.confidence > exact[key].confidence:
            exact[key] = entity

    candidates = sorted(
        exact.values(),
        key=lambda item: (-(item.end - item.start), -item.confidence, item.start),
    )
    kept: list[PredictedEntity] = []
    for candidate in candidates:
        contained = any(
            candidate.entity_type == other.entity_type
            and other.start <= candidate.start
            and candidate.end <= other.end
            and (other.start, other.end) != (candidate.start, candidate.end)
            for other in kept
        )
        if not contained:
            kept.append(candidate)
    return tuple(sorted(kept, key=lambda item: (item.start, item.end, item.entity_type)))


class NerInferencePipeline:
    """Tokenize text, run the model, decode BIO labels and merge windows."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_length: int = 512,
        stride: int = 128,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.device = device

    def predict(self, text: str) -> tuple[PredictedEntity, ...]:
        try:
            import torch
        except (ImportError, OSError) as error:
            raise RuntimeError("A working PyTorch installation is required for inference") from error

        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(device)
        self.model.eval()
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            padding=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded.pop("overflow_to_sample_mapping", None)
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            logits = self.model(**model_inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            confidence, labels = probabilities.max(dim=-1)

        entities: list[PredictedEntity] = []
        id2label = {
            int(key): value
            for key, value in getattr(self.model.config, "id2label", ID2LABEL).items()
        }
        for window_labels, window_offsets, window_confidence in zip(
            labels.cpu().tolist(), offsets, confidence.cpu().tolist()
        ):
            entities.extend(
                decode_bio_predictions(
                    window_labels,
                    window_offsets,
                    text,
                    id2label=id2label,
                    confidences=window_confidence,
                )
            )
        return merge_window_predictions(entities)


__all__ = [
    "NerInferencePipeline",
    "PredictedEntity",
    "decode_bio_predictions",
    "merge_window_predictions",
]
