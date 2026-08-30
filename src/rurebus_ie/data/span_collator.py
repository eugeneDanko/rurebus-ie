"""Dynamic candidate generation and padding for Span NER."""

from __future__ import annotations

from typing import Any, Sequence

from .span_dataset import TokenizedSpanExample
from .span_labels import IGNORE_LABEL_ID, SPAN_LABEL2ID


class SpanClassificationCollator:
    """Generate word-boundary spans and optionally sample training negatives."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_span_width: int = 32,
        training: bool = False,
        negative_to_positive_ratio: int = 20,
        min_negative_spans: int = 128,
        max_negative_spans: int = 1024,
        seed: int = 42,
    ) -> None:
        if max_span_width < 1:
            raise ValueError("max_span_width must be positive")
        self.tokenizer = tokenizer
        self.max_span_width = max_span_width
        self.training = training
        self.negative_to_positive_ratio = negative_to_positive_ratio
        self.min_negative_spans = min_negative_spans
        self.max_negative_spans = max_negative_spans
        self.seed = seed
        self._generator = None

    @staticmethod
    def _word_token_boundaries(example: TokenizedSpanExample) -> list[tuple[int, int]]:
        boundaries: list[list[int]] = []
        previous_word_id: int | None = None
        for token_index, (word_id, offset) in enumerate(
            zip(example.word_ids, example.offset_mapping)
        ):
            if word_id is None or offset[0] == offset[1]:
                previous_word_id = None
                continue
            if not boundaries or word_id != previous_word_id:
                boundaries.append([token_index, token_index])
            else:
                boundaries[-1][1] = token_index
            previous_word_id = word_id
        return [tuple(item) for item in boundaries]

    def _candidates(
        self, example: TokenizedSpanExample, torch: Any
    ) -> list[tuple[int, int, int, int, int]]:
        """Return token start/end, word width, label id and source candidate index."""
        words = self._word_token_boundaries(example)
        gold = {
            (span.start_token, span.end_token): span.label_id
            for span in example.gold_spans
        }
        candidates: list[tuple[int, int, int, int, int]] = []
        for start_word, (start_token, _) in enumerate(words):
            stop = min(len(words), start_word + self.max_span_width)
            for end_word in range(start_word, stop):
                end_token = words[end_word][1]
                label_id = gold.get((start_token, end_token), SPAN_LABEL2ID["NONE"])
                candidates.append(
                    (start_token, end_token, end_word - start_word + 1, label_id, len(candidates))
                )

        if not self.training:
            return candidates
        positive = [item for item in candidates if item[3] != SPAN_LABEL2ID["NONE"]]
        negative = [item for item in candidates if item[3] == SPAN_LABEL2ID["NONE"]]
        budget = max(
            self.min_negative_spans,
            len(positive) * self.negative_to_positive_ratio,
        )
        budget = min(self.max_negative_spans, budget, len(negative))
        if budget < len(negative):
            if self._generator is None:
                self._generator = torch.Generator().manual_seed(self.seed)
            selected = torch.randperm(len(negative), generator=self._generator)[:budget].tolist()
            negative = [negative[index] for index in selected]
        return sorted((*positive, *negative), key=lambda item: item[4])

    def __call__(self, examples: Sequence[TokenizedSpanExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        try:
            import torch
        except (ImportError, OSError) as error:
            raise RuntimeError("PyTorch is required for span batching") from error

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

        candidate_rows = [self._candidates(example, torch) for example in examples]
        max_candidates = max(len(row) for row in candidate_rows)
        if max_candidates == 0:
            raise ValueError("A span batch contains no candidate words")

        starts: list[list[int]] = []
        ends: list[list[int]] = []
        widths: list[list[int]] = []
        labels: list[list[int]] = []
        masks: list[list[bool]] = []
        char_offsets: list[tuple[tuple[int, int], ...]] = []
        for example, candidates in zip(examples, candidate_rows):
            left_shift = (
                sequence_length - len(example.input_ids)
                if self.tokenizer.padding_side == "left"
                else 0
            )
            pad_count = max_candidates - len(candidates)
            starts.append([item[0] + left_shift for item in candidates] + [0] * pad_count)
            ends.append([item[1] + left_shift for item in candidates] + [0] * pad_count)
            widths.append([item[2] for item in candidates] + [0] * pad_count)
            labels.append([item[3] for item in candidates] + [IGNORE_LABEL_ID] * pad_count)
            masks.append([True] * len(candidates) + [False] * pad_count)
            offsets = [
                (
                    example.offset_mapping[item[0]][0],
                    example.offset_mapping[item[1]][1],
                )
                for item in candidates
            ]
            char_offsets.append(tuple(offsets + [(0, 0)] * pad_count))

        batch.update(
            {
                "span_starts": torch.tensor(starts, dtype=torch.long),
                "span_ends": torch.tensor(ends, dtype=torch.long),
                "span_widths": torch.tensor(widths, dtype=torch.long),
                "span_mask": torch.tensor(masks, dtype=torch.bool),
                "labels": torch.tensor(labels, dtype=torch.long),
                "document_id": [example.document_id for example in examples],
                "window_id": [example.window_id for example in examples],
                "span_char_offsets": char_offsets,
            }
        )
        return batch


__all__ = ["SpanClassificationCollator"]
