"""Two-level Span NER with curriculum and supervised contrastive learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.utils import ModelOutput

from rurebus_ie.data.span_hierarchy import (
    DEFAULT_SUPERCLASS_GROUPS,
    build_span_label_hierarchy,
)
from rurebus_ie.data.span_labels import IGNORE_LABEL_ID, SPAN_LABEL2ID
from rurebus_ie.models.span_ner import SpanNerConfig, SpanNerModel


class HierarchicalSpanNerConfig(SpanNerConfig):
    """Serializable configuration for hierarchical Span NER."""

    model_type = "rurebus_hierarchical_span_ner"

    def __init__(
        self,
        *,
        superclass_groups: Sequence[Sequence[str]] | None = None,
        contrastive_projection_size: int = 128,
        contrastive_temperature: float = 0.1,
        contrastive_weight: float = 0.1,
        coarse_aux_weight: float = 0.3,
        contrastive_max_spans: int = 256,
        training_stage: str = "coarse",
        **kwargs: Any,
    ) -> None:
        hierarchy = build_span_label_hierarchy(
            superclass_groups or DEFAULT_SUPERCLASS_GROUPS
        )
        super().__init__(**kwargs)
        self.superclass_groups = [list(group) for group in hierarchy.groups]
        self.coarse_labels = list(hierarchy.coarse_labels)
        self.coarse_label2id = {
            label: index for index, label in enumerate(hierarchy.coarse_labels)
        }
        self.coarse_id2label = {
            index: label for label, index in self.coarse_label2id.items()
        }
        self.contrastive_projection_size = int(contrastive_projection_size)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_weight = float(contrastive_weight)
        self.coarse_aux_weight = float(coarse_aux_weight)
        self.contrastive_max_spans = int(contrastive_max_spans)
        self.training_stage = str(training_stage)
        if self.contrastive_projection_size < 1:
            raise ValueError("contrastive_projection_size must be positive")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive")
        if self.contrastive_weight < 0.0 or self.coarse_aux_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.contrastive_max_spans < 2:
            raise ValueError("contrastive_max_spans must be at least 2")
        if self.training_stage not in {"coarse", "fine"}:
            raise ValueError("training_stage must be 'coarse' or 'fine'")


@dataclass
class HierarchicalSpanNerOutput(ModelOutput):
    """Fine predictions plus individual training-loss components."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    coarse_logits: torch.Tensor | None = None
    fine_loss: torch.Tensor | None = None
    coarse_loss: torch.Tensor | None = None
    contrastive_loss: torch.Tensor | None = None


class HierarchicalSpanNerModel(SpanNerModel):
    """Predict internal superclasses and original RuREBus labels.

    The public ``logits`` field always corresponds to the original label set.
    Superclass logits exist only as an auxiliary/curriculum signal.
    """

    config_class = HierarchicalSpanNerConfig

    def __init__(self, config: HierarchicalSpanNerConfig) -> None:
        super().__init__(config)
        hierarchy = build_span_label_hierarchy(config.superclass_groups)
        representation_size = (
            int(self.encoder.config.hidden_size) * 3 + config.width_embedding_dim
        )
        self.coarse_classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(representation_size, config.span_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.span_hidden_size, hierarchy.num_coarse_labels),
        )
        self.contrastive_projection = nn.Sequential(
            nn.Linear(representation_size, config.span_hidden_size),
            nn.GELU(),
            nn.Linear(config.span_hidden_size, config.contrastive_projection_size),
        )
        # Keep the mapping as regular Python data, not as a non-persistent
        # buffer. Hugging Face may instantiate models on the meta device inside
        # ``from_pretrained``; a non-persistent buffer is then absent from the
        # checkpoint and can be materialized with uninitialized values. That
        # produced random coarse targets and a CUDA device-side assert after a
        # checkpoint reload. The tiny tensor is rebuilt on the labels' device
        # in ``_coarse_labels`` instead.
        self._fine_to_coarse_id_values = tuple(hierarchy.fine_to_coarse_ids)
        # ``SpanNerModel`` already initialized the encoder and fine head.  Only
        # initialize modules added by this subclass, so pretrained weights are
        # not touched when ``from_encoder_pretrained`` replaces the encoder.
        self.coarse_classifier.apply(self._init_weights)
        self.contrastive_projection.apply(self._init_weights)

    @classmethod
    def from_encoder_pretrained(
        cls,
        pretrained_name: str,
        *,
        max_span_width: int = 32,
        width_embedding_dim: int = 32,
        span_hidden_size: int = 512,
        dropout: float = 0.1,
        none_loss_weight: float = 1.0,
        superclass_groups: Sequence[Sequence[str]] | None = None,
        contrastive_projection_size: int = 128,
        contrastive_temperature: float = 0.1,
        contrastive_weight: float = 0.1,
        coarse_aux_weight: float = 0.3,
        contrastive_max_spans: int = 256,
    ) -> "HierarchicalSpanNerModel":
        encoder_config = AutoConfig.from_pretrained(pretrained_name)
        if hasattr(encoder_config, "hidden_dropout_prob"):
            encoder_config.hidden_dropout_prob = dropout
        encoder = AutoModel.from_pretrained(pretrained_name, config=encoder_config)
        config = HierarchicalSpanNerConfig(
            encoder_config=encoder_config.to_dict(),
            max_span_width=max_span_width,
            width_embedding_dim=width_embedding_dim,
            span_hidden_size=span_hidden_size,
            dropout=dropout,
            none_loss_weight=none_loss_weight,
            superclass_groups=superclass_groups,
            contrastive_projection_size=contrastive_projection_size,
            contrastive_temperature=contrastive_temperature,
            contrastive_weight=contrastive_weight,
            coarse_aux_weight=coarse_aux_weight,
            contrastive_max_spans=contrastive_max_spans,
        )
        model = cls(config)
        model.encoder = encoder
        return model

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"coarse", "fine"}:
            raise ValueError("stage must be 'coarse' or 'fine'")
        self.config.training_stage = stage

    def _span_representations(
        self,
        sequence: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
        span_widths: torch.Tensor,
    ) -> torch.Tensor:
        start_values = self._gather(sequence, span_starts)
        end_values = self._gather(sequence, span_ends)
        prefix = torch.cat(
            (
                sequence.new_zeros(sequence.shape[0], 1, sequence.shape[2]),
                sequence.cumsum(dim=1),
            ),
            dim=1,
        )
        summed = self._gather(prefix, span_ends + 1) - self._gather(prefix, span_starts)
        token_lengths = (span_ends - span_starts + 1).clamp_min(1).unsqueeze(-1)
        mean_values = summed / token_lengths
        width_values = self.width_embedding(
            span_widths.clamp(min=0, max=self.config.max_span_width)
        )
        return torch.cat(
            (start_values, end_values, mean_values, width_values), dim=-1
        )

    def _coarse_labels(self, labels: torch.Tensor) -> torch.Tensor:
        result = labels.new_full(labels.shape, IGNORE_LABEL_ID)
        active = labels != IGNORE_LABEL_ID
        mapping = labels.new_tensor(self._fine_to_coarse_id_values)
        result[active] = mapping[labels[active]]
        return result

    def _classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        weights = logits.new_ones(logits.shape[-1])
        weights[0] = self.config.none_loss_weight
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            weight=weights,
            ignore_index=IGNORE_LABEL_ID,
        )

    def _supervised_contrastive_loss(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor,
        span_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        active = (labels != IGNORE_LABEL_ID) & (labels != 0)
        if span_mask is not None:
            active &= span_mask.bool()
        vectors = representations[active]
        targets = labels[active]
        if vectors.shape[0] < 2:
            return representations.sum() * 0.0
        if vectors.shape[0] > self.config.contrastive_max_spans:
            indices = torch.randperm(vectors.shape[0], device=vectors.device)[
                : self.config.contrastive_max_spans
            ]
            vectors = vectors[indices]
            targets = targets[indices]

        projected = nn.functional.normalize(
            self.contrastive_projection(vectors), dim=-1
        )
        similarities = projected @ projected.transpose(0, 1)
        similarities = similarities / self.config.contrastive_temperature
        diagonal = torch.eye(
            similarities.shape[0], dtype=torch.bool, device=similarities.device
        )
        positive_mask = targets[:, None].eq(targets[None, :]) & ~diagonal
        valid_anchors = positive_mask.any(dim=1)
        if not bool(valid_anchors.any()):
            return representations.sum() * 0.0

        denominator_logits = similarities.masked_fill(diagonal, float("-inf"))
        log_probabilities = similarities - torch.logsumexp(
            denominator_logits, dim=1, keepdim=True
        )
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        per_anchor = -(
            log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_count
        )
        return per_anchor[valid_anchors].mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        span_starts: torch.Tensor | None = None,
        span_ends: torch.Tensor | None = None,
        span_widths: torch.Tensor | None = None,
        span_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        training_stage: str | None = None,
        **kwargs: Any,
    ) -> HierarchicalSpanNerOutput:
        if span_starts is None or span_ends is None or span_widths is None:
            raise ValueError("span_starts, span_ends and span_widths are required")
        stage = training_stage or self.config.training_stage
        if stage not in {"coarse", "fine"}:
            raise ValueError("training_stage must be 'coarse' or 'fine'")

        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        sequence = self.encoder(**encoder_inputs, **kwargs).last_hidden_state
        representations = self._span_representations(
            sequence, span_starts, span_ends, span_widths
        )
        fine_logits = self.classifier(representations)
        coarse_logits = self.coarse_classifier(representations)

        loss = fine_loss = coarse_loss = contrastive_loss = None
        if labels is not None:
            coarse_labels = self._coarse_labels(labels)
            coarse_loss = self._classification_loss(coarse_logits, coarse_labels)
            if stage == "coarse":
                contrastive_loss = self._supervised_contrastive_loss(
                    representations, coarse_labels, span_mask
                )
                loss = coarse_loss + self.config.contrastive_weight * contrastive_loss
            else:
                fine_loss = self._classification_loss(fine_logits, labels)
                contrastive_loss = self._supervised_contrastive_loss(
                    representations, labels, span_mask
                )
                loss = (
                    fine_loss
                    + self.config.coarse_aux_weight * coarse_loss
                    + self.config.contrastive_weight * contrastive_loss
                )

        return HierarchicalSpanNerOutput(
            loss=loss,
            logits=fine_logits,
            coarse_logits=coarse_logits,
            fine_loss=fine_loss,
            coarse_loss=coarse_loss,
            contrastive_loss=contrastive_loss,
        )


def build_hierarchical_rubert_span_classifier(
    pretrained_name: str, **options: Any
) -> HierarchicalSpanNerModel:
    return HierarchicalSpanNerModel.from_encoder_pretrained(pretrained_name, **options)


__all__ = [
    "HierarchicalSpanNerConfig",
    "HierarchicalSpanNerModel",
    "HierarchicalSpanNerOutput",
    "build_hierarchical_rubert_span_classifier",
]
