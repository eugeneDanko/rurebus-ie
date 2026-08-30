"""Marker-aware R-BERT classifier for directed RuREBus relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rurebus_ie.data.relation_labels import (
    RELATION_ID2LABEL,
    RELATION_LABEL2ID,
)

try:
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
    from transformers.utils import ModelOutput
except (ImportError, OSError) as error:  # pragma: no cover
    raise RuntimeError(
        "torch and transformers are required to import the relation classifier"
    ) from error


class RelationClassifierConfig(PretrainedConfig):
    model_type = "rurebus_rbert_relation"

    def __init__(
        self,
        *,
        encoder_config: dict[str, Any] | None = None,
        relation_hidden_size: int = 512,
        dropout: float = 0.1,
        e1_start_token_id: int | None = None,
        e2_start_token_id: int | None = None,
        positive_class_weights: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_labels", len(RELATION_LABEL2ID))
        kwargs.setdefault("label2id", RELATION_LABEL2ID)
        kwargs.setdefault("id2label", RELATION_ID2LABEL)
        kwargs.setdefault("problem_type", "multi_label_classification")
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or {}
        self.relation_hidden_size = int(relation_hidden_size)
        self.dropout = float(dropout)
        self.e1_start_token_id = (
            None if e1_start_token_id is None else int(e1_start_token_id)
        )
        self.e2_start_token_id = (
            None if e2_start_token_id is None else int(e2_start_token_id)
        )
        self.positive_class_weights = (
            [1.0] * self.num_labels
            if positive_class_weights is None
            else [float(value) for value in positive_class_weights]
        )
        if len(self.positive_class_weights) != self.num_labels:
            raise ValueError("positive_class_weights must match num_labels")


@dataclass
class RelationClassifierOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class RelationClassifierModel(PreTrainedModel):
    """Pool ``[CLS]``, ``[E1]`` and ``[E2]`` states and predict relations.

    Eleven independent logits are used because RuREBus contains a multi-label
    entity pair.  An all-zero target represents ``NO_RELATION``.
    """

    config_class = RelationClassifierConfig
    base_model_prefix = "encoder"

    def __init__(self, config: RelationClassifierConfig) -> None:
        super().__init__(config)
        if not config.encoder_config:
            raise ValueError("encoder_config must contain a base model configuration")
        encoder_values = dict(config.encoder_config)
        model_type = encoder_values.pop("model_type", None)
        if not model_type:
            raise ValueError("encoder_config.model_type is required")
        encoder_config = AutoConfig.for_model(model_type, **encoder_values)
        self.encoder = AutoModel.from_config(encoder_config)
        hidden_size = int(encoder_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size * 3, config.relation_hidden_size),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.relation_hidden_size, config.num_labels),
        )
        self.post_init()

    @classmethod
    def from_encoder_pretrained(
        cls,
        pretrained_name: str,
        *,
        tokenizer_size: int,
        e1_start_token_id: int,
        e2_start_token_id: int,
        relation_hidden_size: int = 512,
        dropout: float = 0.1,
        positive_class_weights: Sequence[float] | None = None,
    ) -> "RelationClassifierModel":
        encoder_config = AutoConfig.from_pretrained(pretrained_name)
        if hasattr(encoder_config, "hidden_dropout_prob"):
            encoder_config.hidden_dropout_prob = dropout
        encoder = AutoModel.from_pretrained(pretrained_name, config=encoder_config)
        encoder.resize_token_embeddings(int(tokenizer_size))
        config = RelationClassifierConfig(
            encoder_config=encoder.config.to_dict(),
            relation_hidden_size=relation_hidden_size,
            dropout=dropout,
            e1_start_token_id=e1_start_token_id,
            e2_start_token_id=e2_start_token_id,
            positive_class_weights=positive_class_weights,
        )
        model = cls(config)
        model.encoder = encoder
        return model

    @staticmethod
    def _marker_positions(input_ids: torch.Tensor, marker_id: int, name: str) -> torch.Tensor:
        matches = input_ids.eq(marker_id)
        counts = matches.sum(dim=1)
        if not bool(torch.all(counts == 1)):
            invalid = (counts != 1).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Every example must contain one {name} marker; bad rows={invalid}")
        return matches.to(dtype=torch.long).argmax(dim=1)

    @staticmethod
    def _gather(sequence: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(sequence.shape[0], device=sequence.device)
        return sequence[batch, positions]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> RelationClassifierOutput:
        if self.config.e1_start_token_id is None or self.config.e2_start_token_id is None:
            raise ValueError("Entity marker token ids are missing from model config")
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_inputs, **kwargs)
        sequence = outputs.last_hidden_state
        cls_values = sequence[:, 0]
        e1_positions = self._marker_positions(
            input_ids, self.config.e1_start_token_id, "[E1]"
        )
        e2_positions = self._marker_positions(
            input_ids, self.config.e2_start_token_id, "[E2]"
        )
        representation = torch.cat(
            (
                cls_values,
                self._gather(sequence, e1_positions),
                self._gather(sequence, e2_positions),
            ),
            dim=-1,
        )
        logits = self.classifier(representation)
        loss = None
        if labels is not None:
            positive_weights = logits.new_tensor(self.config.positive_class_weights)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels.to(dtype=logits.dtype),
                pos_weight=positive_weights,
            )
        return RelationClassifierOutput(loss=loss, logits=logits)


def build_rubert_relation_classifier(
    pretrained_name: str, **options: Any
) -> RelationClassifierModel:
    return RelationClassifierModel.from_encoder_pretrained(pretrained_name, **options)


__all__ = [
    "RelationClassifierConfig",
    "RelationClassifierModel",
    "RelationClassifierOutput",
    "build_rubert_relation_classifier",
]
