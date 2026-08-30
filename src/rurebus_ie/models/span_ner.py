"""RuBERT encoder with an endpoint/mean-pooling Span NER head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rurebus_ie.data.span_labels import SPAN_ID2LABEL, SPAN_LABEL2ID

try:
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
    from transformers.utils import ModelOutput
except (ImportError, OSError) as error:  # pragma: no cover - dependency error path
    raise RuntimeError(
        "torch and transformers are required to import the Span NER model"
    ) from error


class SpanNerConfig(PretrainedConfig):
    """Serializable configuration for :class:`SpanNerModel`."""

    model_type = "rurebus_span_ner"

    def __init__(
        self,
        *,
        encoder_config: dict[str, Any] | None = None,
        max_span_width: int = 32,
        width_embedding_dim: int = 32,
        span_hidden_size: int = 512,
        dropout: float = 0.1,
        none_loss_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_labels", len(SPAN_LABEL2ID))
        kwargs.setdefault("label2id", SPAN_LABEL2ID)
        kwargs.setdefault("id2label", SPAN_ID2LABEL)
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or {}
        self.max_span_width = int(max_span_width)
        self.width_embedding_dim = int(width_embedding_dim)
        self.span_hidden_size = int(span_hidden_size)
        self.dropout = float(dropout)
        self.none_loss_weight = float(none_loss_weight)


@dataclass
class SpanNerOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class SpanNerModel(PreTrainedModel):
    """Classify every candidate span as an entity type or ``NONE``.

    A span representation concatenates its start token, end token, mean-pooled
    internal tokens and a learned word-width embedding.
    """

    config_class = SpanNerConfig
    base_model_prefix = "encoder"

    def __init__(self, config: SpanNerConfig) -> None:
        super().__init__(config)
        if not config.encoder_config:
            raise ValueError("SpanNerConfig.encoder_config must contain a base model config")
        encoder_values = dict(config.encoder_config)
        model_type = encoder_values.pop("model_type", None)
        if not model_type:
            raise ValueError("encoder_config.model_type is required")
        encoder_config = AutoConfig.for_model(model_type, **encoder_values)
        self.encoder = AutoModel.from_config(encoder_config)
        hidden_size = int(encoder_config.hidden_size)
        self.width_embedding = nn.Embedding(
            config.max_span_width + 1,
            config.width_embedding_dim,
            padding_idx=0,
        )
        representation_size = hidden_size * 3 + config.width_embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(representation_size, config.span_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.span_hidden_size, config.num_labels),
        )
        self.post_init()

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
    ) -> "SpanNerModel":
        """Load pretrained RuBERT weights and initialize a fresh span head."""
        encoder_config = AutoConfig.from_pretrained(pretrained_name)
        if hasattr(encoder_config, "hidden_dropout_prob"):
            encoder_config.hidden_dropout_prob = dropout
        encoder = AutoModel.from_pretrained(pretrained_name, config=encoder_config)
        config = SpanNerConfig(
            encoder_config=encoder_config.to_dict(),
            max_span_width=max_span_width,
            width_embedding_dim=width_embedding_dim,
            span_hidden_size=span_hidden_size,
            dropout=dropout,
            none_loss_weight=none_loss_weight,
        )
        model = cls(config)
        model.encoder = encoder
        return model

    @staticmethod
    def _gather(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        hidden_size = sequence.shape[-1]
        return sequence.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden_size))

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
        **kwargs: Any,
    ) -> SpanNerOutput:
        if span_starts is None or span_ends is None or span_widths is None:
            raise ValueError("span_starts, span_ends and span_widths are required")
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_inputs, **kwargs)
        sequence = outputs.last_hidden_state

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
        representation = torch.cat(
            (start_values, end_values, mean_values, width_values), dim=-1
        )
        logits = self.classifier(representation)

        loss = None
        if labels is not None:
            weights = logits.new_ones(self.config.num_labels)
            weights[SPAN_LABEL2ID["NONE"]] = self.config.none_loss_weight
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.config.num_labels),
                labels.reshape(-1),
                weight=weights,
                ignore_index=-100,
            )
        return SpanNerOutput(loss=loss, logits=logits)


def build_rubert_span_classifier(pretrained_name: str, **options: Any) -> SpanNerModel:
    return SpanNerModel.from_encoder_pretrained(pretrained_name, **options)


__all__ = [
    "SpanNerConfig",
    "SpanNerModel",
    "SpanNerOutput",
    "build_rubert_span_classifier",
]
