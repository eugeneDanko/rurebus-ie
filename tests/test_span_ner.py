from pathlib import Path
from tempfile import TemporaryDirectory
import gc
from types import MethodType, SimpleNamespace
import unittest

import torch
from transformers import BertConfig

from rurebus_ie.data.brat_parser import BratDocument, Entity, TextSpan
from rurebus_ie.data.span_collator import SpanClassificationCollator
from rurebus_ie.data.span_dataset import (
    GoldTokenSpan,
    RuReBusSpanDataset,
    TokenizedSpanExample,
    build_span_examples,
)
from rurebus_ie.data.span_labels import SPAN_LABEL2ID
from rurebus_ie.evaluation.ner_error_analysis import analyze_ner_predictions
from rurebus_ie.inference.span_pipeline import (
    decode_span_predictions,
    merge_span_predictions,
)
from rurebus_ie.inference.ner_pipeline import PredictedEntity
from rurebus_ie.models.span_ner import SpanNerConfig, SpanNerModel
from rurebus_ie.training.span_trainer import SpanNerTrainer


class TinyBatchEncoding(dict):
    def __init__(self, *args, word_ids, **kwargs):
        super().__init__(*args, **kwargs)
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids[batch_index]


class TinyTokenizer:
    is_fast = True
    padding_side = "right"

    def __call__(self, text, **kwargs):
        return TinyBatchEncoding(
            {
                "input_ids": [[1, 2, 3, 4]],
                "attention_mask": [[1, 1, 1, 1]],
                "token_type_ids": [[0, 0, 0, 0]],
                "offset_mapping": [[(0, 0), (0, 5), (6, 14), (0, 0)]],
            },
            word_ids=[[None, 0, 1, None]],
        )

    def pad(self, features, padding, return_tensors):
        width = max(len(feature["input_ids"]) for feature in features)
        return {
            key: torch.tensor(
                [feature[key] + [0] * (width - len(feature[key])) for feature in features],
                dtype=torch.long,
            )
            for key in features[0]
        }

    def save_pretrained(self, path):
        Path(path, "tokenizer.test").write_text("ok", encoding="utf-8")


def tiny_example() -> TokenizedSpanExample:
    return TokenizedSpanExample(
        document_id="sample",
        window_id=0,
        input_ids=(1, 2, 3, 4),
        attention_mask=(1, 1, 1, 1),
        token_type_ids=(0, 0, 0, 0),
        offset_mapping=((0, 0), (0, 5), (6, 14), (0, 0)),
        word_ids=(None, 0, 1, None),
        gold_spans=(
            GoldTokenSpan(0 + 1, 2, SPAN_LABEL2ID["CMP"], 0, 14),
        ),
    )


def tiny_model() -> SpanNerModel:
    encoder = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return SpanNerModel(
        SpanNerConfig(
            encoder_config=encoder.to_dict(),
            max_span_width=2,
            width_embedding_dim=4,
            span_hidden_size=12,
            dropout=0.0,
        )
    )


class SpanNerTest(unittest.TestCase):
    def test_dataset_aligns_gold_to_word_boundaries(self) -> None:
        document = BratDocument(
            document_id="sample",
            text="Новая компания",
            entities=(Entity("T1", "CMP", (TextSpan(0, 14),), "Новая компания"),),
            relations=(),
        )
        dataset = build_span_examples(
            [document], TinyTokenizer(), max_length=16, stride=0, max_span_width=2
        )
        self.assertEqual(dataset[0].gold_spans[0].start_token, 1)
        self.assertEqual(dataset[0].gold_spans[0].end_token, 2)
        self.assertFalse(dataset.unrepresentable_entities)

    def test_collator_keeps_positive_span(self) -> None:
        batch = SpanClassificationCollator(
            TinyTokenizer(), max_span_width=2, training=True, max_negative_spans=1
        )([tiny_example()])
        active_labels = batch["labels"][batch["span_mask"]].tolist()
        self.assertIn(SPAN_LABEL2ID["CMP"], active_labels)
        self.assertLessEqual(len(active_labels), 2)

    def test_model_forward_and_checkpoint_round_trip(self) -> None:
        batch = SpanClassificationCollator(
            TinyTokenizer(), max_span_width=2, training=False
        )([tiny_example()])
        model = tiny_model()
        model_inputs = {
            key: value
            for key, value in batch.items()
            if key in {
                "input_ids", "attention_mask", "token_type_ids", "span_starts",
                "span_ends", "span_widths", "span_mask", "labels"
            }
        }
        output = model(**model_inputs)
        self.assertEqual(tuple(output.logits.shape), (1, 3, len(SPAN_LABEL2ID)))
        self.assertTrue(torch.isfinite(output.loss))
        with TemporaryDirectory() as temporary_directory:
            model.save_pretrained(temporary_directory)
            restored = SpanNerModel.from_pretrained(temporary_directory)
            self.assertEqual(restored.config.max_span_width, 2)
            # Safetensors can keep a memory-mapped file open on Windows until
            # the loaded model is released.
            del restored
            gc.collect()

    def test_one_epoch_training_saves_checkpoint(self) -> None:
        document = BratDocument(
            document_id="sample",
            text="Новая компания",
            entities=(Entity("T1", "CMP", (TextSpan(0, 14),), "Новая компания"),),
            relations=(),
        )
        dataset = RuReBusSpanDataset([tiny_example()], [document])
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=SpanClassificationCollator(
                TinyTokenizer(), max_span_width=2, training=False
            ),
        )
        with TemporaryDirectory() as temporary_directory:
            config = {
                "experiment": {"seed": 42, "output_dir": temporary_directory},
                "training": {
                    "epochs": 1,
                    "encoder_learning_rate": 1e-3,
                    "head_learning_rate": 1e-3,
                    "gradient_accumulation_steps": 1,
                    "mixed_precision": False,
                    "early_stopping_patience": 1,
                },
                "decoding": {"confidence_threshold": 0.0},
            }
            summary = SpanNerTrainer(
                tiny_model(), config, tokenizer=TinyTokenizer(), device="cpu"
            ).fit(loader, loader)
            self.assertEqual(summary.best_epoch, 1)
            self.assertTrue((summary.checkpoint_dir / "config.json").is_file())
            self.assertTrue((Path(temporary_directory) / "history.csv").is_file())

    def test_decoder_suppresses_overlap_by_confidence(self) -> None:
        decoded = decode_span_predictions(
            [SPAN_LABEL2ID["CMP"], SPAN_LABEL2ID["ACT"]],
            [(0, 14), (6, 14)],
            "Новая компания",
            confidences=[0.9, 0.8],
            confidence_threshold=0.0,
        )
        merged = merge_span_predictions(decoded)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].entity_type, "CMP")

    def test_threshold_calibration_selects_best_micro_f1_in_one_pass(self) -> None:
        document = BratDocument(
            document_id="sample",
            text="Новая компания выросла",
            entities=(Entity("T1", "CMP", (TextSpan(0, 14),), "Новая компания"),),
            relations=(),
        )
        dataset = RuReBusSpanDataset([tiny_example()], [document])
        loader = SimpleNamespace(dataset=dataset)
        trainer = SpanNerTrainer(tiny_model(), {"decoding": {"allow_overlapping": False}})
        calls = []

        def fake_collect(
            self, data_loader, *, minimum_confidence, include_labels=True
        ):
            calls.append((minimum_confidence, include_labels))
            return 0.0, {
                "sample": [
                    PredictedEntity("Новая компания", "CMP", 0, 14, 0.8),
                    PredictedEntity("выросла", "ACT", 15, 22, 0.4),
                ]
            }

        trainer._collect_candidates = MethodType(fake_collect, trainer)
        calibration = trainer.calibrate_thresholds(loader, [0.3, 0.5, 0.9])

        self.assertEqual(calls, [(0.3, False)])
        self.assertEqual(calibration.best_threshold, 0.5)
        self.assertEqual(calibration.best_metrics.micro_f1, 1.0)

    def test_class_threshold_calibration_uses_validation_candidates_once(self) -> None:
        document = BratDocument(
            document_id="sample",
            text="Компания действует быстро",
            entities=(
                Entity("T1", "CMP", (TextSpan(0, 8),), "Компания"),
                Entity("T2", "ACT", (TextSpan(9, 17),), "действует"),
            ),
            relations=(),
        )
        dataset = RuReBusSpanDataset([tiny_example()], [document])
        loader = SimpleNamespace(dataset=dataset)
        trainer = SpanNerTrainer(tiny_model(), {"decoding": {"allow_overlapping": False}})
        calls = []

        def fake_collect(
            self, data_loader, *, minimum_confidence, include_labels=True
        ):
            calls.append((minimum_confidence, include_labels))
            return 0.0, {
                "sample": [
                    PredictedEntity("Компания", "CMP", 0, 8, 0.8),
                    PredictedEntity("действует", "ACT", 9, 17, 0.8),
                    PredictedEntity("быстро", "ACT", 18, 24, 0.6),
                ]
            }

        trainer._collect_candidates = MethodType(fake_collect, trainer)
        calibration = trainer.calibrate_class_thresholds(
            loader,
            [0.5, 0.7, 0.9],
            initial_threshold=0.5,
            max_rounds=2,
        )

        self.assertEqual(calls, [(0.5, False)])
        self.assertEqual(calibration.class_thresholds["ACT"], 0.7)
        self.assertEqual(calibration.best_metrics.micro_f1, 1.0)

        analysis = analyze_ner_predictions(
            dataset,
            {
                "sample": (
                    PredictedEntity("Компания", "CMP", 0, 8, 0.8),
                    PredictedEntity("действует", "ACT", 9, 17, 0.8),
                )
            },
        )
        self.assertEqual(analysis.summary["strict_counts"]["true_positive"], 2)


if __name__ == "__main__":
    unittest.main()
