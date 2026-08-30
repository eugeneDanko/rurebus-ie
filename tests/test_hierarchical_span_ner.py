from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig

from rurebus_ie.data.brat_parser import BratDocument, Entity, TextSpan
from rurebus_ie.data.span_collator import SpanClassificationCollator
from rurebus_ie.data.span_dataset import (
    GoldTokenSpan,
    RuReBusSpanDataset,
    TokenizedSpanExample,
)
from rurebus_ie.data.span_hierarchy import build_span_label_hierarchy
from rurebus_ie.data.span_labels import SPAN_LABEL2ID
from rurebus_ie.models.hierarchical_span_ner import (
    HierarchicalSpanNerConfig,
    HierarchicalSpanNerModel,
)
from rurebus_ie.evaluation.ner_metrics import NerMetrics
from rurebus_ie.training.hierarchical_span_experiment import (
    calibrate_hierarchical_span_threshold_experiment,
)
from rurebus_ie.training.hierarchical_span_trainer import (
    HierarchicalSpanNerTrainer,
)
from rurebus_ie.training.span_trainer import SpanThresholdCalibration


class TinyTokenizer:
    padding_side = "right"

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
            GoldTokenSpan(1, 2, SPAN_LABEL2ID["CMP"], 0, 14),
        ),
    )


def tiny_model() -> HierarchicalSpanNerModel:
    encoder = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return HierarchicalSpanNerModel(
        HierarchicalSpanNerConfig(
            encoder_config=encoder.to_dict(),
            max_span_width=2,
            width_embedding_dim=4,
            span_hidden_size=12,
            contrastive_projection_size=8,
            dropout=0.0,
        )
    )


class HierarchicalSpanNerTest(unittest.TestCase):
    def test_hierarchy_preserves_original_labels(self) -> None:
        hierarchy = build_span_label_hierarchy()
        self.assertEqual(hierarchy.fine_to_coarse_labels["ACT"], "ACT+BIN")
        self.assertEqual(hierarchy.fine_to_coarse_labels["BIN"], "ACT+BIN")
        self.assertEqual(hierarchy.fine_to_coarse_labels["CMP"], "CMP+QUA")
        self.assertEqual(set(SPAN_LABEL2ID), {
            "NONE", "MET", "ECO", "BIN", "CMP", "QUA", "ACT", "INST", "SOC"
        })

    def test_both_stages_return_fine_logits_and_finite_losses(self) -> None:
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

        model.set_training_stage("coarse")
        coarse = model(**model_inputs)
        self.assertEqual(tuple(coarse.logits.shape), (1, 3, len(SPAN_LABEL2ID)))
        self.assertEqual(tuple(coarse.coarse_logits.shape), (1, 3, 6))
        self.assertIsNone(coarse.fine_loss)
        self.assertTrue(torch.isfinite(coarse.loss))

        model.set_training_stage("fine")
        fine = model(**model_inputs)
        self.assertIsNotNone(fine.fine_loss)
        self.assertTrue(torch.isfinite(fine.loss))
        self.assertTrue(torch.isfinite(fine.contrastive_loss))

        inference = model(
            **{key: value for key, value in model_inputs.items() if key != "labels"}
        )
        self.assertIsNone(inference.loss)
        self.assertIsNone(inference.fine_loss)
        self.assertIsNone(inference.coarse_loss)
        self.assertIsNone(inference.contrastive_loss)

    def test_two_stage_training_saves_only_fine_best_checkpoint(self) -> None:
        document = BratDocument(
            document_id="sample",
            text="Новая компания",
            entities=(
                Entity("T1", "CMP", (TextSpan(0, 14),), "Новая компания"),
            ),
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
                    "coarse_epochs": 1,
                    "fine_epochs": 1,
                    "encoder_learning_rate": 1e-3,
                    "head_learning_rate": 1e-3,
                    "gradient_accumulation_steps": 1,
                    "mixed_precision": False,
                    "fine_early_stopping_min_epochs": 1,
                    "early_stopping_patience": 1,
                },
                "decoding": {"confidence_threshold": 0.0},
            }
            summary = HierarchicalSpanNerTrainer(
                tiny_model(), config, tokenizer=TinyTokenizer(), device="cpu"
            ).fit(loader, loader)
            self.assertEqual([row["stage"] for row in summary.history], ["coarse", "fine"])
            self.assertEqual(summary.best_epoch, 2)
            self.assertTrue((summary.checkpoint_dir / "config.json").is_file())
            self.assertFalse(
                (Path(temporary_directory) / "checkpoints" / "coarse").exists()
            )
            restored = HierarchicalSpanNerModel.from_pretrained(
                summary.checkpoint_dir
            )
            self.assertEqual(restored.config.training_stage, "fine")
            self.assertEqual(restored.config.label2id, SPAN_LABEL2ID)
            # Regression: a former non-persistent fine-to-coarse buffer was
            # materialized with garbage values by meta-device checkpoint
            # loading and triggered a CUDA device-side assert on evaluation.
            restored_batch = next(iter(loader))
            restored_inputs = {
                key: value
                for key, value in restored_batch.items()
                if key in {
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                    "span_starts",
                    "span_ends",
                    "span_widths",
                    "span_mask",
                    "labels",
                }
            }
            restored_output = restored(**restored_inputs)
            self.assertTrue(torch.isfinite(restored_output.loss))
            active_coarse = restored._coarse_labels(restored_batch["labels"])
            self.assertEqual(
                active_coarse[restored_batch["labels"] != -100].tolist(),
                [0, 2, 0],
            )
            del active_coarse, restored_output, restored_inputs, restored_batch, restored
            gc.collect()

    def test_calibration_workflow_writes_reproducibility_artifacts(self) -> None:
        metrics = NerMetrics(
            precision=0.6,
            recall=0.5,
            micro_f1=0.5454545454545454,
            macro_f1=0.5,
            per_class={},
        )
        calibration = SpanThresholdCalibration(
            best_threshold=0.7,
            best_metrics=metrics,
            rows=(
                {
                    "threshold": 0.7,
                    "precision": 0.6,
                    "recall": 0.5,
                    "micro_f1": metrics.micro_f1,
                    "macro_f1": metrics.macro_f1,
                    "predicted_entities": 10,
                },
            ),
        )
        trainer = SimpleNamespace(
            calibrate_thresholds=lambda loader, thresholds: calibration
        )
        with TemporaryDirectory() as temporary_directory:
            Path(temporary_directory, "checkpoints", "best").mkdir(parents=True)
            bundle = {"experiment": {"output_dir": temporary_directory}}
            with (
                patch(
                    "rurebus_ie.training.hierarchical_span_experiment.load_experiment_bundle",
                    return_value=bundle,
                ),
                patch(
                    "rurebus_ie.training.hierarchical_span_experiment._checkpoint_evaluator",
                    return_value=(trainer, object()),
                ),
            ):
                result = calibrate_hierarchical_span_threshold_experiment(
                    "experiment.yaml", thresholds=[0.7]
                )

            self.assertEqual(result.best_threshold, 0.7)
            output = Path(temporary_directory)
            self.assertTrue((output / "threshold_calibration.csv").is_file())
            payload = json.loads(
                (output / "threshold_calibration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["prediction_head"], "fine")
            self.assertEqual(payload["best_threshold"], 0.7)


if __name__ == "__main__":
    unittest.main()
