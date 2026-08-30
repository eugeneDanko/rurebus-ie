from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import unittest

import torch
from transformers import BertConfig

from rurebus_ie.data.brat_parser import BratDocument, Entity, Relation, TextSpan
from rurebus_ie.data.relation_dataset import build_relation_examples
from rurebus_ie.data.relation_labels import RELATION_LABEL2ID
from rurebus_ie.evaluation.relation_metrics import compute_relation_metrics
from rurebus_ie.inference.relation_pipeline import PredictedRelation
from rurebus_ie.models.relation_classifier import (
    RelationClassifierConfig,
    RelationClassifierModel,
)
from rurebus_ie.training.relation_trainer import RelationTrainer


def tiny_document() -> BratDocument:
    text = "Компания увеличила доход"
    return BratDocument(
        document_id="sample",
        text=text,
        entities=(
            Entity("T1", "CMP", (TextSpan(0, 8),), "Компания"),
            Entity("T2", "ACT", (TextSpan(9, 18),), "увеличила"),
            Entity("T3", "ECO", (TextSpan(19, 24),), "доход"),
        ),
        relations=(
            Relation("R1", "TSK", "T1", "T2"),
            Relation("R2", "NNG", "T1", "T2"),
        ),
    )


def tiny_model() -> RelationClassifierModel:
    encoder = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return RelationClassifierModel(
        RelationClassifierConfig(
            encoder_config=encoder.to_dict(),
            relation_hidden_size=12,
            dropout=0.0,
            e1_start_token_id=2,
            e2_start_token_id=3,
        )
    )


class RelationClassifierTest(unittest.TestCase):
    def test_dataset_preserves_multilabel_pair(self) -> None:
        dataset = build_relation_examples(
            [tiny_document()],
            max_pair_distance=32,
            context_margin=0,
            negative_to_positive_ratio=0,
        )
        self.assertEqual(len(dataset), 1)
        example = dataset[0]
        self.assertEqual(example.labels[RELATION_LABEL2ID["TSK"]], 1.0)
        self.assertEqual(example.labels[RELATION_LABEL2ID["NNG"]], 1.0)
        self.assertIn("[E1]Компания[/E1]", example.marked_text)
        self.assertEqual(dataset.candidate_recall, 1.0)

    def test_model_forward_and_checkpoint_round_trip(self) -> None:
        model = tiny_model()
        input_ids = torch.tensor([[1, 2, 5, 6, 3, 7, 8]])
        labels = torch.zeros((1, len(RELATION_LABEL2ID)))
        labels[0, RELATION_LABEL2ID["TSK"]] = 1.0
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            token_type_ids=torch.zeros_like(input_ids),
            labels=labels,
        )
        self.assertEqual(tuple(output.logits.shape), (1, len(RELATION_LABEL2ID)))
        self.assertTrue(torch.isfinite(output.loss))
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = RelationClassifierModel.from_pretrained(directory)
            self.assertEqual(restored.config.e1_start_token_id, 2)
            del restored
            gc.collect()

    def test_strict_metrics_deduplicate_annotation_duplicates(self) -> None:
        predictions = (
            PredictedRelation("sample", "TSK", "T1", "T2", 0.9),
            PredictedRelation("sample", "NNG", "T1", "T2", 0.8),
        )
        references = (
            ("sample", "TSK", "T1", "T2"),
            ("sample", "TSK", "T1", "T2"),
            ("sample", "NNG", "T1", "T2"),
        )
        metrics = compute_relation_metrics(predictions, references)
        self.assertEqual(metrics.micro_f1, 1.0)

    def test_one_epoch_training_saves_relation_checkpoint(self) -> None:
        labels = torch.zeros((1, len(RELATION_LABEL2ID)))
        labels[0, RELATION_LABEL2ID["TSK"]] = 1.0
        batch = {
            "input_ids": torch.tensor([[1, 2, 5, 3, 7]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "token_type_ids": torch.zeros((1, 5), dtype=torch.long),
            "labels": labels,
            "document_id": ["sample"],
            "arg1_id": ["T1"],
            "arg2_id": ["T2"],
            "arg1_signature": [("CMP", 0, 8)],
            "arg2_signature": [("ACT", 9, 18)],
        }

        class TinyRelationDataset:
            gold_relations = (("sample", "TSK", "T1", "T2"),)
            candidate_recall = 1.0

            def __len__(self):
                return 1

        class TinyLoader:
            dataset = TinyRelationDataset()

            def __len__(self):
                return 1

            def __iter__(self):
                return iter((batch,))

        with TemporaryDirectory() as directory:
            config = {
                "experiment": {"seed": 42, "output_dir": directory},
                "training": {
                    "epochs": 1,
                    "encoder_learning_rate": 1e-3,
                    "head_learning_rate": 1e-3,
                    "gradient_accumulation_steps": 1,
                    "mixed_precision": False,
                    "early_stopping_min_epochs": 1,
                    "early_stopping_patience": 1,
                },
                "decoding": {"confidence_threshold": 0.5},
            }
            summary = RelationTrainer(
                tiny_model(), config, device="cpu"
            ).fit(TinyLoader(), TinyLoader())
            self.assertEqual(summary.best_epoch, 1)
            self.assertTrue((summary.checkpoint_dir / "config.json").is_file())
            self.assertTrue((Path(directory) / "validation_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
