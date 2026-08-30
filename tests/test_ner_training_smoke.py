from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from rurebus_ie.data.brat_parser import BratDocument, Entity, TextSpan
from rurebus_ie.data.collators import TokenClassificationCollator
from rurebus_ie.data.ner_dataset import RuReBusNerDataset, TokenizedNerExample
from rurebus_ie.data.ner_labels import ID2LABEL, LABEL2ID
from rurebus_ie.training.ner_trainer import NerTrainer


class TinyTokenizer:
    padding_side = "right"

    def pad(self, features, padding, return_tensors):
        width = max(len(feature["input_ids"]) for feature in features)
        result = {}
        for key in features[0]:
            pad_value = 0
            result[key] = torch.tensor(
                [feature[key] + [pad_value] * (width - len(feature[key])) for feature in features],
                dtype=torch.long,
            )
        return result

    def save_pretrained(self, path):
        Path(path, "tokenizer.test").write_text("ok", encoding="utf-8")


class TinyTokenClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.classifier = torch.nn.Linear(8, len(ID2LABEL))
        self.config = SimpleNamespace(id2label=ID2LABEL)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        logits = self.classifier(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path, "model.test.pt"))


class NerTrainingSmokeTest(unittest.TestCase):
    def test_one_epoch_saves_best_checkpoint_and_history(self) -> None:
        document = BratDocument(
            document_id="sample",
            text="Рост",
            entities=(Entity("T1", "CMP", (TextSpan(0, 4),), "Рост"),),
            relations=(),
        )
        example = TokenizedNerExample(
            document_id="sample",
            window_id=0,
            input_ids=(1, 2, 3),
            attention_mask=(1, 1, 1),
            labels=(-100, LABEL2ID["B-CMP"], -100),
            offset_mapping=((0, 0), (0, 4), (0, 0)),
        )
        dataset = RuReBusNerDataset([example], [document])
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=TokenClassificationCollator(TinyTokenizer()),
        )

        with TemporaryDirectory() as temporary_directory:
            config = {
                "experiment": {"seed": 42, "output_dir": temporary_directory},
                "training": {
                    "epochs": 1,
                    "learning_rate": 1e-3,
                    "gradient_accumulation_steps": 1,
                    "mixed_precision": False,
                    "early_stopping_patience": 1,
                },
            }
            summary = NerTrainer(
                TinyTokenClassifier(), config, tokenizer=TinyTokenizer(), device="cpu"
            ).fit(loader, loader)

            self.assertEqual(summary.best_epoch, 1)
            self.assertTrue((summary.checkpoint_dir / "model.test.pt").is_file())
            self.assertTrue((Path(temporary_directory) / "history.csv").is_file())


if __name__ == "__main__":
    unittest.main()
