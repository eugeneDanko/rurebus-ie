from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rurebus_ie.data.brat_parser import BratDocument, Entity, TextSpan
from rurebus_ie.data.ner_dataset import build_ner_examples, load_documents_from_manifest
from rurebus_ie.data.ner_labels import LABEL2ID


class FakeFastTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [[101, 10, 20, 21, 102]],
            "attention_mask": [[1, 1, 1, 1, 1]],
            "token_type_ids": [[0, 0, 0, 0, 0]],
            "offset_mapping": [[(0, 0), (0, 4), (5, 8), (8, 15), (0, 0)]],
        }


class NerDatasetTest(unittest.TestCase):
    def test_aligns_character_entities_to_subtoken_bio_labels(self) -> None:
        document = BratDocument(
            document_id="sample",
            text="Рост инвестиций",
            entities=(
                Entity("T1", "CMP", (TextSpan(0, 4),), "Рост"),
                Entity("T2", "MET", (TextSpan(5, 15),), "инвестиций"),
            ),
            relations=(),
        )
        dataset = build_ner_examples([document], FakeFastTokenizer())

        self.assertEqual(
            dataset[0].labels,
            (
                -100,
                LABEL2ID["B-CMP"],
                LABEL2ID["B-MET"],
                LABEL2ID["I-MET"],
                -100,
            ),
        )
        self.assertEqual(dataset.gold_entities["sample"][1][:3], ("MET", 5, 15))

    def test_loads_processed_paths_relative_to_data_root(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            split_dir = data_root / "processed" / "train"
            split_dir.mkdir(parents=True)
            (split_dir / "sample.txt").write_text("Рост", encoding="utf-8")
            (split_dir / "sample.ann").write_text(
                "T1\tCMP 0 4\tРост\n", encoding="utf-8"
            )
            manifest = data_root / "processed" / "manifest.csv"
            manifest.write_text(
                "document_id,split,processed_txt_path,processed_ann_path\n"
                "sample,train,processed/train/sample.txt,processed/train/sample.ann\n",
                encoding="utf-8",
            )

            documents = load_documents_from_manifest(manifest, "train")

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].entities[0].entity_type, "CMP")


if __name__ == "__main__":
    unittest.main()
