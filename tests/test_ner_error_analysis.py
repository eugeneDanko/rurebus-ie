from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rurebus_ie.data.brat_parser import BratDocument, Entity, TextSpan
from rurebus_ie.data.ner_dataset import RuReBusNerDataset, TokenizedNerExample
from rurebus_ie.evaluation.ner_error_analysis import (
    analyze_ner_predictions,
    write_ner_error_analysis,
)
from rurebus_ie.inference.ner_pipeline import PredictedEntity


class NerErrorAnalysisTest(unittest.TestCase):
    def test_classifies_exact_boundary_type_and_unmatched_errors(self) -> None:
        text = "Рост инвестиций доходы проект общество лишнее"
        document = BratDocument(
            document_id="doc",
            text=text,
            entities=(
                Entity("T1", "CMP", (TextSpan(0, 4),), "Рост"),
                Entity("T2", "MET", (TextSpan(5, 15),), "инвестиций"),
                Entity("T3", "MET", (TextSpan(16, 22),), "доходы"),
                Entity("T4", "ACT", (TextSpan(23, 29),), "проект"),
                Entity("T5", "SOC", (TextSpan(30, 38),), "общество"),
            ),
            relations=(),
        )
        example = TokenizedNerExample(
            document_id="doc",
            window_id=0,
            input_ids=(1, 2, 3, 4, 5, 6, 7, 8),
            attention_mask=(1, 1, 1, 1, 1, 1, 1, 1),
            labels=(-100, 0, 0, 0, 0, 0, 0, -100),
            offset_mapping=(
                (0, 0),
                (0, 4),
                (5, 15),
                (16, 22),
                (23, 29),
                (30, 38),
                (39, 45),
                (0, 0),
            ),
        )
        dataset = RuReBusNerDataset([example], [document])
        predictions = {
            "doc": (
                PredictedEntity("Рост", "CMP", 0, 4, 0.99),
                PredictedEntity("инвестици", "MET", 5, 14, 0.80),
                PredictedEntity("доходы", "ECO", 16, 22, 0.75),
                PredictedEntity("проек", "SOC", 23, 28, 0.70),
                PredictedEntity("лишнее", "BIN", 39, 45, 0.90),
            )
        }

        analysis = analyze_ner_predictions(dataset, predictions, edge_token_count=1)

        categories = analysis.summary["diagnostic_categories"]
        self.assertEqual(categories["true_positive"], 1)
        self.assertEqual(categories["boundary_error"], 1)
        self.assertEqual(categories["type_error"], 1)
        self.assertEqual(categories["boundary_and_type_error"], 1)
        self.assertEqual(categories["false_positive"], 1)
        self.assertEqual(categories["false_negative"], 1)
        self.assertEqual(
            analysis.summary["strict_counts"],
            {"true_positive": 1, "false_positive": 4, "false_negative": 4},
        )

        with TemporaryDirectory() as temporary_directory:
            output = write_ner_error_analysis(analysis, temporary_directory)
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "errors.csv").is_file())
            self.assertTrue((output / "type_confusion.csv").is_file())


if __name__ == "__main__":
    unittest.main()
