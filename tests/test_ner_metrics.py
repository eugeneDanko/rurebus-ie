import unittest

from rurebus_ie.data.ner_labels import LABEL2ID
from rurebus_ie.evaluation.ner_metrics import compute_strict_ner_metrics
from rurebus_ie.inference.ner_pipeline import (
    PredictedEntity,
    decode_bio_predictions,
    merge_window_predictions,
)


class NerMetricsTest(unittest.TestCase):
    def test_decodes_bio_and_repairs_initial_i_tag(self) -> None:
        text = "Рост инвестиций"
        entities = decode_bio_predictions(
            [-100, LABEL2ID["B-CMP"], LABEL2ID["I-CMP"], -100],
            [(0, 0), (0, 4), (5, 15), (0, 0)],
            text,
        )
        self.assertEqual((entities[0].entity_type, entities[0].start, entities[0].end), ("CMP", 0, 15))

        repaired = decode_bio_predictions(
            [LABEL2ID["I-MET"]], [(5, 15)], text
        )
        self.assertEqual(repaired[0].text, "инвестиций")

    def test_strict_metrics_require_type_and_exact_boundaries(self) -> None:
        predictions = {
            "doc": (
                PredictedEntity("Рост", "CMP", 0, 4, 0.9),
                PredictedEntity("инвестиций", "MET", 5, 15, 0.8),
            )
        }
        references = {"doc": (("CMP", 0, 4, "Рост"), ("MET", 5, 14, "инвестици"))}

        metrics = compute_strict_ner_metrics(predictions, references)

        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.micro_f1, 0.5)

    def test_window_merge_deduplicates_and_removes_contained_fragment(self) -> None:
        merged = merge_window_predictions(
            [
                PredictedEntity("инвестиций", "MET", 5, 15, 0.8),
                PredictedEntity("инвестиций", "MET", 5, 15, 0.9),
                PredictedEntity("вестиций", "MET", 7, 15, 0.95),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
