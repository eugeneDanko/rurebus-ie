import unittest

from rurebus_ie.data.ner_dataset import RuReBusNerDataset, TokenizedNerExample
from rurebus_ie.data.ner_labels import BIO_LABELS, ID2LABEL, LABEL2ID


class NerSchemaTest(unittest.TestCase):
    def test_bio_schema_has_expected_size_and_round_trip(self) -> None:
        self.assertEqual(len(BIO_LABELS), 17)
        self.assertEqual(BIO_LABELS[0], "O")
        self.assertEqual(BIO_LABELS[-1], "I-SOC")
        self.assertEqual({ID2LABEL[index] for index in LABEL2ID.values()}, set(BIO_LABELS))

    def test_tokenized_example_validates_aligned_lengths(self) -> None:
        example = TokenizedNerExample(
            document_id="sample",
            window_id=0,
            input_ids=(101, 102),
            attention_mask=(1, 1),
            labels=(-100, -100),
            offset_mapping=((0, 0), (0, 0)),
        )

        dataset = RuReBusNerDataset([example])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0], example)


if __name__ == "__main__":
    unittest.main()

