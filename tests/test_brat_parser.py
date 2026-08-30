from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rurebus_ie.data.brat_parser import load_brat_document, validate_round_trip


class BratParserTest(unittest.TestCase):
    def test_load_and_round_trip_minimal_document(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            txt_path = temp_path / "sample.txt"
            ann_path = temp_path / "sample.ann"
            txt_path.write_text("Рост инвестиций", encoding="utf-8")
            ann_path.write_text(
                "T1\tCMP 0 4\tРост\n"
                "T2\tMET 5 15\tинвестиций\n"
                "R1\tNPS Arg1:T1 Arg2:T2\n",
                encoding="utf-8",
            )

            document = load_brat_document(txt_path, ann_path)

            self.assertEqual(document.document_id, "sample")
            self.assertEqual(
                [entity.entity_type for entity in document.entities],
                ["CMP", "MET"],
            )
            self.assertEqual(document.relations[0].relation_type, "NPS")
            validate_round_trip(document)


if __name__ == "__main__":
    unittest.main()
