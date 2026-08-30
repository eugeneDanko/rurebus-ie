import csv
import tempfile
import unittest
from pathlib import Path

from rurebus_ie.data.conflict_resolution import (
    CORRECTION_FIELDS,
    Correction,
    apply_corrections,
    collect_occurrences,
    find_surface_conflicts,
    load_dataset,
    read_corrections,
)


class ConflictResolutionTest(unittest.TestCase):
    def _write_document(self, root: Path, split: str, name: str, label: str) -> None:
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.txt").write_text("рост", encoding="utf-8")
        (directory / f"{name}.ann").write_text(
            f"T1\t{label} 0 4\tрост\n", encoding="utf-8"
        )

    def _make_dataset(self, root: Path) -> None:
        self._write_document(root, "train", "a", "CMP")
        self._write_document(root, "validation", "b", "MET")
        self._write_document(root, "test", "c", "CMP")

    def test_conflict_detection_and_safe_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "corrected"
            self._make_dataset(root)

            documents = load_dataset(root)
            conflicts = find_surface_conflicts(collect_occurrences(documents))
            self.assertEqual(set(conflicts), {"рост"})

            correction = Correction(
                split="validation",
                document_id="b",
                entity_id="T1",
                surface="рост",
                old_type="MET",
                new_type="CMP",
                confidence="HIGH",
                reason="test",
                guideline_basis="test",
            )
            self.assertEqual(apply_corrections(root, output, [correction]), 1)
            self.assertIn("MET 0 4", (root / "validation" / "b.ann").read_text())
            self.assertIn("CMP 0 4", (output / "validation" / "b.ann").read_text())

    def test_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrections.csv"
            row = {
                "split": "train",
                "document_id": "a",
                "entity_id": "T1",
                "surface": "рост",
                "old_type": "MET",
                "new_type": "CMP",
                "confidence": "HIGH",
                "reason": "test",
                "guideline_basis": "test",
            }
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CORRECTION_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            self.assertEqual(read_corrections(path), [Correction(**row)])


if __name__ == "__main__":
    unittest.main()
