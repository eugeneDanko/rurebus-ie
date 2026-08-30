import csv
import tempfile
import unittest
from pathlib import Path

from rurebus_ie.data.conflict_resolution import Correction, write_corrections
from rurebus_ie.data.dataset_versioning import (
    build_versioned_dataset,
    validate_versioned_dataset,
)
from rurebus_ie.data.preprocessing import file_sha256


class DatasetVersioningTest(unittest.TestCase):
    def _write_document(self, root: Path, split: str, name: str, label: str) -> None:
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.txt").write_text("рост", encoding="utf-8")
        (directory / f"{name}.ann").write_text(
            f"T1\t{label} 0 4\tрост\n", encoding="utf-8"
        )

    def _make_parent(self, root: Path) -> None:
        for split, name in (("train", "a"), ("validation", "b"), ("test", "c")):
            self._write_document(root, split, name, "BIN")
        with (root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "document_id",
                    "source_id",
                    "split",
                    "processed_txt_path",
                    "processed_ann_path",
                    "text_sha256",
                    "ann_sha256",
                ],
            )
            writer.writeheader()
            for split, name in (("train", "a"), ("validation", "b"), ("test", "c")):
                writer.writerow(
                    {
                        "document_id": name,
                        "source_id": name,
                        "split": split,
                        "processed_txt_path": f"parent/{split}/{name}.txt",
                        "processed_ann_path": f"parent/{split}/{name}.ann",
                        "text_sha256": file_sha256(root / split / f"{name}.txt"),
                        "ann_sha256": file_sha256(root / split / f"{name}.ann"),
                    }
                )

    def test_builds_only_accepted_corrections_and_writes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "parent"
            output = base / "versions" / "corrected_v1"
            self._make_parent(source)
            correction_path = base / "corrections.csv"
            corrections = [
                Correction(
                    split="train",
                    document_id="a",
                    entity_id="T1",
                    surface="рост",
                    old_type="BIN",
                    new_type="CMP",
                    confidence="HIGH",
                    reason="test",
                    guideline_basis="test",
                    decision_status="ACCEPTED",
                ),
                Correction(
                    split="validation",
                    document_id="b",
                    entity_id="T1",
                    surface="рост",
                    old_type="BIN",
                    new_type="CMP",
                    confidence="MEDIUM",
                    reason="test",
                    guideline_basis="test",
                    decision_status="REVIEW_REQUIRED",
                ),
            ]
            write_corrections(correction_path, corrections)

            report = build_versioned_dataset(
                source,
                output,
                corrections,
                correction_manifest_path=correction_path,
                dataset_version="corrected_v1",
                parent_version="original_v1",
            )

            self.assertEqual(report["applied_corrections"], 1)
            self.assertIn("CMP 0 4", (output / "train" / "a.ann").read_text())
            self.assertIn("BIN 0 4", (output / "validation" / "b.ann").read_text())
            self.assertTrue((output / "dataset_report.json").is_file())
            self.assertEqual(validate_versioned_dataset(output)["checks"], "passed")
            with self.assertRaises(FileExistsError):
                build_versioned_dataset(
                    source,
                    output,
                    corrections,
                    correction_manifest_path=correction_path,
                    dataset_version="corrected_v1",
                    parent_version="original_v1",
                )


if __name__ == "__main__":
    unittest.main()
