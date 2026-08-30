from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rurebus_ie.training.artifacts import (
    assert_output_is_unlocked,
    register_baseline_run,
)


class BaselineArtifactsTest(unittest.TestCase):
    def test_registers_hashes_and_locks_completed_baseline(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "results" / "run"
            checkpoint = output / "checkpoints" / "best"
            checkpoint.mkdir(parents=True)
            data = root / "data" / "processed"
            data.mkdir(parents=True)
            files = {
                output / "config.yaml": "experiment: {}\n",
                output / "history.csv": "epoch,train_loss\n1,1.0\n",
                output / "train_metrics.json": "{}",
                output / "validation_metrics.json": "{}",
                output / "test_metrics.json": "{}",
                output / "official_original_test_metrics.json": "{}",
                output / "official_original_test_predictions.jsonl": "",
                output / "predictions.jsonl": "",
                checkpoint / "config.json": "{}",
                checkpoint / "model.safetensors": "weights",
                data / "manifest.csv": "document_id,split\n",
                data / "preprocessing_report.json": "{}",
            }
            for path, content in files.items():
                path.write_text(content, encoding="utf-8")

            record = register_baseline_run(
                output,
                alias="B0",
                manifest_path=data / "manifest.csv",
            )

            self.assertEqual(record["alias"], "B0")
            self.assertTrue((output / "baseline_record.json").is_file())
            artifact_paths = {artifact["path"] for artifact in record["artifacts"]}
            self.assertIn("official_original_test_metrics.json", artifact_paths)
            self.assertIn("official_original_test_predictions.jsonl", artifact_paths)
            self.assertTrue((output / ".baseline_locked").is_file())
            with self.assertRaises(FileExistsError):
                assert_output_is_unlocked(output)
            self.assertEqual(
                register_baseline_run(
                    output,
                    alias="B0",
                    manifest_path=data / "manifest.csv",
                )["registered_at_utc"],
                record["registered_at_utc"],
            )


if __name__ == "__main__":
    unittest.main()
