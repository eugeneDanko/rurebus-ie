from pathlib import Path
import unittest

from rurebus_ie.data.protocol_split import validate_global_test_protocol


class ProtocolSplitTest(unittest.TestCase):
    def test_checked_in_protocol_is_locked_and_disjoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = root / "rurebus_data/versions/corrected_v1/global_v1_manifest.csv"
        report = root / "rurebus_data/versions/corrected_v1/global_v1_report.json"
        if not manifest.is_file() or not report.is_file():
            self.skipTest("corrected_v1 data artifacts are not present")
        result = validate_global_test_protocol(manifest, report)
        self.assertTrue(result["safeguards"]["source_groups_disjoint"])
        self.assertTrue(result["safeguards"]["previous_validation_excluded_from_global_test"])


if __name__ == "__main__":
    unittest.main()
