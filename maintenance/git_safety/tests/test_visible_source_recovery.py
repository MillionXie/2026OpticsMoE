import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maintenance.storage.check_visible_source_recovery import digest, inspect


class VisibleSourceRecoveryTests(unittest.TestCase):
    def audit(self, payload, blob, old_sha=None, returncode=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.py").write_bytes(payload)
            index = {"head": "old", "files": [{"path": "source.py", "sha256": old_sha or digest(payload),
                "archive_matches": [{"commit": "a" * 40, "path": "source.py"}]}]}
            with patch("maintenance.storage.check_visible_source_recovery.subprocess.check_output",
                       side_effect=[b"source.py\0new.json\0", b"head\n"]), patch(
                       "maintenance.storage.check_visible_source_recovery.subprocess.run",
                       return_value=subprocess.CompletedProcess([], returncode, blob, b"")) as run:
                result = inspect(root, index)
                self.assertEqual((root / "source.py").read_bytes(), payload)
                self.assertEqual(result["unindexed_visible_files"], 1)
                self.assertEqual(result["mutations"], [])
                return result["files"][0], run.call_count

    def test_exact_blob(self):
        row, calls = self.audit(b"x=1\n", b"x=1\n")
        self.assertEqual(row["status"], "exact_blob")
        self.assertEqual(calls, 1)

    def test_eol_only_not_exact(self):
        row, _ = self.audit(b"x=1\r\n", b"x=1\n")
        self.assertEqual(row["status"], "eol_only_blob")

    def test_changed_index_never_accepted(self):
        row, calls = self.audit(b"x=2\n", b"x=2\n", digest(b"x=1\n"))
        self.assertEqual(row["status"], "changed_since_index_retained")
        self.assertEqual(calls, 0)

    def test_missing_object_retained(self):
        row, _ = self.audit(b"x=1\n", b"", returncode=128)
        self.assertEqual(row["status"], "unverified_retained")

    def test_content_difference_retained(self):
        row, _ = self.audit(b"x=1\n", b"x=2\n")
        self.assertEqual(row["status"], "unverified_retained")

    def test_reachable_fallback_reads_and_compares_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b"x=1\n"
            (root / "source.py").write_bytes(payload)
            oid = "b" * 40
            index = {"files": [{"path": "source.py", "sha256": digest(payload)}]}
            with patch("maintenance.storage.check_visible_source_recovery.subprocess.check_output",
                       side_effect=[b"source.py\0", (oid + " source.py\n").encode(),
                                    (oid + "\n").encode(), payload, b"head\n"]):
                result = inspect(root, index, all_reachable=True)
            self.assertEqual(result["files"][0]["status"], "reachable_exact_blob")
            self.assertEqual(result["files"][0]["recovery_blob"], oid)
            self.assertEqual(result["mutations"], [])


if __name__ == "__main__":
    unittest.main()
