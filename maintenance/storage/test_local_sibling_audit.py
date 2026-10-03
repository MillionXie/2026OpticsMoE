"""Bounded tests for content verification; no real research directories touched."""
import tempfile
from pathlib import Path

import pytest

from maintenance.storage.audit_local_sibling_projects import inventory, verify


def test_inventory_distinguishes_equal_different_and_missing():
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) / "export"
        repo = Path(scratch) / "main"
        root.mkdir()
        repo.mkdir()
        for name, old, new in (("equal.txt", "a", "a"), ("different.txt", "a", "b")):
            (root / name).write_text(old)
            (repo / name).write_text(new)
        (root / "unique.txt").write_text("unique")
        result = inventory(root, repo)
        assert result["files"] == 3
        assert result["same_as_main_working"] == 1
        assert result["different_from_main_working"] == 1
        assert result["absent_from_main_working"] == 1
        verify(root, result["entries"])


def test_verify_rejects_changed_content_of_equal_size():
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        item = root / "timing.csv"
        item.write_text("123")
        rows = inventory(root, root)["entries"]
        item.write_text("456")
        with pytest.raises(RuntimeError, match="Content changed"):
            verify(root, rows)


def test_verify_rejects_added_file():
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        rows = inventory(root, root)["entries"]
        (root / "new.txt").write_text("new")
        with pytest.raises(RuntimeError, match="File set changed"):
            verify(root, rows)


def test_verify_rejects_missing_file():
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        item = root / "old.txt"
        item.write_text("old")
        rows = inventory(root, root)["entries"]
        item.unlink()
        with pytest.raises(RuntimeError, match="File set changed"):
            verify(root, rows)
