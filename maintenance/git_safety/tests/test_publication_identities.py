import hashlib
import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_publication_tree as module


@pytest.fixture(autouse=True)
def no_external_git_for_synthetic_tests(monkeypatch):
    # These synthetic tests have no repository; evolution-list lookups are empty.
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: b"")


def test_publication_mismatch_is_detected(monkeypatch):
    manifest = {"paths": [{"path": "model.py", "source_sha256": hashlib.sha256(b"original").hexdigest()}]}
    monkeypatch.setattr(module, "read", lambda root, ref, path: json.dumps(manifest).encode() if path == "manifest" else b"changed")
    result = module.verify_reviewed_publications(None, "main", ["manifest"])
    assert result["errors"] == ["reviewed publication mismatch: model.py"]


def test_both_reviewed_manifest_schemas(monkeypatch):
    digest = hashlib.sha256(b"original").hexdigest()
    manifest = {"paths": [{"path": "a.py", "sha256": digest}, {"path": "b.py", "source_sha256": digest}]}
    monkeypatch.setattr(module, "read", lambda root, ref, path: json.dumps(manifest).encode() if path == "manifest" else b"original")
    result = module.verify_reviewed_publications(None, "main", ["manifest"])
    assert result == {"reviewed_publication_hashes_checked": 2, "errors": []}


def test_source_files_schema(monkeypatch):
    digest = hashlib.sha256(b"original").hexdigest()
    manifest = {"source_files": [{"path": "model.py", "sha256": digest}]}
    monkeypatch.setattr(module, "read", lambda root, ref, path: json.dumps(manifest).encode() if path == "manifest" else b"original")
    result = module.verify_reviewed_publications(None, "main", ["manifest"])
    assert result == {"reviewed_publication_hashes_checked": 1, "errors": []}


def test_empty_schema_rejected(monkeypatch):
    import pytest
    monkeypatch.setattr(module, "read", lambda *args: b'{"source_files": []}')
    with pytest.raises(ValueError, match="Missing reviewed source rows"):
        module.verify_reviewed_publications(None, "main", ["manifest"])
