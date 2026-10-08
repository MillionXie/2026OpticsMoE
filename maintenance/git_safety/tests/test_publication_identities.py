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


@pytest.mark.parametrize("path,scope,accepted", [
    ("LightGenV2/scripts/profile_narrow_optical_electronics_a100.py", "historical_timing_contract", True),
    ("LightGenV2/scripts/unrelated.py", "historical_timing_contract", False),
    ("LightGenV2/reports/20260915_baseline_methods/CODE_HANDOFF.md", "historical_baseline_navigation", True),
    ("LightGenV2/reports/unrelated.md", "historical_baseline_navigation", False),
    ("LightGenV2/reports/20260915_baseline_methods/CODE_HANDOFF.md", "historical_timing_contract", False),
])
def test_only_named_reviewed_non_task_evolutions_are_permitted(monkeypatch, path, scope, accepted):
    old = hashlib.sha256(b"old").hexdigest()
    new = hashlib.sha256(b"new").hexdigest()
    row = {"path": path, "scope": scope,
           "historical_source_commit": "old", "historical_sha256": old,
           "source_commit": "new", "sha256": new, "review_reason": "audited contract"}
    def read(root, ref, name):
        if name.endswith("TASK_SOURCE_EVOLUTION_20261005.json"):
            return json.dumps({"paths": [row]}).encode()
        if name == "manifest":
            return json.dumps({"paths": [{"path": path, "sha256": old}]}).encode()
        return b"old" if ref == "old" else b"new"
    monkeypatch.setattr(module, "read", read)
    monkeypatch.setattr(module.subprocess, "check_output", lambda args: b"present" if args[-1].endswith("TASK_SOURCE_EVOLUTION_20261005.json") else b"")
    if accepted:
        assert module.verify_reviewed_publications(None, "main", ["manifest"])["errors"] == []
    else:
        with pytest.raises(ValueError, match="Unsafe/unreviewed task evolution"):
            module.verify_reviewed_publications(None, "main", ["manifest"])


@pytest.mark.parametrize("corrupt", ["history", "new", "reason"])
def test_task_evolution_rejects_unverified_history(monkeypatch, corrupt):
    path = "LightGenV2/tasks/t04_openmoji_robust_ablation/lab_shs_capture.py"
    old, new = hashlib.sha256(b"old").hexdigest(), hashlib.sha256(b"new").hexdigest()
    row = {"path": path, "historical_source_commit": "old", "historical_sha256": old,
           "source_commit": "new", "sha256": new, "review_reason": "audited"}
    if corrupt == "reason":
        row["review_reason"] = ""
    def read(root, ref, name):
        if name.endswith("TASK_SOURCE_EVOLUTION_20261005.json"):
            return json.dumps({"paths": [row]}).encode()
        return b"wrong" if ref == {"history": "old", "new": "new"}.get(corrupt) else ref.encode()
    monkeypatch.setattr(module, "read", read)
    monkeypatch.setattr(module.subprocess, "check_output", lambda args: b"present" if args[-1].endswith("TASK_SOURCE_EVOLUTION_20261005.json") else b"")
    with pytest.raises(ValueError):
        module.verify_reviewed_publications(None, "main", [])
