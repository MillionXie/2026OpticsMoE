import importlib.util
import hashlib
import json
import subprocess
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load("check_task_registry")
planner = load("plan_source_import")


def fixture_registry(tmp_path, extra=None):
    base = tmp_path / "LightGenV2"
    base.mkdir()
    (base / "README.md").write_text("# Task\n", encoding="utf-8")
    task = {"id": "example", "entry": "README.md", "source_status": "integration_pending"}
    task.update(extra or {})
    registry = {"migration_complete": False, "tasks": [task]}
    (base / "TASK_REGISTRY.json").write_text(json.dumps(registry), encoding="utf-8")
    return base, registry


def test_real_registry_entries_navigation_weights_and_imported_source():
    root = Path(__file__).resolve().parents[3]
    assert checker.inspect(root, True)["errors"] == []


def test_duplicate_ids_and_invalid_sha_are_not_silent(tmp_path):
    base, registry = fixture_registry(tmp_path, {"weights": [{"sha256": "wrong"}]})
    registry["tasks"].append(dict(registry["tasks"][0]))
    (base / "TASK_REGISTRY.json").write_text(json.dumps(registry), encoding="utf-8")
    errors = checker.inspect(tmp_path)["errors"]
    assert any("duplicate task id" in item for item in errors)
    assert any("invalid SHA256" in item for item in errors)


def test_published_tree_check_does_not_use_dirty_working_entry(tmp_path):
    base, registry = fixture_registry(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "--",
                    "LightGenV2/README.md", "LightGenV2/TASK_REGISTRY.json"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    (base / "README.md").unlink()
    assert checker.inspect_tree(tmp_path, "HEAD")["errors"] == []
    assert checker.inspect(tmp_path)["errors"]


def test_published_tree_detects_missing_evidence(tmp_path):
    fixture_registry(tmp_path, {"evidence": ["absent.md"]})
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "--",
                    "LightGenV2/README.md", "LightGenV2/TASK_REGISTRY.json"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    assert checker.inspect_tree(tmp_path, "HEAD")["errors"] == [
        "missing entry/evidence: LightGenV2/absent.md"]


def test_missing_entry_and_navigation_are_detected(tmp_path):
    base, registry = fixture_registry(tmp_path, {"entry": "absent.md"})
    registry["navigation_documents"] = ["README.md"]
    (base / "README.md").write_text("[missing](absent.md) [web](https://example.org)\n")
    (base / "TASK_REGISTRY.json").write_text(json.dumps(registry), encoding="utf-8")
    errors = checker.inspect(tmp_path)["errors"]
    assert any("missing entry" in item for item in errors)
    assert any("broken navigation" in item for item in errors)


def test_entry_cannot_escape_repository(tmp_path):
    fixture_registry(tmp_path, {"entry": "../../outside.md"})
    assert any("escapes repository" in item for item in checker.inspect(tmp_path)["errors"])


def test_private_evidence_is_not_required_in_a_source_only_clone(tmp_path):
    fixture_registry(tmp_path, {"evidence": ["../handoffs/example/result.json"]})
    report = checker.inspect(tmp_path)
    assert report["errors"] == []
    assert report["private_artifact_validation_required"]
    assert any("missing entry/evidence" in item for item in checker.inspect(tmp_path, verify_private=True)["errors"])


def test_dynamic_dependency_hash_and_escape_are_checked(tmp_path):
    base, registry = fixture_registry(tmp_path)
    registry["source_import_manifests"] = ["source.json"]
    manifest = {"files": [], "additional_dependency_hashes": [
        {"path": "LightGenV2/README.md", "source_blob_sha256": "0" * 64},
        {"path": "../outside.py", "source_blob_sha256": "0" * 64}],
        "additional_existing_dependencies": ["../outside.py"]}
    (base / "source.json").write_text(json.dumps(manifest), encoding="utf-8")
    (base / "TASK_REGISTRY.json").write_text(json.dumps(registry), encoding="utf-8")
    errors = checker.inspect(tmp_path)["errors"]
    assert any("SHA mismatch" in item for item in errors)
    assert any("source escapes" in item for item in errors)
    assert any("dependency escapes" in item for item in errors)


def test_source_normalization_is_explicit_and_only_line_endings(tmp_path):
    base, registry = fixture_registry(tmp_path)
    registry["source_import_manifests"] = ["source.json"]
    expected = hashlib.sha256(b"# Task\n").hexdigest()
    manifest = {"files": [{"path": "LightGenV2/README.md",
                           "source_blob_sha256": expected,
                           "hash_normalization": "crlf_to_lf"}]}
    (base / "README.md").write_bytes(b"# Task\r\n")
    (base / "source.json").write_text(json.dumps(manifest), encoding="utf-8")
    (base / "TASK_REGISTRY.json").write_text(json.dumps(registry), encoding="utf-8")
    assert checker.inspect(tmp_path)["errors"] == []
    (base / "README.md").write_bytes(b"# Different\r\n")
    assert any("SHA mismatch" in error for error in checker.inspect(tmp_path)["errors"])
    manifest["files"][0]["hash_normalization"] = "strip_everything"
    (base / "source.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("unknown source hash" in error for error in checker.inspect(tmp_path)["errors"])


def test_source_planner_follows_relative_imports_and_refuses_overwrite(tmp_path, monkeypatch):
    blobs = {"pkg/__init__.py": b"", "pkg/main.py": b"from .helper import value\n",
             "pkg/helper.py": b"value = 1\n"}

    def fake_git(root, *args):
        if args[0] == "rev-parse":
            return b"pinned\n"
        if args[0] == "ls-tree":
            return "\n".join(blobs).encode()
        if args[0] == "show":
            return blobs[args[1].split(":", 1)[1]]
        raise AssertionError(args)

    monkeypatch.setattr(planner, "git", fake_git)
    report = planner.plan(tmp_path, "ref", ["pkg/main.py"])
    assert {row["path"] for row in report["new_files"]} == set(blobs)
    assert not (tmp_path / "pkg").exists()  # Planner is read-only.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/helper.py").write_text("value = 2\n")
    report = planner.plan(tmp_path, "ref", ["pkg/main.py"])
    assert report["existing_conflicts"] == [{"path": "pkg/helper.py", "reason": "different existing local file; never overwrite"}]
    assert (tmp_path / "pkg/helper.py").read_text() == "value = 2\n"
