import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_t06_runtime_closure as module
from test_main_preparation import git, synthetic_repo


def fixture(tmp_path, monkeypatch):
    synthetic_repo(tmp_path)
    path = module.BACKEND + "modeling.py"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"value = 1\n")
    git(tmp_path, "add", path)
    git(tmp_path, "commit", "-m", "old backend")
    main = git(tmp_path, "rev-parse", "HEAD").decode().strip()
    git(tmp_path, "update-ref", "refs/heads/main", main)
    target.write_bytes(b"value = 2\n")
    git(tmp_path, "add", path)
    git(tmp_path, "commit", "-m", "reviewed source")
    source = git(tmp_path, "rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(module, "SOURCE", source)
    return {"main_commit": main, "source_commit": source, "profile_source_contracts": [],
            "files": [{"path": path, "bytes": 10,
                       "source_sha256": hashlib.sha256(b"value = 2\n").hexdigest(),
                       "main_sha256": hashlib.sha256(b"value = 1\n").hexdigest(),
                       "main_relation": "different_requires_review"}]}, target


def test_preserves_user_source_index_and_refs(tmp_path, monkeypatch):
    audit, target = fixture(tmp_path, monkeypatch)
    target.write_bytes(b"user = 'keep'\n")
    (tmp_path / "user.py").write_bytes(b"user = 1\n")
    git(tmp_path, "add", "user.py")
    before = git(tmp_path, "status", "--porcelain")
    result = module.prepare(tmp_path, audit, "candidate")
    assert git(tmp_path, "status", "--porcelain") == before
    assert target.read_bytes() == b"user = 'keep'\n"
    assert git(tmp_path, "rev-parse", "main").decode().strip() == audit["main_commit"]
    assert git(tmp_path, "show", result["candidate_commit"] + ":" + audit["files"][0]["path"]) == b"value = 2\n"


@pytest.mark.parametrize("field", ["source_sha256", "main_sha256"])
def test_sha_gate(tmp_path, monkeypatch, field):
    audit, _ = fixture(tmp_path, monkeypatch)
    audit["files"][0][field] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        module.prepare(tmp_path, audit, "rejected")


def test_rejects_other_task(tmp_path, monkeypatch):
    audit, _ = fixture(tmp_path, monkeypatch)
    audit["files"][0]["path"] = "LightGenV2/tasks/t07_abo_image_retrieval/model.py"
    with pytest.raises(RuntimeError, match="Unaudited"):
        module.prepare(tmp_path, audit, "rejected")


def test_rejects_mismatched_profile(tmp_path, monkeypatch):
    audit, _ = fixture(tmp_path, monkeypatch)
    audit["profile_source_contracts"] = [{"matches": False}]
    with pytest.raises(RuntimeError, match="profile/backend"):
        module.prepare(tmp_path, audit, "rejected")


def test_rejects_stale_main(tmp_path, monkeypatch):
    audit, _ = fixture(tmp_path, monkeypatch)
    audit["main_commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="Stale main"):
        module.prepare(tmp_path, audit, "rejected")
