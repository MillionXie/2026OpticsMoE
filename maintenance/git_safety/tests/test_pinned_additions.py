import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_pinned_additions import prepare
from test_main_preparation import git, synthetic_repo


def source(tmp_path, path="model.py", data=b"model = 1\n"):
    main = synthetic_repo(tmp_path)
    (tmp_path / path).write_bytes(data)
    git(tmp_path, "add", path)
    git(tmp_path, "commit", "-m", "source")
    return main, {"source_commit": git(tmp_path, "rev-parse", "HEAD").decode().strip(),
                  "paths": [{"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}


def test_missing_only_preserves_dirty_checkout_and_staged_user_file(tmp_path):
    main, manifest = source(tmp_path)
    (tmp_path / "model.py").write_text("user model = modified\n")
    (tmp_path / "user.py").write_text("user = True\n")
    git(tmp_path, "add", "user.py")
    before = git(tmp_path, "status", "--porcelain")
    result = prepare(tmp_path, manifest, "candidate")
    assert result["added_paths"] == ["model.py"]
    assert git(tmp_path, "status", "--porcelain") == before
    assert git(tmp_path, "rev-parse", "main").decode().strip() == main
    assert git(tmp_path, "show", result["candidate_commit"] + ":model.py") == b"model = 1\n"
    assert not git(tmp_path, "ls-tree", result["candidate_commit"], "--", "user.py")


def test_refuses_existing_different_target(tmp_path):
    _, manifest = source(tmp_path, "README.md", b"changed\n")
    with pytest.raises(RuntimeError, match="never overwrite"):
        prepare(tmp_path, manifest, "refused")


def test_refuses_unreviewed_sha(tmp_path):
    _, manifest = source(tmp_path)
    manifest["paths"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="mismatch"):
        prepare(tmp_path, manifest, "refused")


@pytest.mark.parametrize("path", ["weights.pt", "model.py"])
def test_refuses_artifacts_or_binary(tmp_path, path):
    _, manifest = source(tmp_path, path, b"synthetic\0payload")
    with pytest.raises(RuntimeError, match="Unsafe source"):
        prepare(tmp_path, manifest, "refused")


def test_refuses_connection_helper(tmp_path):
    _, manifest = source(tmp_path, "server_sync.py")
    with pytest.raises(RuntimeError, match="Unsafe/duplicate path"):
        prepare(tmp_path, manifest, "refused")


def test_refuses_main_checkout(tmp_path):
    _, manifest = source(tmp_path)
    git(tmp_path, "checkout", "main")
    with pytest.raises(RuntimeError, match="main is checked out"):
        prepare(tmp_path, manifest, "refused")


def test_base_candidate_can_compose_without_updating_main(tmp_path):
    main, manifest = source(tmp_path)
    tree = git(tmp_path, "rev-parse", main + "^{tree}").decode().strip()
    parent = git(tmp_path, "commit-tree", tree, "-p", main, "-m", "metadata candidate").decode().strip()
    result = prepare(tmp_path, manifest, "composed", parent)
    assert result["candidate_parent"] == parent
    assert git(tmp_path, "rev-parse", result["candidate_commit"] + "^").decode().strip() == parent
    assert git(tmp_path, "rev-parse", "main").decode().strip() == main
