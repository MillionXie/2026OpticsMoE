import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base))
spec = importlib.util.spec_from_file_location("prepare_scoped_main", base / "prepare_scoped_main.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def synthetic_repo(tmp_path):
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "user.email", "synthetic@example.invalid")
    git(tmp_path, "config", "user.name", "Synthetic Test")
    git(tmp_path, "config", "core.autocrlf", "false")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    git(tmp_path, "add", "README.md")
    git(tmp_path, "commit", "-m", "base")
    main = git(tmp_path, "rev-parse", "main").decode().strip()
    git(tmp_path, "checkout", "--detach")
    return main


def test_prepare_preserves_head_index_and_main_and_excludes_other_changes(tmp_path):
    main = synthetic_repo(tmp_path)
    (tmp_path / "README.md").write_text("base\naudited\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("unrelated = True\n", encoding="utf-8")
    git(tmp_path, "add", "README.md", "unrelated.py")
    git(tmp_path, "commit", "-m", "source")
    source = git(tmp_path, "rev-parse", "HEAD").decode().strip()
    (tmp_path / "user.py").write_text("user = True\n", encoding="utf-8")
    git(tmp_path, "add", "user.py")
    before = git(tmp_path, "status", "--porcelain")
    report = module.prepare(tmp_path, [source], "candidate", ["README.md"])
    assert report["changed_paths"] == ["README.md"]
    assert git(tmp_path, "status", "--porcelain") == before
    assert git(tmp_path, "rev-parse", "main").decode().strip() == main
    assert git(tmp_path, "rev-parse", "HEAD").decode().strip() == source
    assert git(tmp_path, "show", report["candidate_commit"] + ":README.md") == git(tmp_path, "show", source + ":README.md")
    assert not git(tmp_path, "ls-tree", report["candidate_commit"], "--", "unrelated.py")


def test_prepare_refuses_checked_out_main(tmp_path):
    main = synthetic_repo(tmp_path)
    git(tmp_path, "checkout", "main")
    with pytest.raises(RuntimeError, match="main is checked out"):
        module.prepare(tmp_path, [main], "not permitted")


def test_prepare_rejects_weights(tmp_path):
    synthetic_repo(tmp_path)
    (tmp_path / "weights.pt").write_bytes(b"synthetic")
    git(tmp_path, "add", "weights.pt")
    git(tmp_path, "commit", "-m", "unsafe source")
    with pytest.raises(RuntimeError, match="Unsafe publication paths"):
        module.prepare(tmp_path, ["HEAD"], "not permitted")
