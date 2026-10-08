"""Use disposable miniature repositories, never the user's real branch refs."""
from pathlib import Path

import pytest

from maintenance.git_safety.archive_inactive_local_branches import apply, digest, git, metadata_path, plan, state


def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Archive Test")
    git(root, "config", "user.email", "archive-test@example.invalid")
    (root / "source.py").write_text("baseline\n")
    git(root, "add", "source.py")
    git(root, "commit", "-m", "baseline")
    git(root, "checkout", "-b", "old-trial")
    (root / "source.py").write_text("unique trial source\n")
    git(root, "commit", "-am", "unique trial")
    git(root, "checkout", "main")
    return root


def test_plan_is_read_only_and_protects_main(tmp_path):
    root = repository(tmp_path)
    before = state(root)
    receipt = plan(root)
    assert [row["branch"] for row in receipt["branches"]] == ["old-trial"]
    assert not receipt["branches"][0]["merged_into_main"]
    assert state(root) == before


def test_metadata_path_resolves_legacy_relative_index(tmp_path):
    root = repository(tmp_path)
    index = metadata_path(root, 'index')
    assert index == root/'.git/index'
    assert state(root)['index_sha256'] == digest(index)


def test_archive_preserves_unique_commit_bundle_and_user_edit(tmp_path):
    root = repository(tmp_path)
    (root / "source.py").write_text("user's unsaved edit\n")
    receipt = plan(root)
    old = receipt["branches"][0]
    result = apply(root, receipt, tmp_path / "private_backup")
    assert result["branch_count_before"] == 2
    assert result["branch_count_after"] == 1
    assert state(root) == receipt["before"]
    assert git(root, "rev-parse", old["archive_ref"]).decode().strip() == old["head"]
    assert git(root, "show", old["archive_ref"] + ":source.py") == b"unique trial source\n"
    assert (root / "source.py").read_text() == "user's unsaved edit\n"
    git(root, "bundle", "verify", result["bundle"])


def test_concurrent_working_edit_aborts_before_archival(tmp_path):
    root = repository(tmp_path)
    receipt = plan(root)
    (root / "source.py").write_text("concurrent edit\n")
    with pytest.raises(RuntimeError, match="changed; aborting"):
        apply(root, receipt, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()
    assert git(root, "rev-parse", "refs/heads/old-trial")


def test_plain_stdin_batch_does_not_partially_create_on_conflict(tmp_path):
    root = repository(tmp_path)
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    git(root, 'update-ref', 'refs/archive/already-exists', head)
    batch = (f'create refs/archive/new-one {head}\n'
             f'create refs/archive/already-exists {head}\n').encode()
    with pytest.raises(RuntimeError):
        git(root, 'update-ref', '--stdin', data=batch)
    refs = git(root, 'for-each-ref', '--format=%(refname)', 'refs/archive').decode().splitlines()
    assert refs == ['refs/archive/already-exists']


def test_plain_stdin_conditional_delete_is_atomic_on_changed_identity(tmp_path):
    root = repository(tmp_path)
    main = git(root, 'rev-parse', 'main').decode().strip()
    trial = git(root, 'rev-parse', 'old-trial').decode().strip()
    git(root, 'update-ref', 'refs/archive/first', main)
    git(root, 'update-ref', 'refs/archive/second', trial)
    batch = (f'delete refs/archive/first {main}\n'
             f'delete refs/archive/second {main}\n').encode()
    with pytest.raises(RuntimeError):
        git(root, 'update-ref', '--stdin', data=batch)
    assert git(root, 'rev-parse', 'refs/archive/first').decode().strip() == main
    assert git(root, 'rev-parse', 'refs/archive/second').decode().strip() == trial


def test_ignore_rules_hide_payload_not_source_or_tracked_evidence(tmp_path):
    root = repository(tmp_path)
    report = root / "handoffs" / "test"
    report.mkdir(parents=True)
    tracked = report / "curated.png"
    tracked.write_bytes(b"evidence")
    git(root, "add", "handoffs/test/curated.png")
    git(root, "commit", "-m", "retain curated evidence")
    source_rules = Path(__file__).resolve().parents[2] / ".gitignore"
    (root / ".gitignore").write_text(source_rules.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("run.py", "config.yaml", "README.md", "manifest.json", "metrics.csv", "raw.png"):
        (report / name).write_text("local file")
    untracked = git(root, "ls-files", "--others", "--exclude-standard").decode().splitlines()
    for name in ("run.py", "config.yaml", "README.md", "manifest.json", "metrics.csv"):
        assert "handoffs/test/" + name in untracked
    assert "handoffs/test/raw.png" not in untracked
    assert git(root, "ls-files", "handoffs/test/curated.png").strip()
    assert tracked.read_bytes() == b"evidence"
