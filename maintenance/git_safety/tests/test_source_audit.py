import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "audit_task_sources.py"
spec = importlib.util.spec_from_file_location("audit_task_sources", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_byte_identity_endings_missing_and_local_changes(tmp_path, monkeypatch):
    prefix = "LightGenV2/tasks/example"
    blobs = {"same.py": b"same\n", "eol.py": b"eol\n", "changed.py": b"old\n", "missing.py": b"absent\n"}
    target = tmp_path / prefix
    target.mkdir(parents=True)
    (target / "same.py").write_bytes(blobs["same.py"])
    (target / "eol.py").write_bytes(b"eol\r\n")
    (target / "changed.py").write_bytes(b"new\n")

    def fake_git(root, *args, stdin=None):
        if args[0] == "rev-parse":
            return b"pinned\n"
        if args[0] == "ls-tree":
            return b"".join(b"100644 blob oid\t" + (prefix + "/" + name).encode() + b"\0" for name in blobs)
        if args[0] == "cat-file":
            return b"".join(b"oid blob " + str(len(data)).encode() + b"\n" + data + b"\n" for data in blobs.values())
        raise AssertionError(args)

    monkeypatch.setattr(module, "git", fake_git)
    report = module.audit(tmp_path, "ref", prefix)
    assert report["counts"] == {"byte_identical": 1, "line_endings_only": 1, "missing": 1,
                               "different_preserve_local": 1, "unsafe_path": 0}
    assert (target / "changed.py").read_bytes() == b"new\n"
    assert not (target / "missing.py").exists()


def test_audit_rejects_other_directories(tmp_path):
    with pytest.raises(ValueError):
        module.audit(tmp_path, "ref", "../outside")


@pytest.mark.parametrize("prefix", ["LightGenV2/tasks/example/../../common",
                                    "LightGenV2/tasks", "LightGenV2/tasks/../../../tmp"])
def test_audit_rejects_task_boundary_traversal(tmp_path, prefix):
    with pytest.raises(ValueError):
        module.audit(tmp_path, "ref", prefix)
