import importlib.util
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location("scope", Path(__file__).with_name("source_snapshot_scope.py"))
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)


def test_exact_identity():
    data = b"print('x')\n"
    assert scope.identity(data, {scope.blob_id(data): ["canonical.py"]}) == (
        "exact_git_blob", ["canonical.py"])


def test_line_endings_are_explicit():
    assert scope.identity(b"x\r\n", {scope.blob_id(b"x\n"): ["a.py"]})[0] == "crlf_normalized_git_blob"


def test_different_content_is_not_equal():
    assert scope.identity(b"x\n", {scope.blob_id(b"y\n"): ["a.py"]}) == (
        "no_byte_identity_in_reference", [])


def test_hash_matches_git():
    actual = subprocess.check_output(["git", "hash-object", "--stdin"], input=b"hello\n").decode().strip()
    assert scope.blob_id(b"hello\n") == actual
