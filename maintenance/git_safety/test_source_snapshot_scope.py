import importlib.util
from pathlib import Path
import subprocess
import hashlib
import os
import pytest

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


def test_deep_source_path_is_readable(tmp_path):
    relative = '/'.join(['long_package_component'] * 14) + '/source.py'
    path = scope.source_path(tmp_path.resolve(), relative)
    path.parent.mkdir(parents=True)
    path.write_bytes(b'original source\n')
    assert path.is_file()
    assert path.read_bytes() == b'original source\n'
    if os.name == 'nt':
        assert str(path).startswith('\\\\?\\')


def test_source_path_does_not_change_relative_identity(tmp_path):
    path = scope.source_path(tmp_path.resolve(), 'package/source.py')
    assert path.name == 'source.py'
    assert path.suffix == '.py'


def history_fixture(tmp_path, monkeypatch, payload=b"old implementation\n"):
    (tmp_path / 'old.py').write_bytes(payload)
    current = {'head': 'abc', 'reference_commit': 'abc',
        'scope': 'selected_source_extensions_up_to_2_mib',
        'source_file_count': 1, 'skipped_counts': {},
        'counts': {'no_byte_identity_in_reference': 1},
        'files': [{'path': 'old.py', 'identity': 'no_byte_identity_in_reference',
                   'sha256': hashlib.sha256(payload).hexdigest()}]}
    monkeypatch.setattr(scope, 'audit', lambda *args: current)
    monkeypatch.setattr(scope, 'history_refs', lambda root: {'refs/archive/old': 'def'})
    def fake_git(root, *args):
        if args[0] == 'ls-tree':
            return b'100644 blob ' + scope.blob_id(b'old implementation\n').encode() + b'\tcanonical/old.py\0'
        assert args == ('rev-parse', 'HEAD')
        return b'abc\n'
    monkeypatch.setattr(scope, 'git', fake_git)
    return current


def test_existing_history_identity_is_not_main_adoption(tmp_path, monkeypatch):
    history_fixture(tmp_path, monkeypatch)
    result = scope.audit_history(tmp_path)
    assert result['history_counts'] == {'exact_git_blob': 1}
    assert result['files'][0]['archive_matches'][0]['commit'] == 'def'
    assert result['mutations'] == []


def test_history_line_ending_identity_is_explicit(tmp_path, monkeypatch):
    history_fixture(tmp_path, monkeypatch, b'old implementation\r\n')
    assert scope.audit_history(tmp_path)['history_counts'] == {'crlf_normalized_git_blob': 1}


def test_history_source_change_rejected(tmp_path, monkeypatch):
    history_fixture(tmp_path, monkeypatch)
    (tmp_path / 'old.py').write_bytes(b'new user edit')
    with pytest.raises(RuntimeError, match='Source changed'):
        scope.audit_history(tmp_path)


def test_history_ref_change_rejected(tmp_path, monkeypatch):
    history_fixture(tmp_path, monkeypatch)
    refs = iter([{'refs/archive/old': 'def'}, {'refs/archive/old': 'ghi'}])
    monkeypatch.setattr(scope, 'history_refs', lambda root: next(refs))
    with pytest.raises(RuntimeError, match='Git identity changed'):
        scope.audit_history(tmp_path)
