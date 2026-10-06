"""CLI rejection and static mutation boundary, without SSH/credentials/devices."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).with_name("readonly_server_policy.py")


@pytest.mark.parametrize("phase", ["sync", "publish-bundle"])
def test_old_write_phases_reject_before_transport(phase):
    result = subprocess.run([sys.executable, str(SOURCE), "--phase", phase], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Legacy write phase retired" in result.stderr
    assert "SSH password" not in result.stdout and "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize("option,value", [("--repo", "relative"), ("--repo", "/data/../repo"),
                                          ("--checkout", "relative"), ("--checkout", "/data/../repo")])
def test_invalid_paths_rejected_before_transport(option, value):
    result = subprocess.run([sys.executable, str(SOURCE), "--phase", "inspect", option, value], capture_output=True, text=True)
    assert result.returncode == 2 and "parent traversal" in result.stderr


@pytest.mark.parametrize("pin", [None, "main", "f" * 39, "G" * 40])
def test_test_requires_exact_pin_before_transport(pin):
    command = [sys.executable, str(SOURCE), "--phase", "test"]
    if pin is not None:
        command += ["--commit", pin]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2 and "exact 40-character" in result.stderr


def test_no_legacy_git_mutation_implementation_remains():
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not any(token in value for value in strings for token in (
        "worktree add", "checkout --detach", "push origin", "mkdir -p", "update-ref"))
    assert "open_sftp" not in text and "sftp.put" not in text
    assert "paramiko" not in text and "getpass" not in text


def test_default_checkout_not_old_worktree():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"]
    call = next(n for n in calls if n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--checkout")
    assert not call.keywords
