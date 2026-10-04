"""No hardware, repository mutation, or private archive needed for these tests."""
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

import check_shs_source_identity as checker


def fixture_reader():
    rows = []
    blobs = {}
    for name in ('sdk.py', 'capture.py', 'phase_hdmi.py', 'slm_camera.py'):
        body = b'value = 1\n'
        path = 'LightGenV2/hardware_common/shs/' + name
        rows.append({'archive_commit': 'private', 'archive_path': name,
                     'published_path': path,
                     'source_sha256': hashlib.sha256(body).hexdigest(),
                     'published_sha256': hashlib.sha256(body).hexdigest()})
        blobs['main:' + path] = body
        blobs['private:' + name] = body
    blobs['main:' + checker.MANIFEST] = json.dumps({
        'paths': rows, 'original_windows_runtime_changed': False,
        'task_acquisition_runner_migration_complete': False}).encode()
    # Use the real published driver; do not open/import any SDK.
    root = Path(__file__).resolve().parents[2]
    blobs['main:experiments/hardware_sdk/devices.py'] = subprocess.check_output(
        ['git', '-C', str(root), 'show', 'main:experiments/hardware_sdk/devices.py'])
    return blobs


def test_normal_clone_does_not_read_private_archive():
    blobs = fixture_reader()
    def read(argv):
        assert not argv[-1].startswith('private:')
        return blobs[argv[-1]]
    with patch.object(checker.subprocess, 'check_output', side_effect=read):
        result = checker.check(Path('.'), 'main')
    assert result['published_controller_sources_checked'] == 4
    assert not result['private_recovery_archive_checked']


def test_explicit_archive_audit_checks_original_hash():
    blobs = fixture_reader()
    with patch.object(checker.subprocess, 'check_output', side_effect=lambda argv: blobs[argv[-1]]):
        assert checker.check(Path('.'), 'main', audit_archive=True)['actual_controller_sources_checked'] == 4
    blobs['private:capture.py'] = b'value = 2\n'
    with patch.object(checker.subprocess, 'check_output', side_effect=lambda argv: blobs[argv[-1]]):
        with pytest.raises(AssertionError):
            checker.check(Path('.'), 'main', audit_archive=True)


def test_published_corruption_is_rejected_without_archive():
    blobs = fixture_reader()
    blobs['main:LightGenV2/hardware_common/shs/sdk.py'] = b'value = 9\n'
    with patch.object(checker.subprocess, 'check_output', side_effect=lambda argv: blobs[argv[-1]]):
        with pytest.raises(AssertionError):
            checker.check(Path('.'), 'main')
