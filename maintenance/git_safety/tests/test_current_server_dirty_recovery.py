"""Verify recorded runtime bytes remain retrievable without altering checkouts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def test_all_current_tracked_differences_have_exact_recovery_bytes():
    data = json.loads((ROOT / 'maintenance/storage/CURRENT_SERVER_DIRTY_SOURCE_RECOVERY_20261007.json').read_text(encoding='utf8'))
    files = [f for row in data['rows'] for f in row['changed_files']]
    assert len(data['rows']) == 9
    assert len(files) == data['changed_file_count'] == 35
    assert sum(f['raw_equals_main'] for f in files) == 13
    for f in files:
        raw = subprocess.check_output(['git', 'show', f['recovery_commit'] + ':' + f['path']], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == f['sha256']
    assert not data['runtime_checkout_changed']
    assert not data['directory_removal_authorized_by_this_check']
