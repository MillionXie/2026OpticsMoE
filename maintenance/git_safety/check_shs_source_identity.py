"""Verify published SHS identities; optionally audit private recovery objects.

The default check works in a normal clone of main. Recovery refs are deliberately
not required: they are preservation records, not public runtime dependencies.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


MANIFEST = 'maintenance/storage/SHS_SHARED_CONTROLLER_SOURCE_20261004.json'


def check(root, commit, *, audit_archive=False):
    def read(ref, path):
        return subprocess.check_output(['git', '-C', str(root), 'show', f'{ref}:{path}'])
    manifest = json.loads(read(commit, MANIFEST))
    assert len(manifest['paths']) == 4
    for row in manifest['paths']:
        adopted = read(commit, row['published_path']).replace(b'\r\n', b'\n')
        assert hashlib.sha256(adopted).hexdigest() == row['published_sha256']
        ast.parse(adopted)
        if audit_archive:
            original = read(row['archive_commit'], row['archive_path'])
            assert hashlib.sha256(original).hexdigest() == row['source_sha256']
            if row['published_path'].endswith(('/sdk.py', '/phase_hdmi.py')):
                assert original.replace(b'\r\n', b'\n') == adopted
    driver = read(commit, 'experiments/hardware_sdk/devices.py').replace(b'\r\n', b'\n')
    assert hashlib.sha256(driver).hexdigest() == 'ba858591d2d71ad5e6fdea34427a503093ae60d742b54951503671f648bb3df7'
    assert not manifest['original_windows_runtime_changed']
    assert not manifest['task_acquisition_runner_migration_complete']
    return {'commit': commit, 'published_controller_sources_checked': 4,
            'private_recovery_archive_checked': audit_archive,
            'actual_controller_sources_checked': 4 if audit_archive else 0,
            'shared_driver_sha_matches_actual_windows_source': True,
            'device_or_vendor_binary_opened': False,
            'full_task_hardware_migration_claimed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', default='main')
    parser.add_argument('--audit-archive', action='store_true',
                        help='Also require and verify private recovery Git objects')
    args = parser.parse_args()
    print(json.dumps(check(Path(__file__).resolve().parents[2], args.commit,
                           audit_archive=args.audit_archive), indent=2))
