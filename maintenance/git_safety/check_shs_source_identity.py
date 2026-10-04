"""Verify the adopted SHS helpers against reviewed source identities in Git."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


MANIFEST = 'maintenance/storage/SHS_SHARED_CONTROLLER_SOURCE_20261004.json'


def check(root, commit):
    def read(ref, path):
        return subprocess.check_output(['git', '-C', str(root), 'show', f'{ref}:{path}'])
    manifest = json.loads(read(commit, MANIFEST))
    assert len(manifest['paths']) == 4
    for row in manifest['paths']:
        original = read(row['archive_commit'], row['archive_path'])
        adopted = read(commit, row['published_path']).replace(b'\r\n', b'\n')
        assert hashlib.sha256(original).hexdigest() == row['source_sha256']
        assert hashlib.sha256(adopted).hexdigest() == row['published_sha256']
        ast.parse(adopted)
        if row['published_path'].endswith(('/sdk.py', '/phase_hdmi.py')):
            assert original.replace(b'\r\n', b'\n') == adopted
    driver = read(commit, 'experiments/hardware_sdk/devices.py').replace(b'\r\n', b'\n')
    assert hashlib.sha256(driver).hexdigest() == 'ba858591d2d71ad5e6fdea34427a503093ae60d742b54951503671f648bb3df7'
    assert not manifest['original_windows_runtime_changed']
    assert not manifest['task_acquisition_runner_migration_complete']
    return {'commit': commit, 'actual_controller_sources_checked': 4,
            'shared_driver_sha_matches_actual_windows_source': True,
            'device_or_vendor_binary_opened': False,
            'full_task_hardware_migration_claimed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', default='main')
    args = parser.parse_args()
    print(json.dumps(check(Path(__file__).resolve().parents[2], args.commit), indent=2))
