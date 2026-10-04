"""Save verified source-only Git recovery and CPU compatibility receipts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / '.codex_tmp/t11_source_20261004'


def result(name):
    raw = json.loads((PRIVATE/name).read_text(encoding='utf-8'))
    if raw['exit_code']:
        raise RuntimeError('Failed server receipt: '+name)
    return json.loads(raw['stdout'])


def main():
    snapshot = result('t09_overlay_snapshot.json')
    compatibility = result('t09_vision_compatibility.json')
    bundle = PRIVATE / 't09_reviewed_overlay.bundle'
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if (snapshot['archive_commit'] != '190ceb9da7add8d93ea75c3086ace9ce58019d71'
            or digest != 'd81b8efc15468f21fe7dbd0df6ff697414f650de87c98062cc81daea32f1222f'
            or compatibility['exact_default_output_cases'] != 12
            or compatibility['overlay_vision_sha256'] != snapshot['files'][0]['sha256']
            or not snapshot['user_index_unchanged'] or snapshot['source_files_changed']):
        raise RuntimeError('Unexpected recovery/compatibility identity')
    report = dict(source_runtime='/DATA/DATA1/guest3/demo_reproduction_20260915',
                  overlay_snapshot=snapshot,
                  bundle=dict(sha256=digest, bytes=bundle.stat().st_size,
                              required_parent=snapshot['source_head'], standalone=False,
                              server_path='/DATA/DATA1/guest3/storage_cleanup_manifests/t09_reviewed_overlay_20261004.bundle',
                              local_path='.codex_tmp/t11_source_20261004/t09_reviewed_overlay.bundle'),
                  default_vision_compatibility=compatibility,
                  original_runtime_retained=True, full_t09_main_import_complete=False,
                  t16_formal_checkpoint_reload_pending=True, source_cleanup_authorized_by_this_receipt=False)
    dest = ROOT / 'maintenance/storage/T09_OVERLAY_REVIEW_20261004.json'
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(archive_commit=snapshot['archive_commit'],
                         default_compatibility_cases=12, runtime_migration_complete=False)))


if __name__ == '__main__':
    main()
