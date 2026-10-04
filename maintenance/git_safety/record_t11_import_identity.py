"""Generate audited T11 source/asset receipts; never copy model or dataset payloads."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    transport = json.loads((ROOT / '.codex_tmp/t11_source_20261004/assets_transport.json').read_text(encoding='utf-8'))
    if transport['exit_code']:
        raise RuntimeError('Remote asset audit failed')
    assets = json.loads(transport['stdout'])
    additions = json.loads((ROOT / 'maintenance/storage/T11_PINNED_ADDITIONS_20261004.json').read_text(encoding='utf-8'))
    source = {'source_head': additions['source_commit'], 'files': [
        {'path': row['path'], 'source_blob_sha256': row['sha256'], 'bytes': row['bytes']}
        for row in additions['paths']], 'additional_existing_dependencies': [],
        'scientific_source_changed': False,
        'readme_difference': 'Main task README adds an authored provenance banner to complete original server protocols; source files stay byte-identical',
        'excluded_report_images': 'PNG originals retained at original server/local locations; not new Git payloads'}
    for relative, value in (
        ('maintenance/storage/T11_ASSET_IDENTITY_20261004.json', assets),
        ('LightGenV2/tasks/t11_lifelong_optics/source_import_20261004.json', source)):
        target = ROOT / relative
        if target.exists():
            raise FileExistsError('Earlier identity protected: ' + relative)
        target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'source_files':len(source['files']), 'asset_records':len(assets['files']),
                      'weights_hashed':sum(row['kind']=='weight' for row in assets['files'])}))


if __name__ == '__main__':
    main()
