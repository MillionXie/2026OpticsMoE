"""Reconcile two read-only T11 receipts. No model, dataset decode, writes or remote calls."""
import argparse
import json
from pathlib import Path


def payload(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    return json.loads(value['stdout'])


def inspect(metadata, content):
    assets={r['path']:r for r in content['assets']}
    errors=[]
    bindings={p:set(r.get('expected_sha256',[])) for p,r in assets.items()}
    manifest_checks=[]
    for row in metadata['records']:
        if not row['matches_registered']:
            errors.append('metadata changed: '+row['path'])
        m=row['metadata'];cmd=m.get('command',[])
        if '--data' in cmd and m.get('data_sha256'):
            p=cmd[cmd.index('--data')+1]
            if p in bindings:bindings[p].add(m['data_sha256'])
        if '--manifest' in cmd and isinstance(m.get('manifest'),dict):
            p=cmd[cmd.index('--manifest')+1]
            if p in assets:
                ok=assets[p].get('manifest')==m['manifest']
                manifest_checks.append(dict(path=p,run=row['path'],matches_embedded_manifest=ok))
                if not ok:errors.append('embedded manifest differs: '+p)
        expected=m.get('manifests',[])
        paths=[p for p in cmd if isinstance(p,str) and p in assets and p.endswith('manifest.json')]
        if isinstance(expected,list) and len(paths)==len(expected):
            for p,original in zip(paths,expected):
                ok=assets[p].get('manifest')==original
                manifest_checks.append(dict(path=p,run=row['path'],matches_embedded_manifest=ok))
                if not ok:errors.append('embedded manifest differs: '+p)
    for p,row in assets.items():
        if not row.get('exists'):errors.append('missing: '+p)
        if bindings[p] and bindings[p]!={row['sha256']}:
            errors.append('data hash conflict: '+p)
    npz=[dict(path=p,sha256=r['sha256'],matches_historical=bool(bindings[p]))
         for p,r in assets.items() if p.endswith('.npz')]
    return dict(metadata_records=len(metadata['records']),files=len(assets),
                npz=npz,embedded_manifest_comparisons=len(manifest_checks),
                embedded_manifest_paths=sorted({r['path'] for r in manifest_checks}),
                errors=sorted(set(errors)),read_only=True,
                scope='Stored receipt reconciliation; no fresh remote hash or pixel/split audit')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata-receipt',required=True)
    p.add_argument('--content-receipt',required=True)
    args=p.parse_args()
    result=inspect(payload(args.metadata_receipt),payload(args.content_receipt))
    print(json.dumps(result,indent=2))
    raise SystemExit(bool(result['errors']))
