"""Verify two archives then install into a NEW directory; never overwrite a lab."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe ZIP path')
    return p


def install(base,overlay,expected_overlay,output):
    if output.exists():raise FileExistsError('New project required; old experiments preserved')
    if digest(overlay)!=expected_overlay:raise ValueError('Overlay SHA mismatch')
    with zipfile.ZipFile(overlay) as z:
        source=json.loads(z.read('SHS_SOURCE.json'))
        if digest(base)!=source['base_simulation_zip_sha256']:raise ValueError('Base ZIP SHA mismatch')
        for name,value in source['files'].items():
            safe_name(name)
            if hashlib.sha256(z.read(name)).hexdigest()!=value:raise ValueError('Overlay source mismatch')
        with zipfile.ZipFile(base) as b:
            prefix=b.namelist()[0].split('/')[0]+'/'
            mf=json.loads(b.read(prefix+'MANIFEST.json'))
            for row in mf['files']:
                safe_name(row['path'])
                if hashlib.sha256(b.read(prefix+row['path'])).hexdigest()!=row['sha256']:raise ValueError('Base file mismatch')
            output.mkdir(parents=True)
            for row in mf['files']:
                target=output/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(b.read(prefix+row['path']))
            (output/'MANIFEST.json').write_bytes(b.read(prefix+'MANIFEST.json'))
        for name in list(source['files'])+['SHS_SOURCE.json']:
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    print('INSTALLED',output,source['source_commit'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True)
    p.add_argument('--overlay-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    install(a.base,a.overlay,a.overlay_sha256,a.output)
