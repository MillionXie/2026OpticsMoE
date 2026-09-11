"""Import only missing inference assets from a complete old ABO project; no sessions/configs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
CHECKPOINT='a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);a=p.parse_args()
    source=a.source.resolve()
    if source==ROOT:raise ValueError('Source must be a different complete ABO project')
    if digest(source/'assets/best_checkpoint.pt')!=CHECKPOINT:raise ValueError('Wrong checkpoint; original 80.583% reference weights required')
    names=['runtime','models','assets/best_checkpoint.pt','assets/test_dataset'];files=[]
    for name in names:
        path=source/name
        if not path.exists():raise FileNotFoundError(path)
        for f in ([path] if path.is_file() else path.rglob('*')):
            if not f.is_file() or '__pycache__' in f.parts:continue
            rel=f.relative_to(source);dest=ROOT/rel
            if dest.exists():
                if digest(f)!=digest(dest):raise FileExistsError('Different file already exists: '+str(dest))
            else:files.append((f,dest))
    need=sum(f.stat().st_size for f,_ in files)
    if shutil.disk_usage(ROOT).free<need+2*1024**3:raise RuntimeError('Insufficient disk for import + 2 GiB reserve')
    for f,dest in files:
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        if digest(f)!=digest(dest):raise RuntimeError('Copy checksum mismatch')
    report={'source':str(source),'checkpoint_sha256':CHECKPOINT,'new_files':len(files),'copied_bytes':need,
            'old_sessions_and_hardware_config_copied':False}
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/abo_import.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__':main()
