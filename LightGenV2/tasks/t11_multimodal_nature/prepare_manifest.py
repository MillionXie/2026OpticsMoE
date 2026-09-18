"""Create a reproducible, hash-audited manifest for SEN12MS or SONYC-UST.
No raw data are copied into the repository. The script intentionally fails on
missing expected metadata so an incomplete download cannot silently become a
formal run input.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

def sha256(p: Path, chunk=1024*1024):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(chunk), b''): h.update(b)
    return h.hexdigest()

def files(root, suffixes):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in suffixes)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset', choices=['sen12ms','sonyc_ust'], required=True); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True); ap.add_argument('--max-files', type=int, default=0)
    a=ap.parse_args(); root=Path(a.root).resolve(); assert root.is_dir(), root
    if a.dataset=='sen12ms':
        expected=['train_list.txt','test_list.txt']
        found={p.name:p for p in root.rglob('*') if p.is_file()}
        missing=[x for x in expected if x not in found]
        if missing: raise SystemExit('missing official split files: '+', '.join(missing))
        cand=files(root,{'.txt','.pkl','.json','.md','.pdf','.tif','.tiff','.npy'})
        license_name='CC BY 4.0 (verify repository/data release text in evidence)'
        source='https://github.com/schmitt-muc/SEN12MS'
        checks={'split_files':{n:{'path':str(found[n].relative_to(root)),'sha256':sha256(found[n]),'bytes':found[n].stat().st_size} for n in expected}}
    else:
        cand=files(root,{'.csv','.txt','.json','.md','.pdf','.wav','.flac','.mp3'})
        names={p.name.lower() for p in cand}
        if not any('annotation' in n for n in names): raise SystemExit('missing SONYC annotation CSV')
        license_name='CC BY 4.0 (Zenodo record 3966543)'; source='https://zenodo.org/records/3966543'; checks={}
    if a.max_files: cand=cand[:a.max_files]
    rec=[]
    for p in cand:
        rec.append({'path':str(p.relative_to(root)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha256(p)})
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema_version':1,'dataset':a.dataset,'root_name':root.name,'source_url':source,'license':license_name,'files':rec,'required_checks':checks}
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'dataset':a.dataset,'files':len(rec),'manifest':str(out),'manifest_sha256':sha256(out)},ensure_ascii=False))
if __name__=='__main__': main()
