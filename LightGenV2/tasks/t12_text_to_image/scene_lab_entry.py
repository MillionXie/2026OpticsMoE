"""SDK-free TRAIN cohort preparation/inspection; no real capture command."""
import argparse
import json
from pathlib import Path
from .scene_lab_protocol import Ledger,identity,sha,put
from .scene_lab_native import F9,F6


def train_requests(manifest,reference_root,count=2):
    rows=json.loads(Path(manifest).read_text(encoding='utf-8'));root=Path(reference_root).resolve()
    if not rows or any(r.get('split')!='train'for r in rows):raise ValueError('TRAIN-only input manifest required')
    selected=[]
    # Deterministic category-round-robin, independent of TEST or performance.
    groups={k:[r for r in rows if r['category']==k]for k in sorted({r['category']for r in rows})}
    if not 1<=count<=len(rows):raise ValueError('Invalid cohort size')
    for index in range(max(map(len,groups.values()))):
        for category,items in groups.items():
            if index<len(items):selected.append(items[index])
            if len(selected)==count:break
        if len(selected)==count:break
    result=[]
    for row in selected:
        path=(root/row['reference']).resolve()
        if not path.is_relative_to(root)or sha(path)!=row['reference_sha256']:raise ValueError('Reference mismatch')
        key=dict(reference_sha256=row['reference_sha256'],prompt=row['prompt'],split='train')
        result.append(dict(sample_id=identity(key)[:24],split='train',reference=row['reference'],
            reference_sha256=row['reference_sha256'],prompt=row['prompt'],category=row['category'],seed=int(identity(key)[:8],16)))
    if len({r['sample_id']for r in result})!=len(result):raise ValueError('Duplicate source/prompt')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['inspect','init'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--session',type=Path)
    p.add_argument('--train-manifest',type=Path);p.add_argument('--reference-root',type=Path)
    p.add_argument('--count',type=int,default=2);p.add_argument('--source-commit',required=True)
    a=p.parse_args()
    if a.mode=='init':
        requests=train_requests(a.train_manifest,a.reference_root,a.count)
        session=json.loads(a.session.read_text(encoding='utf-8'))
        ledger=Ledger(a.out,session,dict(F9=F9,F6=F6,source_commit=a.source_commit),requests,create=True)
        print(json.dumps(dict(initialized=True,requests=len(requests),ledger_sha256=ledger.key,capture_started=False)))
    else:
        c=json.loads((a.out/'contract.json').read_text());ledger=Ledger(a.out,c['session'],c['models'],c['requests'])
        available={}
        for stage in c['stages']:
            available[stage]=sum((ledger.folder(stage,r)/'receipt.json').exists()for r in c['requests'])
        print(json.dumps(dict(schema=c['schema'],requests=len(c['requests']),receipts=available,no_SDK=True,
                              receipt_count_is_not_validation=True,calibration_scope=c['session']['scope'])))


if __name__=='__main__':main()
