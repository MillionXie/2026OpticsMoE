import argparse,json,traceback
from pathlib import Path
from .runtime import ROOT,verify_source,setup,build,require_scope
from .data import prepare,plan,signature,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','full'],default='pilot');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--name',required=True,choices=['A','B_only','M1','M2','D_A','D_B_only','D1']);args=p.parse_args()
    verify_source();require_scope(args.mode,args.name);split=prepare();pre=json.loads((ROOT/'runs/preflight.json').read_text())
    assert pre['status']=='passed' and pre['plan_sha256']==signature(plan()) and pre['split_sha256']==split['split_sha256']
    name=args.name;arch='d2nn' if name.startswith('D') else 'moe';transfer=name in ('M1','M2','D1')
    parent=ROOT/'runs'/args.mode/('seed'+str(args.seed));out=parent/name
    loaded,s=setup(args.seed,out);r,h=build(loaded,s,arch)
    from .train import train
    try:
        train(loaded,r,h,s,split,out,args.seed,args.mode,'transfer' if transfer else 'source',
            domain=1 if 'B_only' in name else 0,variant='all' if name=='M2' else 'reserved',
            source=parent/('D_A' if arch=='d2nn' else 'A')/'selected.pt' if transfer else None)
    except BaseException as e:
        atomic_json(out/'failure.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise
    finally:r.close()

if __name__=='__main__':main()
