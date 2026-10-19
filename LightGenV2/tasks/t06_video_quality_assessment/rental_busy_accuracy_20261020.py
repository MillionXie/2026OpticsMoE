"""Independent frozen full558 accuracy; never replace complete-window costs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from . import rental_precision_recheck as base
from . import rental_busy_resource_20261020 as busy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args()
    if not 0<args.deadline-time.time()<=7200 or os.environ.get('CUDA_VISIBLE_DEVICES')!=busy.GPU:
        raise ValueError('Authorized finite exact physical GPU only')
    args.model='kimi';args.precision='fp32';args.batch=1;args.stage='independent-accuracy'
    args.gpu_uuid=busy.GPU;args.power_limit_w=400;args.utilization_source=None
    folder=busy.ROOT/'accuracy_kimi_fp32_batch1'
    if folder.exists():raise RuntimeError('Preserve prior independent accuracy')
    folder.mkdir();guard=base.Guard(args)
    identity=dict(model=args.model,precision=args.precision,batch=args.batch,gpu_uuid=busy.GPU,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),
        sources={**base.sources(),str(Path(__file__).relative_to(base.REPO)):base.old.sha(Path(__file__))},
        accuracy_only=True,costs_are_audit_not_complete_window_comparison=True)
    try:
        import torch
        guard.check();torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        model,processor,prompt,first,transfer,compute=base.runtime(args,guard,identity)
        base.old.write(folder/'identity.json',identity)
        prepare=lambda rows:base.prior.prepare_batch(args.model,rows,processor,prompt)
        base.evaluate(args,compute,prepare,transfer,first,folder/'evaluation',guard,1,identity)
        if base.old.verify_assets(Path('/data/models')/args.model)!=identity['assets']:raise RuntimeError('Assets changed')
        if base.old.sha(Path(__file__))!=identity['sources'][str(Path(__file__).relative_to(base.REPO))]:raise RuntimeError('Entry changed')
        guard.check();base.old.write(folder/'status.json',dict(status='complete',time=time.time(),test_videos=558))
    except BaseException as error:
        base.old.write(folder/'status.json',dict(status='failed',error=repr(error),time=time.time()));raise
    finally:
        base.old.write(folder/'identity.json',identity);guard.close(folder/'process_audit.json')


if __name__=='__main__':main()
