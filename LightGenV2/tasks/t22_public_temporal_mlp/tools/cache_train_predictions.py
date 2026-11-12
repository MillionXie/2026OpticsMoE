"""Cache TRAIN-only predictions for output distillation during router tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

if __package__:
    from . import _paths
else:
    import _paths

if __package__ and __package__ != 'tools':
    from .data import batch, groups, load_inputs, sha256
    from ..model import load_lightgen
    from .run import ROOT, regression_metrics, write
else:
    from data import batch, groups, load_inputs, sha256
    from model import load_lightgen
    from run import ROOT, regression_metrics, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    model,metadata=load_lightgen(args.weights,args.device)
    payload=load_inputs(json.loads(args.paths.read_text('utf-8')))
    if payload['identity']['manifest']['sha256']!=metadata['dataset']['manifest']['sha256']:
        raise ValueError('Teacher manifest differs')
    indices,valid=groups(payload,'train')
    total=torch.zeros(len(payload['ids']),dtype=torch.float64)
    count=torch.zeros(len(payload['ids']),dtype=torch.int64)
    shifts=(0,4,8,12)
    with torch.inference_mode():
        for shift in shifts:
            for start in range(0,len(indices),4):
                ii=indices[start:start+4].roll(shift,1)
                vv=valid[start:start+4].roll(shift,1)
                item=batch(payload,ii,vv,args.device)
                prediction=model(item['vision'])['prediction']
                total.index_add_(0,ii[vv],prediction[item['valid']].cpu().double())
                count.index_add_(0,ii[vv],torch.ones(int(vv.sum()),dtype=torch.int64))
    keep=torch.tensor([split=='train' for split in payload['splits']])
    if not bool((count[keep]==len(shifts)).all()) or bool(count[~keep].any()):
        raise ValueError('TRAIN/padding population differs')
    scores=(total[keep]/count[keep]).float()
    record={
        'sample_ids':[name for name,flag in zip(payload['ids'],keep) if flag],
        'target_names':['temporal'],
        'predictions':scores[:,None],
        'inference_dependency':False,
        'teacher_checkpoint':str(args.weights),
        'teacher_checkpoint_sha256':sha256(args.weights),
        'slot_shifts_averaged':list(shifts),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_script_sha256':sha256(Path(__file__)),
        'manifest_sha256':payload['identity']['manifest']['sha256'],
        'TRAIN_metrics':regression_metrics(scores,payload['target'][keep]),
        'TEST_entries':0,
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(record,args.out)
    write(args.out.with_suffix('.json'),{key:value for key,value in record.items()
                                      if key not in ('sample_ids','predictions')})
    print(json.dumps(record['TRAIN_metrics']))


if __name__=='__main__':
    main()
