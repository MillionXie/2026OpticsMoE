"""Post-training global bypass and seeded randomization on the selected model."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

if __package__:
    from .data import batch, groups, load_inputs, sha256
    from .model import GlobalStage, load_lightgen
    from .run import ROOT, ensure_idle_gpu, regression_metrics, write
else:
    from data import batch, groups, load_inputs, sha256
    from model import GlobalStage, load_lightgen
    from run import ROOT, ensure_idle_gpu, regression_metrics, write


STAGES={'vision':('frame_global',),'joint':('video_global',),
        'both':('frame_global','video_global')}


def identity_global(tokens):
    return tokens,tokens.new_zeros(())


def tensor_fingerprint(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def ablation_metrics(prediction,target):
    if np.ptp(prediction):
        return regression_metrics(prediction,target)
    difference=np.asarray(prediction)-np.asarray(target)
    return {'n':len(target),'srcc':None,'krcc':None,'plcc':None,
            'rmse':float(np.sqrt(np.mean(difference**2))),
            'mae':float(np.abs(difference).mean()),'constant_prediction':True}


@torch.inference_mode()
def evaluate_condition(model,payload,device,out):
    indices,valid=groups(payload,'test')
    records=[]
    for start in range(0,len(indices),4):
        ii=indices[start:start+4];vv=valid[start:start+4]
        item=batch(payload,ii,vv,device)
        prediction=model(item['vision'])['prediction'][item['valid']].cpu().tolist()
        target=item['target'][item['valid']].cpu().tolist()
        records.extend({'sample_id':payload['ids'][index],'target':y,'prediction':p}
                       for index,y,p in zip(ii[vv].tolist(),target,prediction))
    if len(records)!=558 or len({row['sample_id'] for row in records})!=558:
        raise ValueError('TEST identity or padding population differs')
    metrics=ablation_metrics([r['prediction'] for r in records],[r['target'] for r in records])
    out.mkdir(parents=True)
    with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample_id','target','prediction'])
        writer.writeheader();writer.writerows(records)
    write(out/'metrics.json',metrics)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(7100,7110)))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--allow-shared-gpu',action='store_true')
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if len(args.seeds)!=len(set(args.seeds)):
        raise ValueError('Random seeds must be unique')
    if args.device.startswith('cuda') and not args.allow_shared_gpu:
        ensure_idle_gpu()
    torch.set_num_threads(4)
    weights_sha256=sha256(args.weights)
    model,metadata=load_lightgen(args.weights,args.device)
    payload=load_inputs(json.loads(args.paths.read_text('utf-8')))
    if payload['identity']['manifest']['sha256']!=metadata['dataset']['manifest']['sha256']:
        raise ValueError('Dataset manifest differs')
    original=tensor_fingerprint(model.state_dict())
    args.out.mkdir(parents=True)
    rows=[]
    learned=evaluate_condition(model,payload,args.device,args.out/'learned')
    rows.append({'kind':'learned','stage':'both','seed':None,**learned})
    if abs(learned['srcc']-metadata['metrics']['srcc'])>1e-5:
        raise ValueError('Reference differs from the selected checkpoint')
    for kind in ('bypass','random_phase','random_global'):
        for stage_label,names in STAGES.items():
            for seed in ([None] if kind=='bypass' else args.seeds):
                identities={}
                with ExitStack() as stack:
                    for name in names:
                        stage=getattr(model,name)
                        if kind=='bypass':
                            stack.enter_context(patch.object(stage,'forward',identity_global))
                            continue
                        stage_index=('frame_global','video_global').index(name)
                        generator=torch.Generator().manual_seed(seed+1000*stage_index)
                        phase=(2*math.pi*torch.rand(16,111,111,generator=generator)).to(args.device)
                        identities[name]={'phase_sha256':tensor_fingerprint({'phase':phase})}
                        if kind=='random_global':
                            devices=[stage.raw_global_phase.device.index] if stage.raw_global_phase.is_cuda else []
                            with torch.random.fork_rng(devices=devices):
                                torch.manual_seed(seed+10000+1000*stage_index)
                                replacement=GlobalStage(model.cfg,stage.frame_level)
                            replacement=replacement.to(args.device).eval()
                            identities[name]['random_module_sha256']=tensor_fingerprint(replacement.state_dict())
                            stack.enter_context(patch.object(model,name,replacement))
                            stage=replacement
                        stack.enter_context(patch.object(stage,'phase',lambda *,wrapped=False,value=phase:value))
                    case=f'{kind}_{stage_label}'+('' if seed is None else f'_{seed}')
                    metrics=evaluate_condition(model,payload,args.device,args.out/case)
                rows.append({'kind':kind,'stage':stage_label,'seed':seed,**metrics})
                write(args.out/case/'randomization.json',identities)
                write(args.out/'status.json',{'state':'running','conditions_finished':len(rows)})
                print(json.dumps({'case':case,'srcc':metrics['srcc']}),flush=True)
    if tensor_fingerprint(model.state_dict())!=original or sha256(args.weights)!=weights_sha256:
        raise ValueError('Selected state changed during ablation')
    summary=[]
    for kind,stage in sorted({(row['kind'],row['stage']) for row in rows}):
        scores=[row['srcc'] for row in rows if row['kind']==kind and row['stage']==stage and row['srcc'] is not None]
        summary.append({'kind':kind,'stage':stage,'defined_seeds':len(scores),
            'mean_srcc':float(np.mean(scores)) if scores else None,
            'std_srcc':float(np.std(scores,ddof=1)) if len(scores)>1 else 0.,
            'minimum_srcc':min(scores) if scores else None,'maximum_srcc':max(scores) if scores else None})
    with (args.out/'all_conditions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['kind','stage','seed','n','srcc','krcc','plcc','rmse','mae'],extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    with (args.out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    write(args.out/'summary.json',summary)
    write(args.out/'protocol.json',{
        'role':'post_training_global_ablation_on_original_TEST_development',
        'checkpoint_sha256':sha256(args.weights),'model_state_sha256':original,
        'trained_parameters_modified':False,'retraining':False,'seeds':args.seeds,
        'bypass':'identity on the entire GlobalStage input/output token tensor',
        'random_phase':'independent uniform physical phase in [0,2*pi), all learned mappings retained',
        'random_global':'default constructor initialization of GlobalStage mappings/readout plus uniform physical phase',
        'unchanged_components':'all modules outside the named GlobalStage replacements',
        'padding_excluded':True,'videos':558,'data_identity':payload['identity'],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'script_sha256':sha256(Path(__file__)),
        'command':sys.argv,
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':args.device}})
    write(args.out/'status.json',{'state':'completed','conditions':len(rows)})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
