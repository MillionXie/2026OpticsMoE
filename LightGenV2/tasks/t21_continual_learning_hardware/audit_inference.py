"""Immutable same-subset original reference and strict CPU native replays."""
import argparse
import json
from pathlib import Path
import time

import torch

from .train_native import TASKS,sha,put,load_data,evaluate,summary,gpu_gate
from .hardware_bridge import load_checkpoint


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=('reference','native_cpu'),required=True)
    p.add_argument('--test',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run',type=Path);p.add_argument('--phase',choices=('A','B','C','D'))
    p.add_argument('--gpu-uuid');a=p.parse_args();torch.set_num_threads(2)
    a.out.mkdir(parents=True,exist_ok=False);began=time.time();data=load_data(a.test,'test');rows=[]
    if a.mode=='reference':
        from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.model import DirectCCDOptics
        cfg=json.loads((Path(__file__).parent/'config_native_memory300_20261024.json').read_text())
        gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0')
        source=json.loads((Path(__file__).parent/'source_identity.json').read_text())
        for i,identity in enumerate(source['models']):
            phase=i%4+1;arch='moe' if i<4 else 'd2nn'
            if sha(identity['checkpoint'])!=identity['sha256']:raise ValueError('Reference checkpoint mismatch')
            original_config=json.loads(Path(identity['checkpoint']).with_name('config.json').read_text())
            keys=('router_side','router_pitch','output_side','x_pitch','y_pitch','routing_temperature','activation_order')
            model=DirectCCDOptics(arch,**{k:original_config[k] for k in keys if k in original_config}).to(device)
            checkpoint=torch.load(identity['checkpoint'],map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model'],strict=True)
            scores=evaluate(model,data,TASKS[:phase],device,4)
            put(a.out/(arch+'_'+chr(64+phase)+'.json'),dict(scores=scores,
                checkpoint_sha256=identity['sha256'],scope='original 17um graph, same stratified1000 TEST',
                historical_replay='full' if arch=='moe' else '300 old records, 50 cycles'))
            rows.append(dict(architecture=arch,phase=chr(64+phase),scores=summary(scores)))
            put(a.out/'progress.json',dict(completed=len(rows),latest=rows[-1]));del model,checkpoint
        put(a.out/'report.json',dict(passed=True,rows=rows,gpu=gpu,seconds=time.time()-began))
    else:
        device=torch.device('cpu')
        if a.run is None:raise ValueError('Native run required')
        for name in ((a.phase,) if a.phase else ('A','B','C','D')):
            folder=a.run/name;receipt=json.loads((folder/'completion.json').read_text())
            model,checkpoint=load_checkpoint(folder/'best_checkpoint.pt',receipt['best_sha256'])
            scores=evaluate(model,data,TASKS[:checkpoint['stage']],device,4)
            stored=json.loads((folder/'selected_test.json').read_text())['scores']
            errors={}
            for task,value in scores.items():
                agreement=sum(x==y for x,y in zip(value['predictions'],stored[task]['predictions']))/1000
                max_error=max(abs(x-y) for r,s in zip(value['probabilities'],stored[task]['probabilities']) for x,y in zip(r,s))
                errors[task]=dict(argmax_agreement=agreement,probability_max_error=max_error)
                if agreement!=1 or max_error>.002:raise RuntimeError('Strict CPU/GPU mismatch: '+repr(errors))
            put(a.out/(name+'.json'),dict(scores=scores,strict_CPU=True,comparison=errors,
                checkpoint_sha256=receipt['best_sha256']))
            rows.append(dict(phase=name,comparison=errors,scores=summary(scores)))
            put(a.out/'progress.json',dict(completed=len(rows),latest=rows[-1]))
        put(a.out/'report.json',dict(passed=True,rows=rows,strict_CPU=True,seconds=time.time()-began))


if __name__=='__main__':main()
