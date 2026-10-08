"""Read-only depth diagnosis: phase updates, gradients and training/validation fit."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    archive=Path(__file__).resolve().parents[1]/'adrenal_softsign_code_export_20260915_145336/code'
    sys.path.insert(0,str(archive))
    import torch
    import run_experiment as r
    a.out.mkdir(parents=True,exist_ok=False)
    r.EXP['data_npz']=str(a.data);r.setup();r.setseed(17)
    train,val=r.getdata('train'),r.getdata('val')
    y=train[1].cpu().numpy()
    indices=torch.tensor(np.r_[np.flatnonzero(y==0)[:4],np.flatnonzero(y==1)[:4]])
    metadata=dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=archive,text=True).strip(),
                  source_sha256=r.sha(__file__),data_sha256=r.sha(a.data),python=sys.version,
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(),gradient_batch_ids=train[2][indices].tolist(),
                  gradient_batch_labels=train[1][indices].cpu().tolist(),test_pixels_used=False,
                  scope='read-only best/last re-evaluation and matched balanced training minibatch backward',
                  environment=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    r.save(a.out/'metadata.json',metadata);results=[]
    for source_run in a.runs:
        lock=r.read(source_run/'subset_test_lock.json')
        for variant in lock['variants']:
            name=variant['id'];source=source_run/'runs'/name/'seed17';dest=a.out/name;dest.mkdir()
            with (source/'history.csv').open() as f:history=list(csv.DictReader(f))
            assert [int(x['epoch']) for x in history]==list(range(1,51))
            assert r.sha(source/'best.pt')==lock['checkpoints'][f'runs/{name}/seed17/best.pt']
            r.setseed(17);model=r.build(variant['architecture'],r.CONFIGS[name]).cuda()
            initial={n:v.detach().clone() for n,v in model.named_parameters()}
            assert all('phase' in n for n in initial)
            metrics={};phase_checks={}
            for state in ['initial','best','last']:
                if state!='initial':
                    checkpoint=torch.load(source/(state+'.pt'),map_location='cpu',weights_only=False)
                    model.load_state_dict(checkpoint['model'],strict=True)
                for split,data in [('train',train),('validation',val)]:
                    measured,rows=r.evaluate(model,data,predictions=True)
                    measured.pop('routing',None)
                    metrics[state+'_'+split]=measured
                    if split=='validation':r.csvwrite(dest/(state+'_validation_predictions.csv'),rows)
                model.zero_grad(set_to_none=True)
                prediction=model(train[0][indices]);loss=r.objective(prediction,train[1][indices],model.masks)
                loss.backward()
                gradients={}
                for n,param in model.named_parameters():
                    grad=param.grad
                    assert grad is not None and bool(torch.isfinite(grad).all())
                    delta=param.detach()-initial[n]
                    gradients[n]=dict(shape=list(param.shape),gradient_l2=float(grad.norm()),gradient_rms=float(grad.square().mean().sqrt()),
                                      gradient_nonzero_fraction=float((grad!=0).float().mean()),
                                      update_rms=float(delta.square().mean().sqrt()),update_max=float(delta.abs().max()),
                                      sigmoid_derivative_mean=float((torch.sigmoid(param)*(1-torch.sigmoid(param))).mean()),
                                      raw_abs_gt8_fraction=float((param.abs()>8).float().mean()))
                phase_checks[state]=dict(loss_on_fixed_batch=float(loss),parameters=gradients)
            completed=r.read(source/'completed.json')
            row=dict(variant=name,source_run=str(source_run),selected_epoch=r.read(source/'selection.json')['epoch'],
                     epochs=50,updates=completed['updates'],metrics=metrics,phase_checks=phase_checks,
                     first_epoch=history[0],last_epoch=history[-1],checkpoint_sha256={s:r.sha(source/(s+'.pt')) for s in ['best','last']})
            r.save(dest/'diagnostics.json',row);results.append(row)
            print(json.dumps(dict(variant=name,state='complete',best_validation_auroc=metrics['best_validation']['auroc'])),flush=True)
            del model,initial,prediction,loss,checkpoint;torch.cuda.empty_cache()
    r.save(a.out/'results.json',results);r.save(a.out/'status.json',dict(state='complete',weights_modified=False))


if __name__=='__main__':main()
