"""One equal-task full-batch TRAIN-only convex LG B readout fit, frozen CCD.

Removes raw background conditioning and task-count imbalance in fit coordinates.
Coordinates fold into the same original7840 Linear; no inference preprocessing.
"""
import argparse,copy,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .refine_measured_ce import contrast_transform
from .fit_gated_linear_teacher import full_covariance_transform
from .terminal_fit import evaluate,brief,score
from .train_native import sha,put


def task_weights(tasks):
    return torch.tensor([1/(len(set(tasks))*int((tasks==n).sum())) for n in tasks],dtype=torch.float64)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261025)
    cr=json.loads((a.capture/'report.json').read_text());model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if not cr['passed'] or cr['pilot'] or not cr['SDK_released'] or sha(a.capture/'features.npz')!=cr['features_sha256'] or cp['architecture']!='moe' or cp['stage']!=2 or cp['measured_features_sha256']!=cr['features_sha256']:raise ValueError('Complete frozen measured LG B required')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features']);y=torch.tensor(d['labels']);tasks=d['tasks'].copy();splits=d['splits'].copy();ids=d['source_indices'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if len(tr)!=1300 or {n:int((tasks[tr]==n).sum()) for n in set(tasks)}!={'eurosat':300,'clevr':1000} or any(int((tasks[te]==n).sum())!=1000 for n in set(tasks)):raise ValueError('Original fixed TRAIN/EVAL identities only')
    transform,conditioning=full_covariance_transform(x[tr],contrast_transform(x[tr]));z=x[tr].double()@transform.T;weights=task_weights(tasks[tr]);weight=torch.nn.Parameter(torch.zeros((10,784),dtype=torch.float64))
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();baseline=evaluate(model.shared_head,x.float(),y,tasks,te);best_value=score(baseline);best_weight=model.shared_head.weight.detach().clone();best_step=0;history=[]
    put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,capture_features_SHA256=cr['features_sha256'],task_TRAIN_counts=dict(eurosat=300,clevr=1000),fit='float64 equal-task smoothedCE .02 +coordinate weightL2 1e-4, zero readout initialization plus parent E0 fallback',maximum_steps=240,maximum_seconds=300,conditioning=conditioning,original_parameters=7840,new_inference_layers=0,TEST_gradients=False,teacher_TEST_inputs=False,TEST_selected_development_metric=True,independent_generalization_claim=False,SDK=False,GPU=False,CPU_threads=2,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__)))
    opt=torch.optim.LBFGS([weight],lr=1.,max_iter=1,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11)
    def closure():
        opt.zero_grad(set_to_none=True);loss=(weights*F.cross_entropy(z@weight.T,y[tr],label_smoothing=.02,reduction='none')).sum()+.0001*weight[:,:-1].square().sum()/10
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite pure TRAIN objective')
        loss.backward();return loss
    for step in range(1,241):
        opt.step(closure);loss=float(closure().detach());gradient=float(weight.grad.norm())
        if step==1 or step%10==0 or gradient<1e-7:
            with torch.no_grad():model.shared_head.weight.copy_((weight.detach()@transform).float())
            actual=evaluate(model.shared_head,x.float(),y,tasks,te);value=score(actual)
            if value>best_value:best_value=value;best_step=step;best_weight=model.shared_head.weight.detach().clone()
            history.append(dict(step=step,TRAIN_loss=loss,gradient_norm=gradient,scores=brief(actual),best_step=best_step));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='TRAIN_balanced_covariance',**history[-1],updated=time.time()))
        if gradient<1e-7 or time.monotonic()-started>=300:break
    rows={}
    for role,w in [('best',best_weight),('last',(weight.detach()@transform).float())]:
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before:raise RuntimeError('Frozen optical prefix changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,TEST_gradients=False,teacher_TEST_inputs=False,TEST_selected_development_metric=True);path=a.out/(role+'_checkpoint.pt');torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,w):raise RuntimeError('Strict CPU replay changed')
        actual=evaluate(loaded.shared_head,x.float(),y,tasks,te);rows[role]=brief(actual);put(a.out/(role+'_test.json'),dict(scores=actual,source_indices=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),TEST_selected_development_metric=True))
    put(a.out/'report.json',dict(passed=True,baseline=brief(baseline),best=rows['best'],last=rows['last'],best_step=best_step,retained_parent=best_step==0,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),strict_CPU_reloaded=True,frozen_upstream=True,frozen_upstream_SHA256=before,original_parameters=7840,SDK=False,GPU=False,TEST_gradients=False,teacher_TEST_inputs=False,TEST_selected_development_metric=True,independent_generalization_claim=False,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
