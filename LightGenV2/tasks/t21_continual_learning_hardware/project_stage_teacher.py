"""One fixed weak-TRAIN function projection, original7840 Linear unchanged.

This projection inherits partial TEST information from its sealed teacher.
No EVAL inputs in fitting, no independent TEST claim, no new inference layer.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_native import sha,put
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .fit_augmented_residual_teacher import residual_teacher_from_artifact
from .distill_offline_teacher_projection import solve_projection
from .fit_B_balanced_residual_teacher import whole_stage_selection
from .fit_gated_linear_teacher import DISCLOSURE
from .terminal_fit import evaluate,brief


def held_TRAIN(ids,y,tasks):
    rng=np.random.default_rng(20261025);held=[]
    for name in sorted(set(tasks)):
        pool=np.flatnonzero(tasks==name)
        if name=='eurosat':
            for label in range(10):
                ix=pool[y[pool]==label];held.extend(rng.choice(ix,max(1,len(ix)//5),replace=False).tolist())
        else:
            groups=np.unique(ids[pool]//2);chosen=rng.choice(groups,len(groups)//5,replace=False);held.extend(pool[np.isin(ids[pool]//2,chosen)].tolist())
    return np.array(sorted(held),dtype=int)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','teacher_run','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261025)
    cr=json.loads((a.capture/'report.json').read_text());r=json.loads((a.teacher_run/'report.json').read_text());pr=json.loads((a.teacher_run/'protocol.json').read_text());profile=pr['profile']
    if not cr['passed'] or not cr['SDK_released'] or cr['pilot'] or sha(a.capture/'features.npz')!=cr['features_sha256'] or pr['capture_features_SHA256']!=cr['features_sha256']:raise ValueError('Exact full immutable capture required')
    if not r['passed'] or not r['teacher_quality_gate_passed'] or r['valid_for_formal_test_table'] is not False or sha(a.teacher_run/'offline_teacher.pt')!=r['teacher_sha256'] or pr['parent_SHA256']!=a.parent_sha256:raise ValueError('Qualified sealed same-parent partial teacher required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp['architecture']!='d2nn' or cp['stage']!=profile['stage'] or cp['measured_features_sha256']!=cr['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Original measured phase required')
    artifact=torch.load(a.teacher_run/'offline_teacher.pt',map_location='cpu',weights_only=False);teacher=residual_teacher_from_artifact(artifact);transform=artifact['transform']
    for v in teacher.parameters():v.requires_grad_(False)
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');names=sorted(profile['development_floor']);selected=pr['selection_source_ids'];exposed=np.array([i for i in te if int(ids[i]) in selected[str(tasks[i])]],dtype=int);unex=np.setdiff1d(te,exposed)
    if len(tr)!=1000+300*(cp['stage']-1) or any(int((tasks[te]==n).sum())!=1000 for n in names):raise ValueError('Full phase source count changed')
    held=held_TRAIN(ids[tr],y[tr].numpy(),tasks[tr]);fit=np.setdiff1d(np.arange(len(tr)),held)
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,teacher_SHA256=r['teacher_sha256'],capture_features_SHA256=cr['features_sha256'],
        method='Fixed24 weak TRAIN perturbations .001/.005/.01 TRAINstd, function projection; four ridge strengths by held TRAIN KL only',
        held_TRAIN_source_ids={n:ids[tr][held][tasks[tr][held]==n].tolist() for n in names},selection_source_ids=selected,
        original_student_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,max_seconds=240,CPU_threads=2,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),**DISCLOSURE))
    def stats(indices):
        source=x[tr][indices];tt=tasks[tr][indices];std=source.std(0,unbiased=False);gram=torch.zeros(784,784,dtype=torch.float64);cross=torch.zeros(784,10,dtype=torch.float64)
        weights=torch.tensor([1/(24*len(names)*int((tt==n).sum())) for n in tt],dtype=torch.float64)
        for repeat in range(24):
            if time.monotonic()-started>240:raise RuntimeError('Finite projection bound reached')
            values=source if repeat==0 else (source+torch.randn_like(source)*std*(.001,.005,.01)[repeat%3]).clamp_min(0)
            values=values/values.mean(1,keepdim=True)
            for ix in torch.arange(len(source)).split(256):
                z=values[ix]@transform.T;target=teacher(z);target=target-target.mean(1,keepdim=True);zz=z.double();w=weights[ix,None]
                gram+=zz.T@(w*zz);cross+=zz.T@(w*target.double())
        return gram,cross
    gram,cross=stats(fit);target=teacher(x[tr][held]@transform.T);choice=[]
    for ridge in (.0001,.001,.01,.1):
        w=solve_projection(gram,cross,ridge);logits=(x[tr][held]@transform.T).double()@w.T;loss=F.kl_div(F.log_softmax(logits/2,1),F.softmax(target.double()/2,1),reduction='none').sum(1)*4
        metric=float(np.mean([float(loss[tasks[tr][held]==n].mean()) for n in names]));choice.append(dict(ridge=ridge,held_TRAIN_teacher_KL=metric))
    ridge=min(choice,key=lambda v:(v['held_TRAIN_teacher_KL'],-v['ridge']))['ridge'];gram,cross=stats(np.arange(len(tr)));w=solve_projection(gram,cross,ridge);folded=(w@transform.double()).float()
    baseline=evaluate(model.shared_head,x,y,tasks,te);model.shared_head.weight.copy_(folded);candidate=evaluate(model.shared_head,x,y,tasks,te);retain=whole_stage_selection(candidate,profile['development_floor'])<=whole_stage_selection(baseline,profile['development_floor']);result={}
    for role,weight in [('best',cp['model']['shared_head.weight'].clone() if retain else folded),('last',folded)]:
        model.shared_head.weight.copy_(weight)
        if upstream_state(model)!=before:raise RuntimeError('Frozen upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**DISCLOSURE);path=a.out/(role+'_checkpoint.pt');torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,weight):raise RuntimeError('Original graph reload differs')
        scores={n:evaluate(loaded.shared_head,x,y,tasks,ix) for n,ix in [('TRAIN',tr),('all_development',te),('exposed_development',exposed),('unexposed_development',unex)]};result[role]={n:brief(v) for n,v in scores.items()}
        put(a.out/(role+'_development.json'),dict(scores=scores,source_indices=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),**DISCLOSURE))
    put(a.out/'report.json',dict(passed=True,baseline=brief(baseline),candidate=brief(candidate),retained_parent=retain,best=result['best'],last=result['last'],original_student_parameters=7840,new_inference_layers=0,teacher_sha256=r['teacher_sha256'],strict_CPU_reloaded=True,frozen_upstream=True,ridge_selection=choice,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),SDK=False,GPU=False,seconds=time.monotonic()-started,**DISCLOSURE))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
