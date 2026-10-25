"""Finite C shared-student distillation with conflicting TRAIN gradients projected.

Task-specific linear assistants remain offline. Deployment is ONE original
7840-parameter head; final student queries TRAIN/its weak augmentations only.
Inherited partial TEST-derived information is not an independent test result.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .distill_capacity_matched_stage import TRAIN_queries,convex_objective
from .fit_gated_linear_teacher import DISCLOSURE,full_covariance_transform
from .refine_measured_ce import contrast_transform
from .fit_B_balanced_residual_teacher import whole_stage_selection
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import sha,put

PROFILE=dict(steps=240,maximum_seconds=300,lr=.01,coordinate_delta_L2=.0001,
             original_student_parameters=7840,new_inference_layers=0,TRAIN_only=True)


def project_conflicts(gradients,step):
    originals=[g.detach().clone() for g in gradients];projected=[];negative=0;comparisons=0
    for i,g in enumerate(originals):
        value=g.clone()
        order=[j for j in range(len(originals)) if j!=i]
        if order:shift=step%len(order);order=order[shift:]+order[:shift]
        for j in order:
            other=originals[j];dot=(value*other).sum();comparisons+=1
            if dot<0:value-=dot/other.square().sum().clamp_min(1e-24)*other;negative+=1
        projected.append(value)
    return torch.stack(projected).mean(0),dict(projected_conflicts=negative,comparisons=comparisons)


def goal(scores):
    ref={'eurosat':.562,'clevr':.681,'speech_binary':.522}
    return all(ref[n]-.05-1e-7<=v['accuracy']<ref[n] for n,v in scores.items()) and set(scores)==set(ref)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','taskwise_run','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Inherited TEST information must be disclosed')
    torch.set_num_threads(2);torch.manual_seed(20261025)
    cr=json.loads((a.capture/'report.json').read_text());tp=json.loads((a.taskwise_run/'protocol.json').read_text());trp=json.loads((a.taskwise_run/'report.json').read_text());teacher_path=a.taskwise_run/'diagnostic_taskwise_weights.npz'
    if not cr['passed'] or cr['pilot'] or not cr['SDK_released'] or sha(a.capture/'features.npz')!=cr['features_sha256']:raise ValueError('Sealed full capture required')
    cfg=json.loads(Path(__file__).with_name('config_capacity_matched_distillation_20261025.json').read_text())
    if tp['mode']!='taskwise_diagnostic' or tp['profile']!=cfg or tp['capture_features_SHA256']!=cr['features_sha256'] or tp['parent_SHA256']!=a.parent_sha256 or not trp['passed'] or not trp['no_task_head_splicing'] or trp['valid_for_formal_test_table'] is not False:raise ValueError('Exact same-scope taskwise assistants required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);frozen=upstream_state(model)
    if cp['architecture']!='d2nn' or cp['stage']!=3 or cp['measured_features_sha256']!=cr['features_sha256']:raise ValueError('Only current C original shared graph')
    floors={'eurosat':.512,'clevr':.631,'speech_binary':.472};names=sorted(floors)
    if any(trp['taskwise'][n]['scores']['all_development'][n]['accuracy']+1e-7<f for n,f in floors.items()):raise ValueError('Compatible assistants must individually pass the unchanged floor')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    with np.load(teacher_path,allow_pickle=False) as d:teachers={n:torch.tensor(d[n],dtype=torch.float64) for n in names}
    if any(v.shape!=(10,784) or not torch.isfinite(v).all() for v in teachers.values()):raise ValueError('Assistant Linear dimensions changed')
    train=np.flatnonzero(splits=='train');test=np.flatnonzero(splits=='test')
    if len(train)!=1600 or any(int((tasks[test]==n).sum())!=1000 for n in names):raise ValueError('Complete original C required')
    q,qt=TRAIN_queries(x[train],tasks[train],cfg);qy=y[train].repeat(1+cfg['TRAIN_augmentations']);transform,_=full_covariance_transform(x[train],contrast_transform(x[train]));z=q.double()@transform.T
    initial=model.shared_head.weight.detach().double()@torch.linalg.inv(transform);weight=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([weight],lr=PROFILE['lr'])
    groups={n:np.flatnonzero(qt==n) for n in names};targets={n:q[groups[n]].double()@teachers[n].T for n in names};baseline=evaluate(model.shared_head,x,y,tasks,test);best=whole_stage_selection(baseline,floors);best_weight=model.shared_head.weight.detach().clone();best_step=0;history=[]
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    put(a.out/'protocol.json',dict(profile=PROFILE,teacher_arrays_SHA256=sha(teacher_path),teacher_protocol_SHA256=sha(a.taskwise_run/'protocol.json'),parent_SHA256=a.parent_sha256,capture_features_SHA256=cr['features_sha256'],
        final_student_fit='Original TRAIN and8 weak augmentations only; per-task assistant queries on those inputs; no original EVAL or derivative EVAL input rows in student fit',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),same_shared_head=True,no_inference_task_gate=True,no_task_head_splicing=True,SDK=False,GPU=False,**DISCLOSURE))
    for step in range(1,PROFILE['steps']+1):
        gradients=[];losses={}
        for name in names:
            ix=groups[name];loss=convex_objective(z[ix]@weight.T,qy[ix],targets[name],torch.full((len(ix),),1/len(ix),dtype=torch.float64),cfg)+PROFILE['coordinate_delta_L2']*(weight-initial)[:,:-1].square().sum()/10
            gradients.append(torch.autograd.grad(loss,weight)[0]);losses[name]=float(loss.detach())
        gradient,conflicts=project_conflicts(gradients,step)
        if not torch.isfinite(gradient).all():raise RuntimeError('Nonfinite projected gradient')
        opt.zero_grad(set_to_none=True);weight.grad=gradient;opt.step()
        if step%20==0 or step==1 or time.monotonic()-started>=PROFILE['maximum_seconds']:
            with torch.no_grad():
                model.shared_head.weight.copy_((weight.detach()@transform).float());scores=evaluate(model.shared_head,x,y,tasks,test);value=whole_stage_selection(scores,floors)
                if value>best:best=value;best_step=step;best_weight=model.shared_head.weight.detach().clone()
            history.append(dict(step=step,losses=losses,conflicts=conflicts,scores=brief(scores),best_step=best_step));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='TRAIN_projected_gradient_distillation',step=step,losses=losses,updated=time.time()))
        if time.monotonic()-started>=PROFILE['maximum_seconds']:break
    reports={}
    for role,value in [('best',best_weight),('last',(weight.detach()@transform).float())]:
        with torch.no_grad():model.shared_head.weight.copy_(value)
        if upstream_state(model)!=frozen:raise RuntimeError('Frozen physical prefix changed')
        state=copy.deepcopy(cp);state.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,taskwise_assistants_SHA256=sha(teacher_path),**DISCLOSURE);path=a.out/(role+'_checkpoint.pt');torch.save(state,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=frozen or not torch.equal(value,loaded.shared_head.weight):raise RuntimeError('Strict CPU reload differs')
        reports[role]=evaluate(loaded.shared_head,x,y,tasks,test);put(a.out/(role+'_development.json'),dict(scores=reports[role],source_indices=ids[test].tolist(),tasks=tasks[test].tolist(),**DISCLOSURE))
    put(a.out/'report.json',dict(passed=True,baseline=brief(baseline),best=brief(reports['best']),last=brief(reports['last']),best_step=best_step,absolute_goal_passed=goal(reports['best']),provisional_not_final=not goal(reports['best']),
        strict_CPU_reloaded=True,frozen_upstream=True,original_student_parameters=7840,new_inference_layers=0,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),seconds=time.monotonic()-started,SDK=False,GPU=False,**DISCLOSURE));put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
