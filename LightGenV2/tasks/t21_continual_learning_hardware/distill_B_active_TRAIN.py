"""Bounded active TRAIN augmentation to compress a sealed offline teacher.

No original EVAL inputs/labels in student updates or synthetic query search.
Teacher inherits partial EVAL information: all scores remain development-only.
Student inference stays the original7840 Linear and frozen physical upstream.
"""
import argparse,copy,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .fit_B_balanced_residual_teacher import dkd_loss,whole_stage_selection
from .fit_augmented_residual_teacher import residual_teacher_from_artifact
from .fit_gated_linear_teacher import DISCLOSURE
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import put,sha,tensor_sha


def project_box_mean_one(values,base,std,radius=.02):
    """Box around original TRAIN, positivity and mean-one simultaneously."""
    lower=(base-radius*std).clamp_min(0);upper=base+radius*std
    if not torch.isfinite(values).all() or not torch.isfinite(base).all():raise ValueError('Nonfinite query')
    if bool((lower.mean(1)>1+1e-6).any() or (upper.mean(1)<1-1e-6).any()):raise ValueError('Infeasible mean-one TRAIN box')
    lo=(values-upper).min(1,keepdim=True).values;hi=(values-lower).max(1,keepdim=True).values
    for _ in range(28):
        mid=(lo+hi)/2;q=torch.maximum(lower,torch.minimum(upper,values-mid));above=q.mean(1,keepdim=True)>1
        lo=torch.where(above,mid,lo);hi=torch.where(above,hi,mid)
    return torch.maximum(lower,torch.minimum(upper,values-(lo+hi)/2))


def active_queries(base,labels,teacher,weight,transform,std,steps=4):
    """Maximize disagreement locally, retaining known TRAIN class under teacher."""
    with torch.no_grad():base_logits=teacher(base@transform.T);eligible=(base_logits.argmax(1)==labels)&(F.softmax(base_logits,1).max(1).values>=.6)
    q=base.detach().clone()
    for _ in range(steps):
        q=q.detach().requires_grad_(True);z=q@transform.T;t=teacher(z);s=z@weight.detach().T
        disagreement=F.kl_div(F.log_softmax(s/2,1),F.softmax(t/2,1),reduction='none').sum(1)*4
        confidence=F.log_softmax(t,1).gather(1,labels[:,None])[:,0]
        objective=((disagreement+.1*confidence)*eligible).sum()/eligible.sum().clamp_min(1)
        gradient=torch.autograd.grad(objective,q)[0]
        q=project_box_mean_one((q+.01*std*gradient.sign()).detach(),base,std)
    with torch.no_grad():keep=eligible&(teacher(q@transform.T).argmax(1)==labels)
    return torch.where(keep[:,None],q.detach(),base)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','teacher_run','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Disclosed inherited EVAL information required')
    torch.set_num_threads(2);torch.manual_seed(20261025);capture=json.loads((a.capture/'report.json').read_text());report=json.loads((a.teacher_run/'report.json').read_text());protocol=json.loads((a.teacher_run/'protocol.json').read_text())
    if (not capture['passed'] or capture['pilot'] or not capture['SDK_released'] or sha(a.capture/'features.npz')!=capture['features_sha256'] or
        protocol['capture_features_SHA256']!=capture['features_sha256'] or protocol['parent_SHA256']!=a.parent_sha256 or not report['teacher_quality_gate_passed'] or report['valid_for_formal_test_table'] is not False):raise ValueError('Exact complete capture, parent and admitted teacher required')
    teacher_path=a.teacher_run/'offline_teacher.pt'
    if sha(teacher_path)!=report['teacher_sha256']:raise ValueError('Teacher changed')
    artifact=torch.load(teacher_path,map_location='cpu',weights_only=False);teacher=residual_teacher_from_artifact(artifact);transform=artifact['transform'];teacher_before={n:tensor_sha(v) for n,v in teacher.state_dict().items()}
    if artifact['parent_SHA256']!=a.parent_sha256:raise ValueError('Teacher-parent differs')
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model);parent_head=copy.deepcopy(model.shared_head).eval()
    if cp['architecture']!='d2nn' or cp['stage']!=2 or cp.get('measured_features_sha256')!=capture['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Original measured D2NN B required')
    for parameter in parent_head.parameters():parameter.requires_grad_(False)
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');old=tr[tasks[tr]=='eurosat'];cur=tr[tasks[tr]=='clevr']
    if (len(old),len(cur),len(te))!=(300,1000,2000):raise ValueError('Fixed original TRAIN/EVAL identity changed')
    if any(int((tasks[te]==n).sum())!=1000 for n in ('eurosat','clevr')):raise ValueError('Incomplete evaluation')
    exposed=np.array([i for i in te if int(ids[i]) in protocol['selection_source_ids'][str(tasks[i])]],dtype=int);unexposed=np.setdiff1d(te,exposed);std=x[tr].std(0,unbiased=False)
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();floors={'eurosat':.483,'clevr':.710}
    put(a.out/'protocol.json',dict(scope='partial_development_inherited_teacher',parent_SHA256=a.parent_sha256,teacher_SHA256=report['teacher_sha256'],capture_features_SHA256=capture['features_sha256'],
        student_input_source='Original TRAIN and active derivatives of those same TRAIN IDs ONLY',student_source_ids={n:ids[tr][tasks[tr]==n].tolist() for n in ('eurosat','clevr')},teacher_selection_source_ids=protocol['selection_source_ids'],
        query_objective='Max teacher-student KL plus.1 known TRAIN-class log probability; teacher and student weights fixed during query search',query_steps=4,query_radius=.02,query_step=.01,query_scale='Original TRAIN feature standard deviation',
        query_constraints='Elementwise TRAIN box, positive mean-one, teacher retains known TRAIN class; otherwise original TRAIN fallback',
        loss='CE.5+DKD(alpha1,beta4,T2)+old TRAIN parentKL1(T2)',epochs=120,steps_per_epoch=12,maximum_seconds=240,CPU_threads=2,SDK=False,GPU=False,
        original_student_parameters=7840,new_inference_layers=0,selection='Worst full-task deficit then mean, E0 retained, no per-cell PT splice or score clipping',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),**DISCLOSURE))
    weight=nn.Parameter((model.shared_head.weight.detach().double()@torch.linalg.inv(transform.double())).float());optimizer=torch.optim.AdamW([weight],lr=.001,weight_decay=.0001)
    baseline=evaluate(model.shared_head,x,y,tasks,te);best_selection=whole_stage_selection(baseline,floors);best_epoch=0;best_weight=model.shared_head.weight.detach().clone();history=[]
    for epoch in range(1,121):
        optimizer.param_groups[0]['lr']=.00001+.00099*.5*(1+np.cos(np.pi*(epoch-1)/119));total=0.
        for _ in range(12):
            oi=old[torch.randint(len(old),(64,))];ci=cur[torch.randint(len(cur),(64,))];indices=torch.cat((torch.as_tensor(oi),torch.as_tensor(ci)));base=x[indices];labels=y[indices]
            q=active_queries(base,labels,teacher,weight,transform,std);values=torch.cat((base,q));targets_y=labels.repeat(2);z=values@transform.T
            with torch.no_grad():target=teacher(z);old_target=parent_head(x[oi])
            logits=z@weight.T;loss=.5*F.cross_entropy(logits,targets_y,label_smoothing=.02)+dkd_loss(logits,target,targets_y)
            loss+=4*F.kl_div(F.log_softmax(logits[:64]/2,1),F.softmax(old_target/2,1),reduction='batchmean')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite active TRAIN loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([weight],1.);optimizer.step();total+=float(loss.detach())
        if epoch%5==0 or time.monotonic()-started>=240:
            with torch.no_grad():model.shared_head.weight.copy_((weight@transform).float());scores=evaluate(model.shared_head,x,y,tasks,te)
            selection=whole_stage_selection(scores,floors)
            if selection>best_selection:best_selection=selection;best_epoch=epoch;best_weight=model.shared_head.weight.detach().clone()
            history.append(dict(epoch=epoch,loss=total/12,scores=brief(scores),best_epoch=best_epoch));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='student_active_TRAIN',**history[-1],updated=time.time()))
        if time.monotonic()-started>=240:break
    result={}
    for role,w in [('best',best_weight),('last',(weight.detach()@transform).float())]:
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before or {n:tensor_sha(v) for n,v in teacher.state_dict().items()}!=teacher_before:raise RuntimeError('Frozen upstream or teacher changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**DISCLOSURE);path=a.out/(role+'_checkpoint.pt');torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,w):raise RuntimeError('Saved original graph differs')
        with torch.no_grad():scores={n:evaluate(loaded.shared_head,x,y,tasks,ix) for n,ix in [('TRAIN',tr),('all_development',te),('exposed_development',exposed),('unexposed_development',unexposed)]}
        result[role]={n:brief(v) for n,v in scores.items()};put(a.out/(role+'_development.json'),dict(scores=scores,**DISCLOSURE))
    put(a.out/'report.json',dict(passed=True,baseline=brief(baseline),best=result['best'],last=result['last'],best_epoch=best_epoch,retained_parent=best_epoch==0,strict_CPU_reloaded=True,frozen_upstream=True,frozen_teacher=True,
        original_student_parameters=7840,new_inference_layers=0,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),teacher_sha256=report['teacher_sha256'],seconds=time.monotonic()-started,SDK=False,GPU=False,**DISCLOSURE))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
