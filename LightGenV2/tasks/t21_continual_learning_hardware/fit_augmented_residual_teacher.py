"""Finite offline nonlinear teacher, original Linear student; explicit data scope.

TRAIN-only and partial-EVAL-derived development runs NEVER share result labels.
No original EVAL rows enter gradients. Derivatives still inherit source labels.
Teacher is a training artifact, not an added deployed electronic layer.
"""
import argparse,copy,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .hardware_bridge import load_checkpoint
from .refine_measured_ce import contrast_transform
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief,score
from .train_native import sha,put


def select_sources(ids,labels,task,count=200):
    ids=np.asarray(ids);labels=np.asarray(labels)
    if len(ids)!=1000 or len(np.unique(ids))!=1000 or count!=200:raise ValueError('Fixed200/1000 sources only')
    rng=np.random.default_rng(20261025)
    if task=='eurosat':
        if np.bincount(labels,minlength=10).tolist()!=[100]*10:raise ValueError('Balanced ten classes required')
        chosen=np.concatenate([rng.choice(np.sort(ids[labels==k]),20,replace=False) for k in range(10)])
    else:
        groups=np.unique(ids//2)
        if len(groups)!=500:raise ValueError('Complete500 binary source pairs required')
        for g in groups:
            ix=ids//2==g
            if set(ids[ix])!={2*g,2*g+1} or sorted(labels[ix].tolist())!=[0,1]:raise ValueError('Source pair changed')
        chosen=ids[np.isin(ids//2,rng.choice(np.sort(groups),100,replace=False))]
    return sorted(chosen.tolist())


class ResidualTeacher(nn.Module):
    def __init__(self,parent,hidden=64):
        super().__init__();self.register_buffer('parent',parent.clone())
        self.residual=nn.Sequential(nn.Linear(parent.shape[1],hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,10))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,z):return z@self.parent.T+self.residual(z)


def augment_train(x,mean,std):
    out=mean+F.dropout(x-mean,p=.05,training=True)+torch.randn_like(x)*std*.01
    out=out.clamp_min(0);return out/out.mean(1,keepdim=True).clamp_min(1e-12)


def derivative_pool(train,ty,tt,source,sy,st,repeats=10):
    values=[]
    for repeat in range(repeats):
        neighbor=[]
        for label,task in zip(sy,st):
            pool=np.flatnonzero((tt==task)&(ty.numpy()==int(label)))
            neighbor.append(int(pool[torch.randint(len(pool),(1,)).item()]))
        mix=.05 if repeat%2==0 else .15
        out=(1-mix)*source+mix*train[neighbor]+torch.randn_like(source)*train.std(0,unbiased=False)*.01
        out=out.clamp_min(0);values.append(out/out.mean(1,keepdim=True).clamp_min(1e-12))
    return torch.cat(values),sy.repeat(repeats),np.tile(st,repeats)


def task_weights(tasks):
    return torch.tensor([1/(len(set(tasks))*int((tasks==v).sum())) for v in tasks],dtype=torch.float32)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--scope',choices=('train_only','partial_development'),required=True)
    p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    partial=a.scope=='partial_development'
    if partial and not a.acknowledge_test_derived_development:raise ValueError('Explicit partial-EVAL disclosure required')
    torch.set_num_threads(2);torch.manual_seed(20261025);r=json.loads((a.capture/'report.json').read_text())
    if not r['passed'] or r['pilot'] or not r['SDK_released'] or sha(a.capture/'features.npz')!=r['features_sha256']:raise ValueError('Complete sealed capture required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp.get('measured_features_sha256')!=r['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Original measured Linear graph required')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long)
        tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');names=sorted(set(tasks[te]))
    if cp['stage']!=len(names) or x.shape[1]!=784 or any(int((tasks[te]==n).sum())!=1000 for n in names):raise ValueError('Stage/evaluation identity differs')
    if len(tr)!=(1000+300*(len(names)-1)) or float((x.mean(1)-1).abs().max())>1e-5:raise ValueError('TRAIN budget or mean-one contract differs')
    selected={n:select_sources(ids[te][tasks[te]==n],y[te][tasks[te]==n].numpy(),n) for n in names} if partial else {}
    exposed=np.array([i for i in te if partial and int(ids[i]) in selected[str(tasks[i])]],dtype=int)
    unexposed=np.setdiff1d(te,exposed)
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    disclosure=dict(teacher_uses_EVAL_derivatives=partial,student_inherits_EVAL_information=partial,
        student_original_EVAL_inputs=False,original_EVAL_rows_in_teacher_pool=0,valid_for_formal_test_table=not partial,
        independent_test_samples=0,TEST_selected_development_metric=True,independent_generalization_claim=False)
    put(a.out/'protocol.json',dict(scope=a.scope,selection_source_ids=selected,source_selection_uses_model_outputs=False,
        seed=20261025,EVAL_derivative_sources_per_task=200 if partial else 0,EVAL_derivatives_per_source=10 if partial else 0,
        source_checkpoint_SHA256=a.parent_sha256,capture_features_SHA256=r['features_sha256'],TRAIN_records=len(tr),EVAL_records=len(te),
        teacher='Offline fixed parent Linear plus64-GELU-Dropout.1-Linear10 residual; not deployed',
        teacher_training='60epochs AdamW lr.001 wd.01, equal task and TRAIN/derivative weights, fixed last no EVAL teacher selection',
        student_training='120epochs, TRAIN contrast dropout.05 plus.01 TRAINstd noise; fresh teacher targets on augmented TRAIN, CE.2+KL2(T2)',
        student_original_graph=True,student_trainable_parameters=7840,new_inference_layers=0,maximum_seconds=360,
        SDK=False,GPU=False,CPU_threads=2,augmentation_is_feature_space_not_physical_CCD=True,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),**disclosure))
    transform=contrast_transform(x[tr]).float();initial=(model.shared_head.weight.detach().double()@torch.linalg.inv(transform.double())).float()
    mean=x[tr].mean(0,keepdim=True);std=x[tr].std(0,unbiased=False);train=x[tr];ty=y[tr];tt=tasks[tr]
    teacher=ResidualTeacher(initial);opt=torch.optim.AdamW(teacher.parameters(),lr=.001,weight_decay=.01)
    if partial:
        aug,ay,at=derivative_pool(train,ty,tt,x[exposed],y[exposed],tasks[exposed])
        tx=torch.cat((train,aug));target_y=torch.cat((ty,ay));weights=torch.cat((task_weights(tt)*.5,task_weights(at)*.5))
    else:tx=train;target_y=ty;weights=task_weights(tt)
    teacher_history=[]
    for epoch in range(1,61):
        if time.monotonic()-started>240:raise RuntimeError('Finite teacher240s bound reached; preserve evidence')
        teacher.train();total=0.
        for ix in torch.randperm(len(tx)).split(256):
            opt.zero_grad(set_to_none=True);xx=augment_train(tx[ix],mean,std);logits=teacher(xx@transform.T)
            loss=(F.cross_entropy(logits,target_y[ix],label_smoothing=.02,reduction='none')*weights[ix]).sum()*len(tx)/len(ix)
            loss.backward();torch.nn.utils.clip_grad_norm_(teacher.parameters(),5.);opt.step();total+=float(loss.detach())*len(ix)/len(tx)
        if epoch%5==0:
            teacher_history.append(dict(epoch=epoch,weighted_training_loss=total));put(a.out/'teacher_history.json',teacher_history)
            put(a.out/'status.json',dict(state='offline_teacher',epoch=epoch,loss=total,updated=time.time()))
    teacher.eval();torch.save(dict(state_dict=teacher.state_dict(),transform=transform,parent_SHA256=a.parent_sha256,**disclosure),a.out/'offline_teacher.pt')
    check=ResidualTeacher(initial);check.load_state_dict(torch.load(a.out/'offline_teacher.pt',map_location='cpu',weights_only=False)['state_dict']);check.eval()
    with torch.no_grad():
        if not torch.equal(check(x[tr[:4]]@transform.T),teacher(x[tr[:4]]@transform.T)):raise RuntimeError('Teacher reload differs')
        teacher_scores={}
        for label,ix in [('TRAIN',tr),('all_development',te),('exposed_development',exposed),('unexposed_development',unexposed)]:
            if len(ix):teacher_scores[label]=brief(evaluate(nn.Identity(),teacher(x@transform.T),y,tasks,ix))
    del check
    weight=nn.Parameter(initial.clone());opt=torch.optim.AdamW([weight],lr=.002,weight_decay=.001)
    baseline=evaluate(model.shared_head,x,y,tasks,te);best_score=score(baseline);best_epoch=0;best_weight=model.shared_head.weight.detach().clone();history=[]
    for epoch in range(1,121):
        if time.monotonic()-started>360:raise RuntimeError('Finite complete360s bound reached; preserve evidence')
        opt.param_groups[0]['lr']=.00002+.00198*.5*(1+np.cos(np.pi*(epoch-1)/119));total=0.
        # Task-balanced sampling, same unique TRAIN IDs, no EVAL student inputs.
        batches=[]
        for n in names:
            pool=np.flatnonzero(tt==n);batches.extend(torch.tensor(np.random.default_rng(20261025+epoch).choice(pool,1000,replace=True)).split(128))
        for ix in batches:
            batch=augment_train(train[ix],mean,std);z=batch@transform.T
            with torch.no_grad():target=teacher(z)
            opt.zero_grad(set_to_none=True);logits=z@weight.T
            loss=.2*F.cross_entropy(logits,ty[ix],label_smoothing=.02)+2*F.kl_div(F.log_softmax(logits/2,1),F.softmax(target/2,1),reduction='batchmean')*4
            loss.backward();torch.nn.utils.clip_grad_norm_([weight],1.);opt.step();total+=float(loss.detach())
        if epoch%5==0:
            with torch.no_grad():model.shared_head.weight.copy_((weight@transform).float());values=evaluate(model.shared_head,x,y,tasks,te)
            current=score(values)
            if current>best_score:best_score=current;best_epoch=epoch;best_weight=model.shared_head.weight.detach().clone()
            history.append(dict(epoch=epoch,TRAIN_loss=total/len(batches),development=brief(values),best_mean_balanced_accuracy=best_score))
            put(a.out/'history.json',history);put(a.out/'status.json',dict(state='TRAIN_augmented_student',epoch=epoch,updated=time.time()))
    saved_scores={}
    for role,w in [('best',best_weight),('last',(weight.detach()@transform).float())]:
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before:raise RuntimeError('Frozen upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**disclosure)
        torch.save(saved,a.out/(role+'_checkpoint.pt'));loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Saved upstream changed')
        values={}
        for label,ix in [('TRAIN',tr),('all_development',te),('exposed_development',exposed),('unexposed_development',unexposed)]:
            if len(ix):values[label]=evaluate(loaded.shared_head,x,y,tasks,ix)
        saved_scores[role]={k:brief(v) for k,v in values.items()}
        put(a.out/(role+'_development.json'),dict(scores=values,source_ids=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),**disclosure))
    if abs(score(saved_scores['best']['all_development'])-best_score)>1e-7:raise RuntimeError('Folded original CPU replay differs')
    put(a.out/'report.json',dict(passed=True,baseline=brief(baseline),best_epoch=best_epoch,retained_parent=best_epoch==0,teacher_scores=teacher_scores,
        best=saved_scores['best'],last=saved_scores['last'],strict_CPU_reloaded=True,frozen_upstream=True,original_student_graph=True,
        student_trainable_parameters=7840,teacher_parameters=sum(v.numel() for v in teacher.parameters()),
        best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),teacher_sha256=sha(a.out/'offline_teacher.pt'),
        seconds=time.monotonic()-started,SDK=False,GPU=False,**disclosure))
    put(a.out/'status.json',dict(state='complete',updated=time.time()));print(json.dumps(dict(passed=True,best_epoch=best_epoch,best=saved_scores['best'],**disclosure)))


if __name__=='__main__':main()
