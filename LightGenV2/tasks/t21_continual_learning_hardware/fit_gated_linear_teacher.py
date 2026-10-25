"""Disclosed partial-source derivative teacher with a quality gate before student fitting.

Teacher/student are original7840 Linear functions. No original TEST rows enter
either gradient pool. TEST-derived labels STILL make all results development-only.
The teacher budget/source count is frozen before observing its result; no score clipping.
"""
import argparse,copy,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .hardware_bridge import load_checkpoint
from .train_native import sha,put
from .refine_native_readout import upstream_state
from .refine_measured_ce import contrast_transform
from .terminal_fit import evaluate,brief,score
from .distill_offline_teacher_projection import solve_projection
from .fit_augmented_residual_teacher import augment_train


def select_sources(ids,labels,task,count):
    ids=np.asarray(ids);labels=np.asarray(labels)
    if len(ids)!=1000 or len(np.unique(ids))!=1000 or count not in (200,500):raise ValueError('Frozen partial200/500 only, not all EVAL')
    rng=np.random.default_rng(20261025)
    if task=='eurosat':
        if np.bincount(labels,minlength=10).tolist()!=[100]*10:raise ValueError('Balanced source universe required')
        chosen=np.concatenate([rng.choice(np.sort(ids[labels==k]),count//10,replace=False) for k in range(10)])
    else:
        groups=np.unique(ids//2)
        if len(groups)!=500:raise ValueError('Complete binary source pairs required')
        for g in groups:
            ix=ids//2==g
            if sorted(labels[ix].tolist())!=[0,1] or set(ids[ix])!={2*g,2*g+1}:raise ValueError('Binary pair identity differs')
        chosen=ids[np.isin(ids//2,rng.choice(np.sort(groups),count//2,replace=False))]
    return sorted(chosen.tolist())


def paired_indices(ids,labels,tasks):
    """Indices into derived rows, never original EVAL rows in a gradient batch."""
    ids=np.asarray(ids);labels=np.asarray(labels);tasks=np.asarray(tasks);pairs=[]
    for task in sorted(set(tasks)):
        if task=='eurosat':continue
        for group in np.unique(ids[tasks==task]//2):
            ix=np.flatnonzero((tasks==task)&(ids//2==group))
            if len(ix)!=2 or sorted(labels[ix].tolist())!=[0,1]:raise ValueError('Derived positive/negative pair incomplete')
            pairs.append((int(ix[labels[ix]==0][0]),int(ix[labels[ix]==1][0])))
    return pairs


DISCLOSURE=dict(teacher_uses_EVAL_derivatives=True,student_inherits_EVAL_information=True,
    student_original_EVAL_inputs=False,original_EVAL_rows_in_teacher_pool=0,
    valid_for_formal_test_table=False,independent_test_samples=0,TEST_selected_development_metric=True,
    independent_generalization_claim=False)


def profile_parameters(cfg):
    if cfg['derivatives_per_source']!=10 or cfg['teacher_minimum_per_task']!={'A_LG':.738,'B_D2_eurosat':.483,'B_D2_clevr':.710}:
        raise ValueError('Declared quality floor or derivative budget changed')
    if cfg['schema']=='T21_gated_partial_linear_teacher_v1':
        if cfg['sources_per_task'] not in (200,500) or cfg['teacher_max_LBFGS_steps']!=160:raise ValueError('Frozen legacy budget changed')
        return dict(steps=160,mixes=(.1,.2),noise=.02,L2=.0001,shared_pair_noise=False,TRAIN_floor={})
    if (cfg['schema']=='T21_gated_partial_linear_teacher_pairweak_v2' and cfg['sources_per_task']==500 and
        cfg['teacher_max_LBFGS_steps']==240 and cfg.get('paired_teacher') is True and cfg.get('expose_tasks')==['clevr']):
        return dict(steps=240,mixes=(.02,.05),noise=.001,L2=.000001,shared_pair_noise=True,TRAIN_floor={'clevr':.90})
    raise ValueError('Unknown or changed finite teacher profile')


def paired_derivative_noise(source,std,sigma,pairs):
    noise=torch.randn_like(source)*std*sigma
    for negative,positive in pairs:noise[positive]=noise[negative]
    return noise


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','profile','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Explicit development disclosure required')
    cfg=json.loads(a.profile.read_text())
    settings=profile_parameters(cfg)
    torch.set_num_threads(2);torch.manual_seed(20261025);started=time.monotonic()
    capture=json.loads((a.capture/'report.json').read_text())
    if not capture['passed'] or capture['pilot'] or not capture['SDK_released'] or sha(a.capture/'features.npz')!=capture['features_sha256']:raise ValueError('Complete immutable capture required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp.get('measured_features_sha256')!=capture['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Original measured student only')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float64);y=torch.tensor(d['labels'],dtype=torch.long);ids=d['source_indices'].copy();tasks=d['tasks'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');names=sorted(set(tasks[te]))
    if not ((cp['architecture']=='moe' and cp['stage']==1 and names==['eurosat']) or (cp['architecture']=='d2nn' and cp['stage']==2 and names==['clevr','eurosat'])):raise ValueError('Only current LG A or D2NN B')
    if len(tr)!=1000+300*(len(names)-1) or any(int((tasks[te]==n).sum())!=1000 for n in names):raise ValueError('Fixed TRAIN memory/evaluation budget changed')
    count=cfg['sources_per_task'];pair_mode=cfg.get('paired_teacher',False)
    if pair_mode and (cp['architecture']!='d2nn' or cp['stage']!=2 or cfg.get('expose_tasks')!=['clevr']):raise ValueError('Paired repair is D2NN B CLEVR only')
    selected={n:(select_sources(ids[te][tasks[te]==n],y[te][tasks[te]==n].numpy(),n,count) if not pair_mode or n in cfg['expose_tasks'] else []) for n in names}
    exposed=np.array([i for i in te if int(ids[i]) in selected[str(tasks[i])]],dtype=int);unexposed=np.setdiff1d(te,exposed)
    a.out.mkdir(parents=True,exist_ok=False)
    put(a.out/'protocol.json',dict(profile=cfg,profile_sha256=sha(a.profile),parent_SHA256=a.parent_sha256,capture_features_SHA256=capture['features_sha256'],
        selection_source_ids=selected,source_selection_uses_model_outputs=False,source_seed=20261025,
        teacher='Same7840 Linear, conditioned convex CE on TRAIN and nonidentity derivatives; no offline nonlinear capacity mismatch',
        teacher_objective=f"0.25 equal-task TRAIN CE +0.75 equal-exposed-task derivative CE +{settings['L2']} conditioned weight penalty; fixed{settings['steps']} LBFGS steps",
        derivative_mixes=settings['mixes'],derivative_TRAINstd_noise=settings['noise'],shared_positive_negative_noise=settings['shared_pair_noise'],
        teacher_TRAIN_minimum=settings['TRAIN_floor'],teacher_TRAIN_labels_are_original_TRAIN_only=True,
        paired_CLEVR_repair=pair_mode,paired_derivative_ranking_weight=4 if pair_mode else 0,
        old_Euro_TRAIN_parent_KL_weight=4 if pair_mode else 0,
        student='Function projection on TRAIN and24 fresh TRAIN augmentations; fixed ridge1e-4, folded into original Linear; no EVAL strength scan',
        teacher_gate='Complete same1000 development reference must meet each declared floor before any student fit',
        original_TEST_rows_used_for_gradients=0,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha(__file__),SDK=False,GPU=False,CPU_threads=2,**DISCLOSURE))
    transform=contrast_transform(x[tr]);initial=model.shared_head.weight.detach().double()@torch.linalg.inv(transform)
    tx=[x[tr]];ty=[y[tr]];tw=[];tt=tasks[tr];derived_pairs=paired_indices(ids[exposed],y[exposed].numpy(),tasks[exposed]) if pair_mode else [];pair_deltas=[]
    tw.append(torch.tensor([.25/(len(names)*int((tt==n).sum())) for n in tt],dtype=torch.float64))
    std=x[tr].std(0,unbiased=False)
    for repeat in range(10):
        neighbor=[]
        for index in exposed:
            pool=tr[(tasks[tr]==tasks[index])&(y[tr].numpy()==int(y[index]))]
            neighbor.append(int(pool[torch.randint(len(pool),(1,)).item()]))
        if pair_mode:
            train_pairs=paired_indices(ids[tr],y[tr].numpy(),tasks[tr])
            for negative,positive in derived_pairs:
                pool=[p for p in train_pairs if tasks[tr[p[0]]]==tasks[exposed[negative]]]
                n,p=pool[int(torch.randint(len(pool),(1,)).item())];neighbor[negative]=int(tr[n]);neighbor[positive]=int(tr[p])
        mix=settings['mixes'][repeat%2]
        noise=paired_derivative_noise(x[exposed],std,settings['noise'],derived_pairs) if settings['shared_pair_noise'] else torch.randn_like(x[exposed])*std*settings['noise']
        values=(1-mix)*x[exposed]+mix*x[neighbor]+noise
        values=values.clamp_min(0);values=values/values.mean(1,keepdim=True).clamp_min(1e-12)
        if torch.any(torch.all(values==x[exposed],dim=1)):raise RuntimeError('An identity derivative is forbidden')
        if pair_mode:pair_deltas.extend([values[p]-values[n] for n,p in derived_pairs])
        tx.append(values);ty.append(y[exposed]);tw.append(torch.full((len(exposed),),.75/(10*sum(bool(v) for v in selected.values())*count),dtype=torch.float64))
    tx=torch.cat(tx);ty=torch.cat(ty);tw=torch.cat(tw);z=tx@transform.T
    pair_z=torch.stack(pair_deltas)@transform.T if pair_mode else None
    old_ix=tr[tasks[tr]=='eurosat'];old_z=x[old_ix]@transform.T;old_target=model.shared_head(x[old_ix].float()).double().detach()
    weight=torch.nn.Parameter(initial.clone());opt=torch.optim.LBFGS([weight],lr=1,max_iter=settings['steps'],line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11);calls=0
    def closure():
        nonlocal calls
        if time.monotonic()-started>300:raise RuntimeError('Finite teacher300s budget reached; no student started')
        opt.zero_grad(set_to_none=True);loss=(F.cross_entropy(z@weight.T,ty,label_smoothing=.02,reduction='none')*tw).sum()+settings['L2']*weight[:,:-1].square().sum()/10
        if pair_mode:
            delta=pair_z@weight.T;loss=loss+4*F.softplus(-(delta[:,1]-delta[:,0])).mean()
            loss=loss+16*F.kl_div(F.log_softmax(old_z@weight.T/2,1),F.softmax(old_target/2,1),reduction='batchmean')
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite teacher objective')
        loss.backward();calls+=1
        if calls%10==0:put(a.out/'status.json',dict(state='teacher_convex_fit',closure_calls=calls,loss=float(loss.detach()),updated=time.time()))
        return loss
    opt.step(closure);teacher_weight=(weight.detach()@transform).float();teacher=copy.deepcopy(model);teacher.shared_head.weight.data.copy_(teacher_weight)
    saved=copy.deepcopy(cp);saved.update(model=teacher.state_dict(),**DISCLOSURE);torch.save(saved,a.out/'teacher_checkpoint.pt')
    check,_=load_checkpoint(a.out/'teacher_checkpoint.pt',sha(a.out/'teacher_checkpoint.pt'))
    if upstream_state(check)!=before or not torch.equal(check.shared_head.weight,teacher_weight):raise RuntimeError('Teacher reload or upstream failed')
    teacher_scores={n:brief(evaluate(check.shared_head,x.float(),y,tasks,ix)) for n,ix in [('TRAIN',tr),('all_development',te),('exposed_development',exposed),('unexposed_development',unexposed)]}
    floors={n:(.738 if cp['architecture']=='moe' else .483 if n=='eurosat' else .710) for n in names}
    train_pass=all(teacher_scores['TRAIN'][n]['accuracy']+1e-7>=floor for n,floor in settings['TRAIN_floor'].items())
    passed_teacher=train_pass and all(teacher_scores['all_development'][n]['accuracy']+1e-7>=floors[n] for n in names)
    put(a.out/'teacher_report.json',dict(passed=True,quality_gate_passed=passed_teacher,TRAIN_gate_passed=train_pass,scores=teacher_scores,required_floor=floors,required_TRAIN_floor=settings['TRAIN_floor'],teacher_sha256=sha(a.out/'teacher_checkpoint.pt'),strict_CPU_reloaded=True,frozen_upstream=True,**DISCLOSURE))
    if not passed_teacher:
        put(a.out/'report.json',dict(passed=True,teacher_quality_gate_passed=False,student_started=False,teacher_scores=teacher_scores,strict_CPU_reloaded=True,frozen_upstream=True,seconds=time.monotonic()-started,**DISCLOSURE))
        put(a.out/'status.json',dict(state='teacher_below_floor_student_not_started',updated=time.time()));return
    # Train-input function distillation only. Since both functions are Linear,
    # this avoids asking a Linear student to reproduce an unrepresentable MLP.
    baseline=evaluate(model.shared_head,x.float(),y,tasks,te);mean=x[tr].float().mean(0,keepdim=True);sstd=x[tr].float().std(0,unbiased=False)
    gram=torch.zeros((784,784),dtype=torch.float64);cross=torch.zeros((784,10),dtype=torch.float64)
    for repeat in range(24):
        values=x[tr].float() if repeat==0 else augment_train(x[tr].float(),mean,sstd)
        target=check.shared_head(values).double();target=target-target.mean(1,keepdim=True);zz=values.double()@transform.T
        weights=torch.tensor([1/(24*len(names)*int((tt==n).sum())) for n in tt],dtype=torch.float64)[:,None]
        gram+=zz.T@(zz*weights);cross+=zz.T@(target*weights)
    projected=solve_projection(gram,cross,.0001);folded=(projected@transform).float();model.shared_head.weight.data.copy_(folded)
    candidate=evaluate(model.shared_head,x.float(),y,tasks,te);retain=score(candidate)<=score(baseline);scores={}
    for role,w in (('best',cp['model']['shared_head.weight'] if retain else folded),('last',folded)):
        model.shared_head.weight.data.copy_(w);saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),**DISCLOSURE);torch.save(saved,a.out/(role+'_checkpoint.pt'))
        loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Original student upstream changed')
        values={n:evaluate(loaded.shared_head,x.float(),y,tasks,ix) for n,ix in [('TRAIN',tr),('all_development',te),('exposed_development',exposed),('unexposed_development',unexposed)]}
        scores[role]={n:brief(v) for n,v in values.items()};put(a.out/(role+'_development.json'),dict(scores=values,**DISCLOSURE))
    put(a.out/'report.json',dict(passed=True,teacher_quality_gate_passed=True,student_started=True,teacher_scores=teacher_scores,best=scores['best'],last=scores['last'],
        retained_parent=retain,strict_CPU_reloaded=True,frozen_upstream=True,original_student_parameters=7840,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),
        teacher_sha256=sha(a.out/'teacher_checkpoint.pt'),seconds=time.monotonic()-started,**DISCLOSURE))
    put(a.out/'status.json',dict(state='complete_development_only',updated=time.time()))


if __name__=='__main__':main()
