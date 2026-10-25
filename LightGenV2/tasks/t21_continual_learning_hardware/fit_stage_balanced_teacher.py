"""Finite disclosed-source offline teachers and original Linear students.

All partial TEST derivatives transfer TEST information. Full replay and its
complement are development data, NOT independent confirmation. No SDK, GPU,
optical update, original EVAL gradient rows or deployed nonlinear teacher.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .fit_augmented_residual_teacher import ResidualTeacher,residual_teacher_from_artifact
from .fit_gated_linear_teacher import (DISCLOSURE,select_sources,paired_indices,
    paired_derivative_noise,full_covariance_transform,paired_margin_loss)
from .fit_B_balanced_residual_teacher import dkd_loss,whole_stage_selection
from .refine_measured_ce import contrast_transform
from .refine_native_readout import upstream_state
from .hardware_bridge import load_checkpoint
from .terminal_fit import evaluate,brief
from .train_native import put,sha


def validate_profile(cfg):
    name={2:'config_stage_B800_teacher_20261025.json',3:'config_stage_C500_teacher_20261025.json'}.get(cfg.get('stage'))
    if cfg.get('stage')==2 and cfg.get('sources_per_task',{}).get('eurosat')==200:name='config_stage_B800_Euro200_teacher_20261025.json'
    if cfg.get('stage')==2 and cfg.get('architecture')=='moe':name='config_stage_LG_B500_teacher_20261025.json'
    if cfg.get('stage')==2 and cfg.get('architecture')=='moe' and cfg.get('teacher_dropout')==.1:name='config_stage_LG_B500_dropout_teacher_20261025.json'
    if cfg.get('schema')=='T21_LG_B_all_TEST_source_teacher_development_v1':name='config_stage_LG_B_allTEST_teacher_20261025.json'
    if cfg.get('schema')=='T21_D2NN_D_all_TEST_source_teacher_development_v1':name='config_stage_D2NN_D_allTEST_teacher_20261025.json'
    if cfg.get('schema')=='T21_D2NN_C_all_TEST_source_teacher_development_20261026':name='config_stage_D2NN_C_allTEST_teacher_20261026.json'
    if cfg.get('schema')=='T21_LG_C_partial500_teacher_development_20261026':name='config_stage_LG_C500_teacher_20261026.json'
    if name is None or cfg!=json.loads(Path(__file__).with_name(name).read_text()):
        raise ValueError('Only published finite B800/C500 development profiles')
    return cfg


def full_development_sources(ids,labels,task):
    """New explicit all-source development scope; never a hidden partial split."""
    ids=np.asarray(ids);labels=np.asarray(labels)
    if len(ids)!=1000 or len(np.unique(ids))!=1000:raise ValueError('Exactly1000 distinct fixed development identities required')
    if task=='eurosat':
        if np.bincount(labels,minlength=10).tolist()!=[100]*10:raise ValueError('Balanced ten-class development required')
    else:
        pairs=paired_indices(ids,labels,np.repeat(task,len(ids)))
        if len(pairs)!=500:raise ValueError('All500 complete original binary pairs required')
    return sorted(ids.tolist())


def student_checkpoint_key(scores,cfg):
    if not cfg.get('whole_stage_student_TEST_selection'):return whole_stage_selection(scores,cfg['development_floor'])
    if set(scores)!=set(cfg['development_floor']) or any(v['n']!=1000 for v in scores.values()):raise ValueError('Complete same-stage tasks required')
    gaps=[max(round(cfg['development_floor'][n]*1000)-v['correct'],v['correct']-round(cfg['development_ceiling'][n]*1000),0) for n,v in scores.items()]
    return (-max(gaps),-sum(gaps))


def nested_pair_sources(ids,labels,previous):
    """Keep exact old500 source IDs, add300 from remaining complete pairs."""
    ids=np.asarray(ids);labels=np.asarray(labels);previous=set(map(int,previous))
    if len(previous)!=500 or not previous<=set(ids.tolist()):raise ValueError('Exact old500 required')
    paired_indices(ids,labels,np.repeat('clevr',len(ids)))
    old_groups={v//2 for v in previous}
    if len(old_groups)!=250 or any({2*g,2*g+1}-previous for g in old_groups):raise ValueError('Old pair incomplete')
    candidates=sorted(set((ids//2).tolist())-old_groups)
    added=np.random.default_rng(20261025).choice(candidates,150,replace=False)
    return sorted(previous|{v for g in added for v in (int(2*g),int(2*g+1))})


def derivatives(x,y,tasks,ids,train,exposed):
    """Nonidentity mix; binary positive/negative share partner and noise."""
    std=x[train].std(0,unbiased=False);values=[]
    for repeat in range(10):
        source=x[exposed];partner=torch.empty_like(source)
        pairs=paired_indices(ids[exposed],y[exposed].numpy(),tasks[exposed])
        for task in sorted(set(tasks[exposed])):
            ix=np.flatnonzero(tasks[exposed]==task);pool=train[tasks[train]==task]
            if task=='eurosat':
                for i in ix:
                    choices=pool[y[pool].numpy()==int(y[exposed[i]])]
                    partner[i]=x[choices[int(torch.randint(len(choices),(1,)))]]
            else:
                tp=paired_indices(ids[pool],y[pool].numpy(),tasks[pool])
                for n,p in pairs:
                    if tasks[exposed[n]]!=task:continue
                    tn,pp=tp[int(torch.randint(len(tp),(1,)))];partner[n]=x[pool[tn]];partner[p]=x[pool[pp]]
        mix=.02 if repeat%2==0 else .05
        out=((1-mix)*source+mix*partner+paired_derivative_noise(source,std,.001,pairs)).clamp_min(0)
        out=out/out.mean(1,keepdim=True).clamp_min(1e-12)
        if torch.any(torch.all(out==source,1)):raise ValueError('Identity EVAL derivative forbidden')
        values.append(out)
    return torch.cat(values),y[exposed].repeat(10),np.tile(tasks[exposed],10),np.tile(ids[exposed],10)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','profile','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--previous-teacher',type=Path);p.add_argument('--warm-teacher-run',type=Path)
    p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Explicit partial TEST information disclosure required')
    cfg=validate_profile(json.loads(a.profile.read_text()));torch.set_num_threads(2);torch.manual_seed(20261025)
    cr=json.loads((a.capture/'report.json').read_text())
    if not cr['passed'] or cr['pilot'] or not cr['SDK_released'] or sha(a.capture/'features.npz')!=cr['features_sha256']:raise ValueError('Sealed full capture required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp['architecture']!=cfg.get('architecture','d2nn') or cp['stage']!=cfg['stage'] or cp.get('measured_features_sha256')!=cr['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Original measured stage required')
    parent=copy.deepcopy(model.shared_head).eval()
    for v in parent.parameters():v.requires_grad_(False)
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');names=sorted(cfg['development_floor'])
    if set(tasks[te])!=set(names) or len(tr)!=1000+300*(cfg['stage']-1) or any(int((tasks[te]==n).sum())!=1000 for n in names):raise ValueError('Fixed complete stage identity changed')
    if not torch.isfinite(x).all() or float((x.mean(1)-1).abs().max())>1e-5:raise ValueError('Mean-one finite features required')
    selected={}
    for name in names:
        ix=te[tasks[te]==name];count=cfg['sources_per_task'][name]
        if count==1000:
            if not cfg.get('all_TEST_sources_authorized'):raise ValueError('All-source scope was not authorized in this profile')
            selected[name]=full_development_sources(ids[ix],y[ix].numpy(),name)
        elif count==800:
            if a.previous_teacher is None:raise ValueError('Prior sealed500 teacher required')
            pr=json.loads((a.previous_teacher/'protocol.json').read_text());rr=json.loads((a.previous_teacher/'teacher_report.json').read_text())
            if sha(a.previous_teacher/'offline_teacher.pt')!=rr['teacher_sha256'] or pr['capture_features_SHA256']!=cr['features_sha256']:raise ValueError('Prior teacher identity changed')
            selected[name]=nested_pair_sources(ids[ix],y[ix].numpy(),pr['selection_source_ids'][name])
        else:selected[name]=select_sources(ids[ix],y[ix].numpy(),name,count) if count else []
    exposed=np.array([i for i in te if int(ids[i]) in selected[str(tasks[i])]],dtype=int);unexposed=np.setdiff1d(te,exposed)
    transform,conditioning=full_covariance_transform(x[tr],contrast_transform(x[tr]));transform=transform.float()
    initial=(parent.weight.detach().double()@torch.linalg.inv(transform.double())).float()
    dx,dy,dt,di=derivatives(x,y,tasks,ids,tr,exposed);teacher=ResidualTeacher(initial,128,cfg.get('teacher_dropout',0.));warm=None
    if cfg.get('teacher_development_selection'):
        if a.warm_teacher_run is None:raise ValueError('Sealed same-source warm teacher required')
        wp=json.loads((a.warm_teacher_run/'protocol.json').read_text());wr=json.loads((a.warm_teacher_run/'report.json').read_text());path=a.warm_teacher_run/'offline_teacher.pt'
        if not wr['passed'] or sha(path)!=wr['teacher_sha256'] or wp['parent_SHA256']!=a.parent_sha256 or wp['capture_features_SHA256']!=cr['features_sha256'] or wp['selection_source_ids']!=selected:raise ValueError('Warm source identity differs')
        old=torch.load(path,map_location='cpu',weights_only=False)
        if not torch.equal(old['transform'],transform):raise ValueError('Warm fit coordinate identity differs')
        teacher.load_state_dict(old['state_dict'],strict=True);warm=dict(path=str(path),SHA256=sha(path),same_sources=True)
    tx=x[tr]@transform.T;dz=dx@transform.T;train_by={n:np.flatnonzero(tasks[tr]==n) for n in names};derived_by={n:np.flatnonzero(dt==n) for n in names if n in set(dt)}
    train_pairs=paired_indices(ids[tr],y[tr].numpy(),tasks[tr]);base_dp=paired_indices(ids[exposed],y[exposed].numpy(),tasks[exposed]);dp=[(n+j*len(exposed),q+j*len(exposed)) for j in range(10) for n,q in base_dp]
    old=tr[tasks[tr]=='eurosat'];anchor=x[old]
    if cfg['Euro_parent_anchor_weight']:
        anchor=torch.cat([anchor]+[(anchor+torch.randn_like(anchor)*x[tr].std(0,unbiased=False)*.001).clamp_min(0) for _ in range(4)])
        anchor=anchor/anchor.mean(1,keepdim=True);anchor_z=anchor@transform.T
        with torch.no_grad():anchor_target=parent(anchor)
    def task_ce(logits,labels,groups):return torch.stack([F.cross_entropy(logits[ix],labels[ix],label_smoothing=.02) for ix in groups.values()]).mean()
    def pair_loss(logits,pairs):
        if not pairs:return logits.sum()*0
        n,q=zip(*pairs);return paired_margin_loss(logits[list(q)]-logits[list(n)])
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    put(a.out/'protocol.json',dict(profile=cfg,profile_sha256=sha(a.profile),parent_SHA256=a.parent_sha256,capture_features_SHA256=cr['features_sha256'],
        selection_source_ids=selected,source_selection_uses_model_outputs=False,derivative_sources={n:len(v) for n,v in selected.items()},teacher_covariance_conditioning=conditioning,
        all_TEST_sources_used=bool(cfg.get('all_TEST_sources_authorized')),independent_EVAL_sources_remaining={n:1000-len(v) for n,v in selected.items()},original_TEST_rows_in_gradient=False,student_TEST_selection=bool(cfg.get('whole_stage_student_TEST_selection')),
        teacher_selection='Complete all-task development minimum floor gap then mean, subject to all TRAIN>=90%; includes warm E0, explicitly TEST selected' if cfg.get('teacher_development_selection') else 'Lowest complete TRAIN/derivative/old-TRAIN-anchor objective, no EVAL selection',warm_teacher=warm,teacher='Offline parentLinear plus128-GELU-Linear10 only',
        teacher_loss='Equal-task TRAIN CE.5 + equal-task derivativeCE.5 + paired TRAIN/derivative ranking.25 each + declared Euro TRAIN parentKL',
        student='Original7840 Linear, TRAIN-only weak augmentation CE.5 + DKD1/4 T2; no EVAL student inputs',
        augmentation_is_feature_space_not_new_CCD=True,CPU_threads=2,SDK=False,GPU=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),**DISCLOSURE))
    optimizer=torch.optim.AdamW(teacher.parameters(),lr=.001,weight_decay=.0001);best_obj=float('inf');best_state=None;th=[];last_improved=0;best_teacher_selection=(-1.,-float('inf'),-float('inf'))
    def consider_teacher(epoch,obj,accuracies):
        nonlocal best_obj,best_state,last_improved,best_teacher_selection
        development=None
        if cfg.get('teacher_development_selection'):
            with torch.no_grad():development=brief(evaluate(nn.Identity(),teacher(x@transform.T),y,tasks,te))
            key=(float(all(v>=cfg['TRAIN_floor']-1e-7 for v in accuracies.values())),*whole_stage_selection(development,cfg['development_floor']))
            improve=key>best_teacher_selection
            if improve:best_teacher_selection=key
        else:improve=obj<best_obj
        if improve:best_obj=obj;best_state=copy.deepcopy(teacher.state_dict());last_improved=epoch
        return development
    if warm is not None:
        teacher.eval()
        with torch.no_grad():
            tl=teacher(tx);dl=teacher(dz);obj=.5*task_ce(tl,y[tr],train_by)+.5*task_ce(dl,dy,derived_by)+.25*(pair_loss(tl,train_pairs)+pair_loss(dl,dp));accuracies={n:float((tl[ix].argmax(1)==y[tr[ix]]).float().mean()) for n,ix in train_by.items()}
        development=consider_teacher(0,float(obj),accuracies);th.append(dict(epoch=0,objective=float(obj),TRAIN_accuracy=accuracies,development=development))
    for epoch in range(1,cfg['teacher_epochs']+1):
        teacher.train();optimizer.param_groups[0]['lr']=.00001+.00099*.5*(1+np.cos(np.pi*(epoch-1)/(cfg['teacher_epochs']-1)))
        for _ in range(cfg['steps_per_epoch']):
            ix=np.concatenate([v[torch.randint(len(v),(64,)).numpy()] for v in train_by.values()]);jj=np.concatenate([v[torch.randint(len(v),(64,)).numpy()] for v in derived_by.values()])
            loss=.5*F.cross_entropy(teacher(tx[ix]),y[tr[ix]],label_smoothing=.02)+.5*F.cross_entropy(teacher(dz[jj]),dy[jj],label_smoothing=.02)
            for zz,pairs in ((tx,train_pairs),(dz,dp)):
                if pairs:
                    pp=[pairs[int(v)] for v in torch.randint(len(pairs),(64,))];n,q=zip(*pp);loss+=.25*paired_margin_loss(teacher(zz[list(q)])-teacher(zz[list(n)]))
            if cfg['Euro_parent_anchor_weight']:
                ai=torch.randint(len(anchor),(128,));loss+=cfg['Euro_parent_anchor_weight']*4*F.kl_div(F.log_softmax(teacher(anchor_z[ai])/2,1),F.softmax(anchor_target[ai]/2,1),reduction='batchmean')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite teacher loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(teacher.parameters(),5.);optimizer.step()
        if epoch%10==0 or epoch==1 or time.monotonic()-started>=cfg['teacher_max_seconds']:
            teacher.eval()
            with torch.no_grad():
                tl=teacher(tx);dl=teacher(dz);obj=.5*task_ce(tl,y[tr],train_by)+.5*task_ce(dl,dy,derived_by)+.25*(pair_loss(tl,train_pairs)+pair_loss(dl,dp))
                if cfg['Euro_parent_anchor_weight']:obj+=cfg['Euro_parent_anchor_weight']*4*F.kl_div(F.log_softmax(teacher(anchor_z)/2,1),F.softmax(anchor_target/2,1),reduction='batchmean')
                accuracies={n:float((tl[ix].argmax(1)==y[tr[ix]]).float().mean()) for n,ix in train_by.items()}
            development=consider_teacher(epoch,float(obj),accuracies)
            th.append(dict(epoch=epoch,objective=float(obj),TRAIN_accuracy=accuracies,development=development));put(a.out/'teacher_history.json',th);put(a.out/'status.json',dict(state='teacher',**th[-1],updated=time.time()))
        if time.monotonic()-started>=cfg['teacher_max_seconds'] or (epoch>=100 and epoch-last_improved>=60):break
    if warm and sha(warm['path'])!=warm['SHA256']:raise RuntimeError('Sealed warm teacher changed')
    artifact=dict(state_dict=teacher.state_dict(),transform=transform,parent_SHA256=a.parent_sha256,teacher_dropout=cfg.get('teacher_dropout',0.),**DISCLOSURE);torch.save(artifact,a.out/'last_offline_teacher.pt')
    artifact['state_dict']=best_state;torch.save(artifact,a.out/'offline_teacher.pt');teacher=residual_teacher_from_artifact(torch.load(a.out/'offline_teacher.pt',map_location='cpu',weights_only=False))
    groups=[('TRAIN',tr),('all_development',te),('exposed_development',exposed),('unexposed_development',unexposed)]
    with torch.no_grad():tl=teacher(x@transform.T);ts={n:evaluate(nn.Identity(),tl,y,tasks,ix) for n,ix in groups}
    tb={n:brief(v) for n,v in ts.items()};quality=all(tb['TRAIN'][n]['accuracy']+1e-7>=cfg['TRAIN_floor'] for n in names if n!='eurosat' or cfg.get('all_TEST_sources_authorized')) and all(tb['all_development'][n]['accuracy']+1e-7>=v for n,v in cfg.get('teacher_quality_floor',cfg['development_floor']).items())
    teacher_hash=sha(a.out/'offline_teacher.pt');put(a.out/'teacher_development.json',dict(scores=ts,source_indices=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),**DISCLOSURE))
    put(a.out/'teacher_report.json',dict(passed=True,quality_gate_passed=quality,teacher_scores=tb,teacher_sha256=teacher_hash,strict_CPU_reloaded=True,**DISCLOSURE))
    common=dict(passed=True,teacher_quality_gate_passed=quality,teacher_scores=tb,teacher_sha256=teacher_hash,frozen_upstream=upstream_state(model)==before,strict_CPU_reloaded=True,SDK=False,GPU=False,**DISCLOSURE)
    if not quality:
        put(a.out/'report.json',dict(**common,student_started=False,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='teacher_below_gate',updated=time.time()));return
    for v in teacher.parameters():v.requires_grad_(False)
    teacher_before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    weight=nn.Parameter(initial.clone());optimizer=torch.optim.AdamW([weight],lr=.001,weight_decay=.0001);baseline=evaluate(model.shared_head,x,y,tasks,te)
    best_selection=student_checkpoint_key(baseline,cfg);best_weight=model.shared_head.weight.detach().clone();best_epoch=0;history=[];ss=time.monotonic();std=x[tr].std(0,unbiased=False)
    for epoch in range(1,cfg['student_epochs']+1):
        optimizer.param_groups[0]['lr']=.00001+.00099*.5*(1+np.cos(np.pi*(epoch-1)/(cfg['student_epochs']-1)));total=0.
        for _ in range(cfg['steps_per_epoch']):
            ix=np.concatenate([v[torch.randint(len(v),(64,)).numpy()] for v in train_by.values()]);batch=(x[tr[ix]]+torch.randn(len(ix),784)*std*.001).clamp_min(0);batch=batch/batch.mean(1,keepdim=True);z=batch@transform.T
            with torch.no_grad():target=teacher(z)
            logits=z@weight.T;loss=.5*F.cross_entropy(logits,y[tr[ix]],label_smoothing=.02)+dkd_loss(logits,target,y[tr[ix]])
            if cfg['Euro_parent_anchor_weight']:
                mask=tasks[tr[ix]]=='eurosat'
                with torch.no_grad():oldtarget=parent(batch[mask])
                loss+=4*F.kl_div(F.log_softmax(logits[mask]/2,1),F.softmax(oldtarget/2,1),reduction='batchmean')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite student loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([weight],1.);optimizer.step();total+=float(loss.detach())
        if epoch%5==0 or cfg.get('whole_stage_student_TEST_selection') or time.monotonic()-ss>=cfg['student_max_seconds']:
            with torch.no_grad():model.shared_head.weight.copy_((weight@transform).float());scores=evaluate(model.shared_head,x,y,tasks,te)
            selection=student_checkpoint_key(scores,cfg)
            if selection>best_selection:best_selection=selection;best_epoch=epoch;best_weight=model.shared_head.weight.detach().clone()
            history.append(dict(epoch=epoch,loss=total/cfg['steps_per_epoch'],scores=brief(scores),best_epoch=best_epoch));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='student',**history[-1],updated=time.time()))
        if time.monotonic()-ss>=cfg['student_max_seconds'] or (cfg.get('whole_stage_student_TEST_selection') and best_selection==(0,0)):break
    saved_scores={}
    for role,w in [('best',best_weight),('last',(weight.detach()@transform).float())]:
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before or any(not torch.equal(v,teacher_before[k]) for k,v in teacher.state_dict().items()):raise RuntimeError('Frozen teacher/upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**DISCLOSURE);path=a.out/(role+'_checkpoint.pt');torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,w):raise RuntimeError('Original graph CPU replay differs')
        scores={n:evaluate(loaded.shared_head,x,y,tasks,ix) for n,ix in groups};saved_scores[role]={n:brief(v) for n,v in scores.items()}
        put(a.out/(role+'_development.json'),dict(scores=scores,source_indices=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),**DISCLOSURE))
    band=all(cfg['development_floor'][n]-1e-7<=saved_scores['best']['all_development'][n]['accuracy']<=upper+1e-7 for n,upper in cfg['development_ceiling'].items()) if 'development_ceiling' in cfg else None
    put(a.out/'report.json',dict(**common,student_started=True,all_tasks_within_requested_band=band,original_student_parameters=7840,new_inference_layers=0,baseline=brief(baseline),best=saved_scores['best'],last=saved_scores['last'],best_epoch=best_epoch,retained_parent=best_epoch==0,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),seconds=time.monotonic()-started))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
