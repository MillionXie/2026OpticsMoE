"""One finite stronger offline teacher and original Linear DKD student.

Same sealed500 CLEVR derivative sources, never original EVAL gradient rows.
All results inherit their information and are development-only, not independent
TEST. No optical, acquisition, upstream, exposure or deployed graph change.
"""
import argparse,copy,json,time,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .fit_augmented_residual_teacher import ResidualTeacher,residual_teacher_from_artifact,augment_train
from .fit_gated_linear_teacher import DISCLOSURE,select_sources,paired_indices,paired_derivative_noise,full_covariance_transform,paired_margin_loss
from .refine_measured_ce import contrast_transform
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import put,sha


def dkd_loss(student,teacher,labels,temperature=2.):
    """Decoupled target-vs-rest and conditional non-target KL, fixed1/4 weights."""
    mask=F.one_hot(labels,num_classes=student.shape[1]).bool()
    sp=F.softmax(student/temperature,1);tp=F.softmax(teacher/temperature,1)
    st=(sp*mask).sum(1,keepdim=True).clamp(1e-7,1-1e-7);tt=(tp*mask).sum(1,keepdim=True).clamp(1e-7,1-1e-7)
    target=F.kl_div(torch.cat((st,1-st),1).log(),torch.cat((tt,1-tt),1),reduction='batchmean')
    sl=student/temperature;tl=teacher/temperature
    sr=sl.masked_select(~mask).reshape(len(student),-1);tr=tl.masked_select(~mask).reshape(len(teacher),-1)
    other=F.kl_div(F.log_softmax(sr,1),F.softmax(tr,1),reduction='batchmean')
    return (target+4*other)*temperature**2


def whole_stage_selection(scores,floors):
    return (min(scores[n]['balanced_accuracy']-floors[n] for n in floors),
            float(np.mean([v['balanced_accuracy'] for v in scores.values()])))


def validate_profile(profile):
    expected=json.loads(Path(__file__).with_name('config_B_balanced_residual_teacher_20261025.json').read_text())
    if profile!=expected:raise ValueError('Only exact published finite B teacher profile is admitted')
    return profile


def weak_pair_values(positive,negative,std,sigma=.001):
    noise=torch.randn_like(positive)*std*sigma
    a=(positive+noise).clamp_min(0);b=(negative+noise).clamp_min(0)
    return a/a.mean(1,keepdim=True).clamp_min(1e-12),b/b.mean(1,keepdim=True).clamp_min(1e-12)


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('capture','parent','profile','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--parent-sha256',required=True);parser.add_argument('--acknowledge-test-derived-development',action='store_true');a=parser.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Disclosed partial-EVAL adaptation required')
    cfg=validate_profile(json.loads(a.profile.read_text()));torch.set_num_threads(2);torch.manual_seed(20261025)
    capture=json.loads((a.capture/'report.json').read_text())
    if not capture['passed'] or capture['pilot'] or not capture['SDK_released'] or sha(a.capture/'features.npz')!=capture['features_sha256']:raise ValueError('Sealed full capture required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp['architecture']!='d2nn' or cp['stage']!=2 or cp.get('measured_features_sha256')!=capture['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Original measured D2NN B only')
    parent_head=copy.deepcopy(model.shared_head).eval()
    for parameter in parent_head.parameters():parameter.requires_grad_(False)
    with np.load(a.capture/'features.npz',allow_pickle=False) as data:
        x=torch.tensor(data['features'],dtype=torch.float32);y=torch.tensor(data['labels'],dtype=torch.long);tasks=data['tasks'].copy();ids=data['source_indices'].copy();splits=data['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');old=tr[tasks[tr]=='eurosat'];current=tr[tasks[tr]=='clevr']
    if (len(old),len(current),len(te))!=(300,1000,2000) or any(int((tasks[te]==n).sum())!=1000 for n in ('eurosat','clevr')):raise ValueError('Fixed phase identity changed')
    selected={'eurosat':[],'clevr':select_sources(ids[te][tasks[te]=='clevr'],y[te][tasks[te]=='clevr'].numpy(),'clevr',500)}
    exposed=np.array([i for i in te if str(tasks[i])=='clevr' and int(ids[i]) in selected['clevr']],dtype=int);unexposed=np.setdiff1d(te,exposed)
    pairs=paired_indices(ids[current],y[current].numpy(),tasks[current]);derivative_pairs=paired_indices(ids[exposed],y[exposed].numpy(),tasks[exposed])
    neg=torch.tensor([current[n] for n,p in pairs]);pos=torch.tensor([current[p] for n,p in pairs]);std=x[tr].std(0,unbiased=False)
    transform,conditioning=full_covariance_transform(x[tr],contrast_transform(x[tr]));transform=transform.float()
    initial=(model.shared_head.weight.detach().double()@torch.linalg.inv(transform.double())).float();teacher=ResidualTeacher(initial,cfg['hidden'],cfg['teacher_dropout'])
    derived=[];derived_y=[]
    for repeat in range(10):
        neighbor=[]
        for _ in derivative_pairs:
            pair=pairs[int(torch.randint(len(pairs),(1,)).item())];neighbor.append(pair)
        target=x[exposed].clone();partner=torch.empty_like(target)
        for (n,p),(tn,tp) in zip(derivative_pairs,neighbor):partner[n]=x[current[tn]];partner[p]=x[current[tp]]
        mix=.02 if repeat%2==0 else .05;values=(1-mix)*target+mix*partner+paired_derivative_noise(target,std,.001,derivative_pairs)
        values=values.clamp_min(0);values=values/values.mean(1,keepdim=True).clamp_min(1e-12)
        if torch.any(torch.all(values==target,1)):raise ValueError('Identity EVAL derivative forbidden')
        derived.append(values);derived_y.append(y[exposed])
    dx=torch.cat(derived);dy=torch.cat(derived_y);all_dp=[(n+j*len(exposed),p+j*len(exposed)) for j in range(10) for n,p in derivative_pairs]
    dn=torch.tensor([n for n,p in all_dp]);dp=torch.tensor([p for n,p in all_dp])
    old_mean=x[old].mean(0,keepdim=True);old_std=x[old].std(0,unbiased=False)
    anchor=torch.cat([x[old]]+[augment_train(x[old],old_mean,old_std) for _ in range(4)])
    with torch.no_grad():anchor_target=parent_head(anchor);az=anchor@transform.T
    def logits(values):return teacher(values@transform.T)
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    put(a.out/'protocol.json',dict(scope='partial_development',profile=cfg,profile_sha256=sha(a.profile),parent_SHA256=a.parent_sha256,capture_features_SHA256=capture['features_sha256'],selection_source_ids=selected,
        source_selection_uses_model_outputs=False,original_EVAL_rows_used_for_gradients=0,teacher_selection='Lowest complete TRAIN/derived/old-TRAIN-anchor objective, no EVAL teacher reselection',
        teacher='Original fixed Linear plus128-GELU-Dropout0-Linear residual, offline only',teacher_objective='TRAIN equal-task CE.5 + derivative CE.5 + pair ranking.25 each + old TRAIN/augmentation parentKL2(T2)',
        teacher_TRAIN_floor={'clevr':.90},teacher_development_floor=cfg['development_floor'],student='Original7840 Linear, TRAIN-only CE.5 + DKD(alpha1,beta4,T2) + pair teacher margin.25 + old TRAIN parentKL1',
        teacher_covariance_conditioning=conditioning,student_checkpoint_selection='Worst full-task deficit then mean; one PT per phase; no upper-score clipping',SDK=False,GPU=False,CPU_threads=2,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),**DISCLOSURE))
    optimizer=torch.optim.AdamW(teacher.parameters(),lr=.001,weight_decay=.0001);best_objective=float('inf');best_state=None;teacher_history=[];last_improvement=0
    def objective():
        with torch.no_grad():
            o=logits(x[old]);c=logits(x[current]);d=logits(dx)
            loss=.25*(F.cross_entropy(o,y[old],label_smoothing=.02)+F.cross_entropy(c,y[current],label_smoothing=.02))+.5*F.cross_entropy(d,dy,label_smoothing=.02)
            loss+=.25*paired_margin_loss(logits(x[pos])-logits(x[neg]))+.25*paired_margin_loss(d[dp]-d[dn])
            loss+=8*F.kl_div(F.log_softmax(teacher(az)/2,1),F.softmax(anchor_target/2,1),reduction='batchmean')
            return float(loss),float((c.argmax(1)==y[current]).float().mean())
    for epoch in range(1,cfg['teacher_epochs']+1):
        teacher.train();total=0.
        optimizer.param_groups[0]['lr']=.00001+.00099*.5*(1+np.cos(np.pi*(epoch-1)/(cfg['teacher_epochs']-1)))
        for _ in range(cfg['steps_per_epoch']):
            oi=old[torch.randint(len(old),(128,))];pi=torch.randint(len(pos),(64,));di=torch.randint(len(dp),(64,));ai=torch.randint(len(anchor),(128,))
            pa,na=weak_pair_values(x[pos[pi]],x[neg[pi]],std);dpa,dna=weak_pair_values(dx[dp[di]],dx[dn[di]],std)
            co,cn=logits(pa),logits(na);do,dnlog=logits(dpa),logits(dna);oldlog=logits(x[oi])
            loss=.25*F.cross_entropy(oldlog,y[oi],label_smoothing=.02)+.125*(F.cross_entropy(co,torch.ones(len(co),dtype=torch.long),label_smoothing=.02)+F.cross_entropy(cn,torch.zeros(len(cn),dtype=torch.long),label_smoothing=.02))
            loss+=.25*(F.cross_entropy(do,torch.ones(len(do),dtype=torch.long),label_smoothing=.02)+F.cross_entropy(dnlog,torch.zeros(len(dnlog),dtype=torch.long),label_smoothing=.02))
            loss+=.25*paired_margin_loss(co-cn)+.25*paired_margin_loss(do-dnlog)
            loss+=8*F.kl_div(F.log_softmax(teacher(az[ai])/2,1),F.softmax(anchor_target[ai]/2,1),reduction='batchmean')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite teacher objective')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(teacher.parameters(),5.);optimizer.step();total+=float(loss.detach())
        if epoch%10==0 or epoch==1 or time.monotonic()-started>=cfg['teacher_max_seconds']:
            teacher.eval();value,accuracy=objective()
            if value<best_objective:best_objective=value;best_state=copy.deepcopy(teacher.state_dict());last_improvement=epoch
            teacher_history.append(dict(epoch=epoch,complete_objective=value,CLEVR_TRAIN_accuracy=accuracy,best_objective=best_objective))
            put(a.out/'teacher_history.json',teacher_history);put(a.out/'status.json',dict(state='teacher',**teacher_history[-1],updated=time.time()))
        if time.monotonic()-started>=cfg['teacher_max_seconds'] or (epoch>=100 and epoch-last_improvement>=60):break
    teacher.eval();artifact=dict(state_dict=teacher.state_dict(),transform=transform,parent_SHA256=a.parent_sha256,teacher_dropout=0.,**DISCLOSURE);torch.save(artifact,a.out/'last_offline_teacher.pt')
    artifact['state_dict']=best_state;torch.save(artifact,a.out/'offline_teacher.pt');teacher=residual_teacher_from_artifact(torch.load(a.out/'offline_teacher.pt',map_location='cpu',weights_only=False))
    if not torch.equal(teacher(x[tr[:8]]@transform.T),residual_teacher_from_artifact(artifact)(x[tr[:8]]@transform.T)):raise RuntimeError('Teacher CPU reload differs')
    groups=[('TRAIN',tr),('all_development',te),('exposed_development',exposed),('unexposed_development',unexposed)]
    with torch.no_grad():teacher_scores={n:evaluate(nn.Identity(),teacher(x@transform.T),y,tasks,ix) for n,ix in groups}
    teacher_brief={n:brief(v) for n,v in teacher_scores.items()};quality=teacher_brief['TRAIN']['clevr']['accuracy']+1e-7>=.90 and all(teacher_brief['all_development'][n]['accuracy']+1e-7>=floor for n,floor in cfg['development_floor'].items())
    put(a.out/'teacher_development.json',dict(scores=teacher_scores,selection_source_ids=selected,**DISCLOSURE));teacher_hash=sha(a.out/'offline_teacher.pt')
    put(a.out/'teacher_report.json',dict(passed=True,quality_gate_passed=quality,teacher_scores=teacher_brief,teacher_sha256=teacher_hash,strict_CPU_reloaded=True,**DISCLOSURE))
    common=dict(passed=True,teacher_quality_gate_passed=quality,teacher_scores=teacher_brief,teacher_sha256=teacher_hash,strict_CPU_reloaded=True,frozen_upstream=upstream_state(model)==before,SDK=False,GPU=False,**DISCLOSURE)
    if not quality:
        put(a.out/'report.json',dict(**common,student_started=False,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='teacher_below_gate',updated=time.time()));return
    student_started=time.monotonic();weight=nn.Parameter(initial.clone());optimizer=torch.optim.AdamW([weight],lr=.001,weight_decay=.0001)
    baseline=evaluate(model.shared_head,x,y,tasks,te);best_selection=whole_stage_selection(baseline,cfg['development_floor']);best_weight=model.shared_head.weight.detach().clone();best_epoch=0;history=[]
    for epoch in range(1,cfg['student_epochs']+1):
        optimizer.param_groups[0]['lr']=.00001+.00099*.5*(1+np.cos(np.pi*(epoch-1)/(cfg['student_epochs']-1)));total=0.
        for _ in range(12):
            oi=old[torch.randint(len(old),(64,))];pi=torch.randint(len(pos),(64,));pa,na=weak_pair_values(x[pos[pi]],x[neg[pi]],std)
            batch=torch.cat((x[oi],pa,na));labels=torch.cat((y[oi],torch.ones(len(pa),dtype=torch.long),torch.zeros(len(na),dtype=torch.long)));z=batch@transform.T
            with torch.no_grad():target=teacher(z);old_target=parent_head(x[oi])
            student=z@weight.T;loss=.5*F.cross_entropy(student,labels,label_smoothing=.02)+dkd_loss(student,target,labels)
            sm=student[64:128,1]-student[64:128,0]-student[128:,1]+student[128:,0];tm=target[64:128,1]-target[64:128,0]-target[128:,1]+target[128:,0]
            loss+=.25*F.smooth_l1_loss(sm,tm)+4*F.kl_div(F.log_softmax(student[:64]/2,1),F.softmax(old_target/2,1),reduction='batchmean')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite student objective')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([weight],1.);optimizer.step();total+=float(loss.detach())
        if epoch%5==0 or time.monotonic()-student_started>=cfg['student_max_seconds']:
            with torch.no_grad():model.shared_head.weight.copy_((weight@transform).float());values=evaluate(model.shared_head,x,y,tasks,te)
            selection=whole_stage_selection(values,cfg['development_floor'])
            if selection>best_selection:best_selection=selection;best_epoch=epoch;best_weight=model.shared_head.weight.detach().clone()
            history.append(dict(epoch=epoch,loss=total/12,scores=brief(values),best_epoch=best_epoch));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='student',**history[-1],updated=time.time()))
        if time.monotonic()-student_started>=cfg['student_max_seconds']:break
    saved_scores={}
    for role,w in [('best',best_weight),('last',(weight.detach()@transform).float())]:
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before:raise RuntimeError('Original upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**DISCLOSURE);path=a.out/(role+'_checkpoint.pt');torch.save(saved,path)
        loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,w):raise RuntimeError('Saved original Linear or upstream differs')
        with torch.no_grad():values={n:evaluate(loaded.shared_head,x,y,tasks,ix) for n,ix in groups}
        saved_scores[role]={n:brief(v) for n,v in values.items()};put(a.out/(role+'_development.json'),dict(scores=values,**DISCLOSURE))
    put(a.out/'report.json',dict(**common,student_started=True,student_original_parameters=7840,new_inference_layers=0,baseline=brief(baseline),best=saved_scores['best'],last=saved_scores['last'],best_epoch=best_epoch,retained_parent=best_epoch==0,
        best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),seconds=time.monotonic()-started))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
