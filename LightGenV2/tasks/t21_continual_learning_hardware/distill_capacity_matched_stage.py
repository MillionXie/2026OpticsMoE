"""Finite convex TRAIN distillation and explicitly partial-source linear assistant.

The assistant sees exactly the sealed teacher's allowed nonidentity derivatives,
never original EVAL rows; it is not a TRAIN-only model. Final function student
queries TRAIN/augmentations only and still inherits partial TEST information.
All inference is the original shared7840 Linear, frozen optical prefix, no SDK.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .fit_augmented_residual_teacher import residual_teacher_from_artifact
from .fit_stage_balanced_teacher import derivatives
from .fit_gated_linear_teacher import DISCLOSURE,paired_indices,paired_margin_loss
from .fit_B_balanced_residual_teacher import whole_stage_selection
from .audit_student_transfer import linear_projection
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import put,sha


def validate_profile(cfg):
    files=('config_capacity_matched_distillation_20261025.json','config_capacity_matched_hardpair_20261025.json')
    if not any(cfg==json.loads(Path(__file__).with_name(n).read_text()) for n in files):raise ValueError('Only published finite B/C profiles')


def equal_task_weights(tasks,mass=1.):
    return torch.tensor([mass/(len(set(tasks))*int((tasks==n).sum())) for n in tasks],dtype=torch.float64)


def convex_objective(logits,labels,target,weights,cfg):
    temp=cfg['temperature'];ce=F.cross_entropy(logits,labels,label_smoothing=.02,reduction='none')
    kl=F.kl_div(F.log_softmax(logits/temp,1),F.softmax(target/temp,1),reduction='none').sum(1)*temp**2
    return (weights*(cfg['cross_entropy_weight']*ce+cfg['teacher_KL_weight']*kl)).sum()


def regret_objective(losses,references):
    if not losses or set(losses)!=set(references):raise ValueError('Exact complete task regret references required')
    return .1*torch.logsumexp(torch.stack([v-references[n] for n,v in losses.items()])/.1,0)


def TRAIN_queries(x,tasks,cfg):
    std=x.std(0,unbiased=False);values=[x]
    for repeat in range(cfg['TRAIN_augmentations']):
        q=(x+torch.randn_like(x)*std*cfg['TRAIN_noise_std'][repeat%3]).clamp_min(0);values.append(q/q.mean(1,keepdim=True).clamp_min(1e-12))
    return torch.cat(values),np.tile(tasks,len(values))


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','teacher_run','profile','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--mode',choices=('train_convex','assistant','taskwise_diagnostic','regret_assistant'),required=True)
    p.add_argument('--taskwise-run',type=Path)
    p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Inherited partial TEST information must be disclosed')
    cfg=json.loads(a.profile.read_text());validate_profile(cfg);torch.set_num_threads(2);torch.manual_seed(20261025)
    if cfg['schema'].endswith('v2') and a.mode!='assistant':raise ValueError('Hard-pair profile is an assistant-only method, no extra TRAIN-convex scan')
    cr=json.loads((a.capture/'report.json').read_text());trp=json.loads((a.teacher_run/'protocol.json').read_text());rr=json.loads((a.teacher_run/'report.json').read_text())
    if not cr['passed'] or cr['pilot'] or not cr['SDK_released'] or sha(a.capture/'features.npz')!=cr['features_sha256'] or trp['capture_features_SHA256']!=cr['features_sha256']:raise ValueError('Full sealed capture required')
    if not rr['teacher_quality_gate_passed'] or rr['valid_for_formal_test_table'] is not False or trp['parent_SHA256']!=a.parent_sha256 or sha(a.teacher_run/'offline_teacher.pt')!=rr['teacher_sha256']:raise ValueError('Exact qualified same-parent teacher required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp['architecture']!='d2nn' or cp['stage'] not in cfg['allowed_stages'] or cp['measured_features_sha256']!=cr['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Original measured B/C only')
    regret=None
    if a.mode=='regret_assistant':
        if cp['stage'] not in (2,3) or a.taskwise_run is None:raise ValueError('Finite B/C regret method requires sealed taskwise diagnostic')
        dp=json.loads((a.taskwise_run/'protocol.json').read_text());dr=json.loads((a.taskwise_run/'report.json').read_text())
        if dp['mode']!='taskwise_diagnostic' or dp['parent_SHA256']!=a.parent_sha256 or dp['capture_features_SHA256']!=cr['features_sha256'] or dp['teacher_SHA256']!=rr['teacher_sha256'] or dp['profile']!=cfg or not dr['passed'] or not dr['no_task_head_splicing']:raise ValueError('Exact same taskwise fit objective/provenance required')
        regret={n:v['history'][-1]['objective'] for n,v in dr['taskwise'].items()}
    artifact=torch.load(a.teacher_run/'offline_teacher.pt',map_location='cpu',weights_only=False);teacher=residual_teacher_from_artifact(artifact);transform=artifact['transform'];teacher_state={k:v.clone() for k,v in teacher.state_dict().items()}
    for v in teacher.parameters():v.requires_grad_(False)
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');profile=trp['profile'];floors=profile['development_floor'];selected=trp['selection_source_ids'];names=sorted(floors)
    exposed=np.array([i for i in te if int(ids[i]) in selected[str(tasks[i])]],dtype=int);unex=np.setdiff1d(te,exposed)
    if len(tr)!=1000+300*(cp['stage']-1) or any(int((tasks[te]==n).sum())!=1000 for n in names):raise ValueError('Full fixed TRAIN and evaluation required')
    q,qt=TRAIN_queries(x[tr],tasks[tr],cfg);qy=y[tr].repeat(1+cfg['TRAIN_augmentations']);qid=np.tile(ids[tr],1+cfg['TRAIN_augmentations'])
    if a.mode in ('assistant','taskwise_diagnostic','regret_assistant'):
        dx,dy,dt,di=derivatives(x,y,tasks,ids,tr,exposed);pool=torch.cat((q,dx));labels=torch.cat((qy,dy));pt=np.concatenate((qt,dt));pid=np.concatenate((qid,di))
        weights=torch.cat((equal_task_weights(qt,cfg['assistant_TRAIN_mass']),equal_task_weights(dt,cfg['assistant_derivative_mass'])))
    else:pool=q;labels=qy;pt=qt;pid=qid;weights=equal_task_weights(qt)
    # Pair ranking is on each complete augmented source copy, not accidentally
    # joining the same ID across distinct augmentation repeats.
    pair_deltas=[];pair_tasks=[]
    if a.mode in ('assistant','taskwise_diagnostic','regret_assistant'):
        for values,tt,ii,yy,chunk in ((q,qt,qid,qy,len(tr)),(dx,dt,di,dy,len(exposed))):
            for start in range(0,len(values),chunk):
                pairs=paired_indices(ii[start:start+chunk],yy[start:start+chunk].numpy(),tt[start:start+chunk])
                pair_deltas.extend([values[start+p]-values[start+n] for n,p in pairs])
                pair_tasks.extend([str(tt[start+p]) for n,p in pairs])
    with torch.no_grad():
        target=torch.cat([teacher(v@transform.T) for v in pool.split(256)]).double()
    z=(pool@transform.T).double();initial=(model.shared_head.weight.detach().double()@torch.linalg.inv(transform.double()));weight=torch.nn.Parameter(initial.clone())
    pair_z=torch.stack(pair_deltas).double()@transform.double().T if pair_deltas else None
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();history=[];best_obj=float('inf');best_weight=weight.detach().clone()
    put(a.out/'protocol.json',dict(mode=a.mode,profile=cfg,profile_sha256=sha(a.profile),capture_features_SHA256=cr['features_sha256'],parent_SHA256=a.parent_sha256,teacher_SHA256=rr['teacher_sha256'],selection_source_ids=selected,
        assistant_scope='TRAIN queries and exact allowed source nonidentity derivatives; inherits TEST information' if a.mode!='train_convex' else None,
        final_student_inputs='Original TRAIN and its8 weak augmentations ONLY; no EVAL query inputs',assistant_source_budget_expanded=False,
        fit=f"float64 deterministic full-batch convex CE{cfg['cross_entropy_weight']} + KL{cfg['teacher_KL_weight']}(T2) + fixed coordinate-delta regularization; no EVAL loss/parameter search",
        assistant_gate='Same complete per-task development floors as sealed teacher; actual TRAIN fit disclosed, not assumed90% for linear approximation',
        new_inference_layers=0,original_student_parameters=7840,SDK=False,GPU=False,CPU_threads=2,maximum_fit_seconds=cfg['maximum_seconds']*(len(names) if a.mode=='taskwise_diagnostic' else 1),
        TRAIN_taskwise_objective_reference=regret,regret_smooth_max_temperature=.1 if regret is not None else None,regret_reference_report_SHA256=sha(a.taskwise_run/'report.json') if regret is not None else None,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),**DISCLOSURE))
    if a.mode=='taskwise_diagnostic':
        # Separate objectives diagnose shared-head interference. Never splice
        # these task-specific functions into a deployment or final score table.
        diagnostic={};arrays={}
        for name in names:
            task_started=time.monotonic();mask=pt==name;zz=z[mask];yy=labels[mask];tt=target[mask];ww=weights[mask];ww=ww/ww.sum()
            one=torch.nn.Parameter(initial.clone());optimizer=torch.optim.LBFGS([one],lr=1.,max_iter=1,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11)
            task_pairs=pair_z[np.asarray(pair_tasks)==name] if pair_z is not None else None
            def task_closure():
                optimizer.zero_grad(set_to_none=True);loss=convex_objective(zz@one.T,yy,tt,ww,cfg)+cfg['assistant_delta_L2']*(one-initial)[:,:-1].square().sum()/10
                if task_pairs is not None and len(task_pairs):loss+=cfg['assistant_paired_ranking_weight']*paired_margin_loss(task_pairs@one.T)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite taskwise diagnostic')
                loss.backward();return loss
            task_history=[]
            for step in range(1,cfg['maximum_steps']+1):
                optimizer.step(task_closure);objective=float(task_closure().detach());gradient=float(one.grad.norm())
                if step%10==0 or step==1:task_history.append(dict(step=step,objective=objective,gradient_norm=gradient));put(a.out/'status.json',dict(state='diagnostic_only_taskwise',task=name,step=step,objective=objective,updated=time.time()))
                if time.monotonic()-task_started>=cfg['maximum_seconds'] or gradient<1e-7:break
            folded=(one.detach()@transform.double()).float();head=torch.nn.Linear(784,10,bias=False)
            with torch.no_grad():head.weight.copy_(folded)
            groups=dict(TRAIN=tr,all_development=te,teacher_exposed_development=exposed,not_in_teacher_gradient_development=unex)
            diagnostic[name]=dict(scores={group:brief(evaluate(head,x,y,tasks,ix[tasks[ix]==name])) for group,ix in groups.items()},history=task_history,seconds=time.monotonic()-task_started)
            arrays[name]=folded.numpy()
        if upstream_state(model)!=before or any(not torch.equal(v,teacher_state[k]) for k,v in teacher.state_dict().items()):raise RuntimeError('Diagnostic changed parent/teacher')
        np.savez(a.out/'diagnostic_taskwise_weights.npz',**arrays)
        put(a.out/'report.json',dict(passed=True,mode=a.mode,diagnostic_not_deployment=True,no_task_head_splicing=True,original_parent_unchanged=True,taskwise=diagnostic,
            comparison_scope='Same authorized TRAIN/derivative pool and convex objective; task-specific diagnostic lacks deployed shared-head constraint; not a universal capacity bound',
            strict_CPU_reloaded=False,frozen_upstream=True,SDK=False,GPU=False,seconds=time.monotonic()-started,**DISCLOSURE));put(a.out/'status.json',dict(state='complete_diagnostic_only',updated=time.time()));return
    opt=torch.optim.LBFGS([weight],lr=1.,max_iter=1,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11)
    strength=cfg[('assistant' if a.mode=='regret_assistant' else a.mode)+'_delta_L2'];calls=0
    task_views={n:(z[pt==n],labels[pt==n],target[pt==n],weights[pt==n]/weights[pt==n].sum(),pair_z[np.asarray(pair_tasks)==n] if pair_z is not None else None) for n in names} if regret is not None else None
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True);regularizer=strength*(weight-initial)[:,:-1].square().sum()/10
        if regret is None:
            loss=convex_objective(z@weight.T,labels,target,weights,cfg)+regularizer
            if pair_z is not None:loss+=cfg['assistant_paired_ranking_weight']*paired_margin_loss(pair_z@weight.T)
        else:
            task_losses={}
            for n,(zz,yy,tt,ww,pairs) in task_views.items():
                task_loss=convex_objective(zz@weight.T,yy,tt,ww,cfg)+regularizer
                if pairs is not None and len(pairs):task_loss+=cfg['assistant_paired_ranking_weight']*paired_margin_loss(pairs@weight.T)
                task_losses[n]=task_loss
            loss=regret_objective(task_losses,regret)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite convex distillation objective')
        loss.backward();calls+=1;return loss
    for step in range(1,cfg['maximum_steps']+1):
        opt.step(closure);objective=float(closure().detach());grad=float(weight.grad.norm())
        if objective<best_obj:best_obj=objective;best_weight=weight.detach().clone()
        if step%10==0 or step==1:
            history.append(dict(step=step,objective=objective,gradient_norm=grad,best_objective=best_obj,closure_calls=calls));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='convex_'+a.mode,**history[-1],updated=time.time()))
        if time.monotonic()-started>=cfg['maximum_seconds'] or grad<1e-7:break
    folded=(best_weight@transform.double()).float();last_folded=(weight.detach()@transform.double()).float();baseline=evaluate(model.shared_head,x,y,tasks,te)
    groups=[('TRAIN',tr),('all_development',te),('teacher_exposed_development',exposed),('not_in_teacher_gradient_development',unex)]
    def save(role,w,tag=''):
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before or any(not torch.equal(v,teacher_state[k]) for k,v in teacher.state_dict().items()):raise RuntimeError('Frozen upstream/teacher changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**DISCLOSURE);path=a.out/(tag+role+'_checkpoint.pt');torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,w):raise RuntimeError('Original graph CPU reload failed')
        scores={label:evaluate(loaded.shared_head,x,y,tasks,ix) for label,ix in groups};put(a.out/(tag+role+'_development.json'),dict(scores=scores,source_indices=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),**DISCLOSURE));return {n:brief(v) for n,v in scores.items()}
    assistant_scores=None;function_audit=None
    if a.mode in ('assistant','regret_assistant'):
        assistant_scores=save('best',folded,'assistant_');save('last',last_folded,'assistant_')
        admitted=all(assistant_scores['all_development'][n]['accuracy']+1e-7>=floor for n,floor in floors.items())
        put(a.out/'assistant_report.json',dict(passed=True,quality_gate_passed=admitted,scores=assistant_scores,strict_CPU_reloaded=True,**DISCLOSURE))
        if not admitted:
            put(a.out/'report.json',dict(passed=True,mode=a.mode,assistant_quality_gate_passed=False,final_student_started=False,assistant=assistant_scores,strict_CPU_reloaded=True,frozen_upstream=True,SDK=False,GPU=False,seconds=time.monotonic()-started,**DISCLOSURE));put(a.out/'status.json',dict(state='assistant_below_gate',updated=time.time()));return
        # The intermediate teacher is now exactly a Linear function. Recover
        # that function solely from TRAIN queries, rather than copying weights
        # or fitting EVAL points in this final stage.
        with torch.no_grad():soft=q.double()@folded.double().T;soft-=soft.mean(1,keepdim=True)
        projected,rank=linear_projection((q@transform.T).double(),soft,qt);candidate=(projected.T@transform.double()).float()
        if rank['rank']!=784:raise RuntimeError('TRAIN query span insufficient to identify original Linear function')
        with torch.no_grad():
            reconstructed=q.double()@candidate.double().T;original=q.double()@folded.double().T;original-=original.mean(1,keepdim=True)
        function_audit=dict(TRAIN_query_rank=rank,max_TRAIN_logit_error=float((reconstructed-original).abs().max()),TRAIN_teacher_argmax_agreement=float((reconstructed.argmax(1)==original.argmax(1)).double().mean()),student_fit_EVAL_inputs=0)
    else:candidate=folded
    with torch.no_grad():model.shared_head.weight.copy_(candidate)
    candidate_scores=evaluate(model.shared_head,x,y,tasks,te);retain=whole_stage_selection(candidate_scores,floors)<=whole_stage_selection(baseline,floors)
    best=save('best',cp['model']['shared_head.weight'].clone() if retain else candidate);last=save('last',candidate)
    put(a.out/'report.json',dict(passed=True,mode=a.mode,baseline=brief(baseline),candidate=brief(candidate_scores),best=best,last=last,retained_parent=retain,
        assistant_quality_gate_passed=True if a.mode in ('assistant','regret_assistant') else None,final_student_started=True,assistant=assistant_scores,TRAIN_function_recovery=function_audit,
        strict_CPU_reloaded=True,frozen_upstream=True,original_student_parameters=7840,new_inference_layers=0,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),
        SDK=False,GPU=False,seconds=time.monotonic()-started,**DISCLOSURE));put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
