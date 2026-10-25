"""One bounded shared Linear margin teacher, then TRAIN-query student recovery.

The intermediate teacher uses the already authorized partial TEST derivatives.
Neither it nor its student is an independent-test result. No source expansion,
task-specific inference, changed optical prefix, new model layer or SDK access.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .distill_capacity_matched_stage import TRAIN_queries
from .fit_augmented_residual_teacher import residual_teacher_from_artifact
from .fit_stage_balanced_teacher import derivatives
from .fit_gated_linear_teacher import DISCLOSURE,paired_indices,paired_derivative_noise,paired_margin_loss
from .audit_student_transfer import linear_projection
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import put,sha


BANDS={2:{'eurosat':(483,533),'clevr':(710,760)},
       3:{'eurosat':(512,562),'clevr':(631,681),'speech_binary':(472,522)}}


def within_bands(scores,stage):
    return set(scores)==set(BANDS[stage]) and all(v['n']==1000 and lo<=v['correct']<=hi for n,v in scores.items() for lo,hi in [BANDS[stage][n]])


def bounded_teacher_margin(logits,labels):
    target=logits.gather(1,labels[:,None]).squeeze(1)
    rivals=logits.clone();rivals.scatter_(1,labels[:,None],-torch.inf)
    # Only correct teacher decisions request additional separation. Wrong
    # decisions receive the label-based minimum, not erroneous soft logits.
    return (target-rivals.max(1).values).detach().clamp(0,1.5)+.5


def margin_rows(logits,labels,margins):
    true=logits.gather(1,labels[:,None]).squeeze(1)
    rivals=logits.clone();rivals.scatter_(1,labels[:,None],-torch.inf)
    return .5*F.cross_entropy(logits,labels,reduction='none')+.5*F.softplus(rivals.max(1).values-true+margins)


def bounded_margin_rows(logits,labels,margins):
    adjusted=logits.clone();adjusted.scatter_add_(1,labels[:,None],-margins[:,None])
    probability=adjusted.softmax(1).gather(1,labels[:,None]).squeeze(1).clamp_min(1e-9)
    return (1-probability.pow(.7))/.7


def source_average(loss,groups,count):
    totals=torch.zeros(count,dtype=loss.dtype,device=loss.device).scatter_add_(0,groups,loss)
    counts=torch.zeros_like(totals).scatter_add_(0,groups,torch.ones_like(loss))
    if bool((counts==0).any()):raise ValueError('Missing declared source group')
    return totals/counts


def smooth_tail(values,threshold,fraction=.4):
    # Smooth CVaR: difficult TRAIN/authorized-derivative sources have influence
    # without selecting original TEST errors or overwhelming another task.
    return threshold+(.05*F.softplus((values-threshold)/.05)).mean()/fraction


def robust_queries(x,labels,tasks,ids,cfg):
    """TRAIN-only bounded noise, shared within each original binary pair."""
    std=x.std(0,unbiased=False);pairs=paired_indices(ids,labels.numpy(),tasks);values=[x]
    for repeat in range(cfg['TRAIN_augmentations']):
        noise=paired_derivative_noise(x,std,cfg['TRAIN_noise_std'][repeat%3],pairs);noise=noise-noise.mean(1,keepdim=True)
        q=(x+noise).clamp_min(0);values.append(q/q.mean(1,keepdim=True).clamp_min(1e-12))
    return torch.cat(values),np.tile(tasks,len(values))


def robust_derivatives(x,y,tasks,ids,train,exposed):
    """Same sealed sources/count; stronger same-class TRAIN mixing, no new IDs."""
    std=x[train].std(0,unbiased=False);values=[];source=x[exposed];pairs=paired_indices(ids[exposed],y[exposed].numpy(),tasks[exposed])
    for repeat in range(10):
        partner=torch.empty_like(source)
        for name in sorted(set(tasks[exposed])):
            ix=np.flatnonzero(tasks[exposed]==name);pool=train[tasks[train]==name]
            if name=='eurosat':
                for i in ix:
                    choices=pool[y[pool].numpy()==int(y[exposed[i]])];partner[i]=x[choices[int(torch.randint(len(choices),(1,)))]]
            else:
                available=paired_indices(ids[pool],y[pool].numpy(),tasks[pool])
                for n,p in pairs:
                    if tasks[exposed[n]]!=name:continue
                    a,b=available[int(torch.randint(len(available),(1,)))];partner[n]=x[pool[a]];partner[p]=x[pool[b]]
        mix=.1 if repeat%2==0 else .2;out=((1-mix)*source+mix*partner+paired_derivative_noise(source,std,.001,pairs)).clamp_min(0);out=out/out.mean(1,keepdim=True).clamp_min(1e-12)
        if torch.any(torch.all(out==source,1)):raise ValueError('Identity derivative forbidden')
        values.append(out)
    return torch.cat(values),y[exposed].repeat(10),np.tile(tasks[exposed],10),np.tile(ids[exposed],10)


def source_partitions(tasks,ids,kinds,labels,bags):
    """Source groups, not augmentation rows; keep both binary partners together."""
    if bags not in (1,3):raise ValueError('Only one finite fit or fixed three-bag average')
    result=np.zeros(len(ids),dtype=int);rng=np.random.default_rng(20261025)
    for name in sorted(set(tasks)):
        for kind in (0,1):
            rows=np.flatnonzero((tasks==name)&(kinds==kind));groups=ids[rows] if name=='eurosat' else ids[rows]//2
            strata=range(10) if name=='eurosat' else (None,)
            for label in strata:
                mask=np.ones(len(rows),bool) if label is None else labels[rows]==label;keys=np.unique(groups[mask]);rng.shuffle(keys);mapping={int(key):i%bags for i,key in enumerate(keys)}
                result[rows[mask]]=[mapping[int(key)] for key in groups[mask]]
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','teacher_run','profile','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--warm-assistant-run',type=Path);p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Inherited TEST information must be disclosed')
    cfg=json.loads(a.profile.read_text())
    if not any(cfg==json.loads(Path(__file__).with_name(n).read_text()) for n in ('config_margin_teacher_20261025.json','config_margin_robust_teacher_20261025.json','config_margin_bagged_teacher_20261025.json','config_margin_bounded_teacher_20261025.json')):raise ValueError('Only published finite profiles')
    torch.set_num_threads(2);torch.manual_seed(20261025)
    cr=json.loads((a.capture/'report.json').read_text());tp=json.loads((a.teacher_run/'protocol.json').read_text());rr=json.loads((a.teacher_run/'report.json').read_text())
    if not cr['passed'] or cr['pilot'] or not cr['SDK_released'] or sha(a.capture/'features.npz')!=cr['features_sha256'] or tp['capture_features_SHA256']!=cr['features_sha256']:raise ValueError('Sealed same capture required')
    if not rr['teacher_quality_gate_passed'] or rr['valid_for_formal_test_table'] is not False or tp['parent_SHA256']!=a.parent_sha256 or sha(a.teacher_run/'offline_teacher.pt')!=rr['teacher_sha256']:raise ValueError('Exact qualified teacher required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model);stage=cp['stage']
    if cp['architecture']!='d2nn' or stage not in BANDS or cp['measured_features_sha256']!=cr['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Frozen measured B/C original graph only')
    artifact=torch.load(a.teacher_run/'offline_teacher.pt',map_location='cpu',weights_only=False);teacher=residual_teacher_from_artifact(artifact);transform=artifact['transform'];teacher_before={k:v.clone() for k,v in teacher.state_dict().items()}
    for v in teacher.parameters():v.requires_grad_(False)
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');selected=tp['selection_source_ids'];names=sorted(BANDS[stage])
    expected={2:{'eurosat':200,'clevr':800},3:{n:500 for n in names}}[stage]
    if {n:len(v) for n,v in selected.items()}!=expected or len(tr)!=1000+300*(stage-1) or any(int((tasks[te]==n).sum())!=1000 for n in names):raise ValueError('Exact fixed source budget/complete stage required')
    exposed=np.array([i for i in te if int(ids[i]) in selected[str(tasks[i])]],dtype=int);unex=np.setdiff1d(te,exposed)
    robust=cfg.get('paired_TRAIN_and_stronger_same_class_mix',False)
    q,qt=robust_queries(x[tr],y[tr],tasks[tr],ids[tr],cfg) if robust or cfg.get('paired_TRAIN') else TRAIN_queries(x[tr],tasks[tr],cfg);qy=y[tr].repeat(1+cfg['TRAIN_augmentations']);qi=np.tile(ids[tr],1+cfg['TRAIN_augmentations'])
    dx,dy,dt,di=(robust_derivatives if robust else derivatives)(x,y,tasks,ids,tr,exposed);pool=torch.cat((q,dx));labels=torch.cat((qy,dy));pt=np.concatenate((qt,dt));pi=np.concatenate((qi,di));kinds=np.concatenate((np.zeros(len(q),int),np.ones(len(dx),int)))
    pair_views=[]
    if robust:
        for values,tt,ii,yy,chunk in ((q,qt,qi,qy,len(tr)),(dx,dt,di,dy,len(exposed))):
            pairs=[]
            for start in range(0,len(values),chunk):
                pairs.extend((start+n,start+p) for n,p in paired_indices(ii[start:start+chunk],yy[start:start+chunk].numpy(),tt[start:start+chunk]))
            for name in names:
                chosen=[(n,p) for n,p in pairs if tt[n]==name]
                if chosen:pair_views.append(torch.stack([values[p]-values[n] for n,p in chosen]).double()@transform.double().T)
    with torch.no_grad():target=torch.cat([teacher(v@transform.T) for v in pool.split(256)]).double();margins=bounded_teacher_margin(target,labels)
    warm=None;initial_head=model.shared_head.weight.detach().double()
    if cfg.get('bounded_generalized_CE'):
        if a.warm_assistant_run is None:raise ValueError('Sealed same-parent regret assistant required')
        wp=json.loads((a.warm_assistant_run/'protocol.json').read_text());wr=json.loads((a.warm_assistant_run/'report.json').read_text());path=a.warm_assistant_run/'assistant_best_checkpoint.pt'
        if wp['mode']!='regret_assistant' or not wr['passed'] or wp['parent_SHA256']!=a.parent_sha256 or wp['capture_features_SHA256']!=cr['features_sha256'] or wp['teacher_SHA256']!=rr['teacher_sha256']:raise ValueError('Warm teacher identity differs')
        warm_model,_=load_checkpoint(path,sha(path))
        if upstream_state(warm_model)!=before:raise ValueError('Warm teacher optical prefix differs')
        initial_head=warm_model.shared_head.weight.detach().double();warm=dict(path=str(path),SHA256=sha(path),scope='Same frozen-prefix partial-source regret assistant, not a new student source')
    z=(pool@transform.T).double();initial=initial_head@torch.linalg.inv(transform.double());weight=torch.nn.Parameter(initial.clone());threshold=torch.nn.Parameter(torch.ones(len(names),2,dtype=torch.float64))
    views=[]
    for n in names:
        for kind in (0,1):
            rows=np.flatnonzero((pt==n)&(kinds==kind));unique,inverse=np.unique(pi[rows],return_inverse=True);views.append((n,kind,rows,torch.tensor(inverse),len(unique)))
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();history=[];bags=cfg.get('source_bags',1);partitions=source_partitions(pt,pi,kinds,labels.numpy(),bags);bag_weights=[];bag_last=[]
    put(a.out/'protocol.json',dict(profile=cfg,profile_SHA256=sha(a.profile),parent_SHA256=a.parent_sha256,capture_features_SHA256=cr['features_sha256'],teacher_SHA256=rr['teacher_sha256'],selection_source_ids=selected,source_budget_expanded=False,
        objective='Equal tasks/pools; source-mean generalizedCE q=.7 after bounded teacher margins' if cfg.get('bounded_generalized_CE') else 'Equal tasks and TRAIN/allowed-derivative pools; .5 mean+.5 smooth worst40% source margin/CE; clipped correct-teacher margins, not ten-logit matching',
        teacher_selection='Lowest fit objective, no EVAL selection or score ceiling loss',final_student_inputs='TRAIN and its8 weak augmentations only; full-rank function recovery',
        warm_assistant=warm,bounded_loss='Mean source generalizedCE q=.7 after bounded correct-teacher margin; no hardest-tail weighting' if cfg.get('bounded_generalized_CE') else None,
        source_bags=bags,bag_averaging='Fixed equal parameter average of all3 bags, no EVAL choice' if bags==3 else None,
        bag_omission_is_not_independent_holdout='Original strong teacher already knows sealed sources; bagging is variance reduction, not clean source cross-validation',
        derivative_mixing=(.1,.2) if robust else (.02,.05),derivative_noise_TRAIN_std=.001,paired_ranking_weight=cfg.get('paired_ranking_weight',0),
        final_gate_inclusive_correct_count=BANDS[stage],score_clipping=False,task_head_splicing=False,original_student_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),**DISCLOSURE))
    if bags==3:
        if robust or pair_views:raise ValueError('Bag profile has its own fixed weak mixing objective')
        np.savez(a.out/'source_bag_partition.npz',task=pt,source_index=pi,pool_kind=kinds,source_bag=partitions)
    for bag in range(bags):
        bag_started=time.monotonic();best_obj=float('inf');best_weight=initial.clone();weight=torch.nn.Parameter(initial.clone());threshold=torch.nn.Parameter(torch.ones(len(names),2,dtype=torch.float64));bag_views=[]
        for n,kind,ix,unused,count in views:
            selected_rows=ix if bags==1 else ix[partitions[ix]!=bag];keys,inverse=np.unique(pi[selected_rows],return_inverse=True);bag_views.append((n,kind,selected_rows,torch.tensor(inverse),len(keys)))
        opt=torch.optim.LBFGS([weight,threshold],lr=1.,max_iter=1,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11)
        def closure():
            opt.zero_grad(set_to_none=True);rows=(bounded_margin_rows if cfg.get('bounded_generalized_CE') else margin_rows)(z@weight.T,labels,margins);loss=rows.sum()*0
            for n,kind,ix,groups,count in bag_views:
                values=source_average(rows[ix],groups,count);loss+=(values.mean() if cfg.get('bounded_generalized_CE') else .5*values.mean()+.5*smooth_tail(values,threshold[names.index(n),kind]))/(2*len(names))
            if pair_views:loss+=cfg['paired_ranking_weight']*torch.stack([paired_margin_loss(v@weight.T) for v in pair_views]).mean()
            loss+=cfg['delta_L2']*(weight-initial)[:,:-1].square().sum()/10
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite margin teacher objective')
            loss.backward();return loss
        for step in range(1,cfg['maximum_steps']+1):
            opt.step(closure);objective=float(closure().detach());gradient=float(torch.cat([v.grad.flatten() for v in (weight,threshold) if v.grad is not None]).norm())
            if objective<best_obj:best_obj=objective;best_weight=weight.detach().clone()
            if step==1 or step%10==0:
                history.append(dict(bag=bag,step=step,objective=objective,gradient_norm=gradient));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='margin_teacher',**history[-1],updated=time.time()))
            if gradient<1e-7 or time.monotonic()-bag_started>=cfg['maximum_seconds']:break
        bag_weights.append(best_weight);bag_last.append(weight.detach().clone())
    folded=(torch.stack(bag_weights).mean(0)@transform.double()).float();last_folded=(torch.stack(bag_last).mean(0)@transform.double()).float()
    if bags==3:np.savez(a.out/'individual_bag_weights.npz',**{'bag_'+str(i):(v@transform.double()).float().numpy() for i,v in enumerate(bag_weights)})
    with torch.no_grad():soft=q.double()@folded.double().T;soft-=soft.mean(1,keepdim=True)
    projected,rank=linear_projection((q@transform.T).double(),soft,qt);candidate=(projected.T@transform.double()).float()
    if rank['rank']!=784:raise RuntimeError('Insufficient TRAIN query span')
    with torch.no_grad():recovered=q.double()@candidate.double().T;original=q.double()@folded.double().T;original-=original.mean(1,keepdim=True)
    function_audit=dict(TRAIN_query_rank=rank,max_TRAIN_logit_error=float((recovered-original).abs().max()),TRAIN_argmax_agreement=float((recovered.argmax(1)==original.argmax(1)).double().mean()),student_fit_EVAL_inputs=0)
    groups=dict(TRAIN=tr,all_development=te,teacher_exposed_development=exposed,not_in_teacher_gradient_development=unex)
    def save(role,w):
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before or any(not torch.equal(v,teacher_before[k]) for k,v in teacher.state_dict().items()):raise RuntimeError('Frozen prefix/teacher changed')
        if warm and sha(warm['path'])!=warm['SHA256']:raise RuntimeError('Sealed warm assistant changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**DISCLOSURE);path=a.out/(role+'_checkpoint.pt');torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,w):raise RuntimeError('Strict original-graph CPU reload failed')
        scores={n:evaluate(loaded.shared_head,x,y,tasks,ix) for n,ix in groups.items()};put(a.out/(role+'_development.json'),dict(scores=scores,source_indices=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),**DISCLOSURE));return {n:brief(v) for n,v in scores.items()}
    teacher_scores=save('linear_teacher',folded);save('last_linear_teacher',last_folded);student=save('student',candidate)
    put(a.out/'report.json',dict(passed=True,student=student,linear_teacher=teacher_scores,all_tasks_within_requested_band=within_bands(student['all_development'],stage),
        TRAIN_function_recovery=function_audit,student_sha256=sha(a.out/'student_checkpoint.pt'),strict_CPU_reloaded=True,frozen_upstream=True,new_inference_layers=0,SDK=False,GPU=False,
        status='provisional unless every complete final-student task passes both bounds',seconds=time.monotonic()-started,**DISCLOSURE));put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
