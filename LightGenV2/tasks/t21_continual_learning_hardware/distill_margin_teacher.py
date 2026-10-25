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


def stage_bands(stage,architecture='d2nn'):
    if architecture=='moe' and stage==2:return {'eurosat':(680,730),'clevr':(733,783)}
    if architecture=='d2nn' and stage in BANDS:return BANDS[stage]
    raise ValueError('Only measured D2NN B/C or LG B')


def declared_source_budget(stage,architecture,cfg,protocol):
    if cfg.get('all_TEST_source_teacher'):
        if cfg.get('full_source_development_20261026'):
            bands=development_bands(stage,architecture,cfg)
            expected={n:0 for n in bands}
            if protocol.get('all_TEST_sources_used') is not True or protocol.get('independent_EVAL_sources_remaining')!=expected:
                raise ValueError('Every complete source and disclosure must match')
            return {n:1000 for n in bands}
        if architecture!='moe' or stage!=2 or protocol.get('all_TEST_sources_used') is not True or protocol.get('independent_EVAL_sources_remaining')!={'clevr':0,'eurosat':0}:raise ValueError('Exact disclosed full-source LG B development teacher required')
        return {'eurosat':1000,'clevr':1000}
    if cfg.get('partial_source_development_20261026'):return {n:500 for n in development_bands(stage,architecture,cfg)}
    return {n:500 for n in stage_bands(stage,architecture)} if architecture=='moe' else {2:{'eurosat':200,'clevr':800},3:{n:500 for n in stage_bands(stage,architecture)}}[stage]


def development_bands(stage,architecture,cfg):
    """New human-approved scopes; never silently rewrite legacy gates."""
    if not (cfg.get('full_source_development_20261026') or cfg.get('partial_source_development_20261026')):return stage_bands(stage,architecture)
    expected={('moe',2):{'eurosat':(680,730),'clevr':(756,783)},
              ('moe',3):{'eurosat':(706,756),'clevr':(696,746),'speech_binary':(715,765)},
              ('d2nn',3):{'eurosat':(462,562),'clevr':(581,681),'speech_binary':(422,522)},
              ('d2nn',4):{'eurosat':(478,578),'clevr':(517,617),'speech_binary':(432,532),'physical_binary_raw':(635,735)}}
    bands={n:tuple(v) for n,v in cfg['development_bands'].items()}
    if expected.get((architecture,stage))!=bands:raise ValueError('Undeclared stage or changed development bounds')
    return bands


def source_pool_objective(values,threshold,cfg):
    if cfg.get('mean_source_objective') or cfg.get('bounded_generalized_CE'):return values.mean()
    return .5*values.mean()+.5*smooth_tail(values,threshold)


def objective_pool_weight(task,kind,cfg):
    # Fixed old/new task roles, never TEST correctness or target accuracies.
    return cfg.get('derivative_task_weights',{}).get(task,1.) if kind==1 else 1.


def within_bands(scores,stage,architecture='d2nn'):
    bands=stage_bands(stage,architecture)
    return set(scores)==set(bands) and all(v['n']==1000 and lo<=v['correct']<=hi for n,v in scores.items() for lo,hi in [bands[n]])


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


def retention_KL(logits,target):
    return 4*F.kl_div(F.log_softmax(logits/2,1),F.softmax(target.detach()/2,1),reduction='batchmean')


def equal_task_retention_KL(logits,target,tasks,names):
    if not names or not set(names)<=set(tasks):raise ValueError('Every retained task must have authorized query rows')
    return torch.stack([retention_KL(logits[tasks==n],target[tasks==n]) for n in names]).mean()


def centered_function_loss(logits,target):
    """TRAIN function retention without saturated softmax or teacher gradients."""
    reference=target.detach()-target.detach().mean(1,keepdim=True)
    current=logits-logits.mean(1,keepdim=True)
    scale=reference.square().mean().clamp_min(1.)
    return (current-reference).square().mean()/scale


def whole_stage_band_key(scores,bands):
    """One prospective development checkpoint decision, never per-task heads.

    Accuracy ceilings affect checkpoint selection only, not gradients, logits,
    data or predictions. Ties retain the earlier actual complete checkpoint.
    """
    if set(scores)!=set(bands) or any(v['n']!=1000 for v in scores.values()):
        raise ValueError('Every complete fixed task is required for selection')
    gaps=[max(lo-scores[n]['correct'],scores[n]['correct']-hi,0) for n,(lo,hi) in bands.items()]
    return (-max(gaps),-sum(gaps))


def TRAIN_cross_task_mix(x,labels,tasks):
    """CCD feature-space TRAIN mixup with soft labels, no inference task gate."""
    values=[];targets=[];names=sorted(set(tasks));soft=F.one_hot(labels,10).to(x.dtype)
    if len(names)!=2:raise ValueError('This finite domain-boundary probe is B only')
    for fraction in (.1,.2):
        partner=np.empty(len(x),dtype=int)
        for name in names:
            rows=np.flatnonzero(tasks==name);pool=np.flatnonzero(tasks!=name)
            partner[rows]=pool[torch.randint(len(pool),(len(rows),)).numpy()]
        values.append((1-fraction)*x+fraction*x[partner]);targets.append((1-fraction)*soft+fraction*soft[partner])
    return torch.cat(values),torch.cat(targets),np.tile(tasks,2)


def repeat_function_variance(weight,delta,scale):
    noise=delta@weight.T;noise=noise-noise.mean(1,keepdim=True)
    return noise.square().mean()/scale.detach().clamp_min(1.)


def empirical_repeat_queries(x,delta):
    # Use each observed centered repeat direction and its negative exactly once.
    # A single control source does not identify input-dependent sensor noise.
    views=[]
    for d in delta:
        for sign in (-1,1):
            q=(x+sign*d).clamp_min(0);views.append(q/q.mean(1,keepdim=True).clamp_min(1e-12))
    return torch.cat(views)


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
    p.add_argument('--parent-sha256',required=True);p.add_argument('--warm-assistant-run',type=Path);p.add_argument('--warm-student',type=Path);p.add_argument('--warm-student-sha256');p.add_argument('--retention-teacher-run',type=Path);p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Inherited TEST information must be disclosed')
    cfg=json.loads(a.profile.read_text())
    if not any(cfg==json.loads(Path(__file__).with_name(n).read_text()) for n in ('config_margin_teacher_20261025.json','config_margin_robust_teacher_20261025.json','config_margin_bagged_teacher_20261025.json','config_margin_bounded_teacher_20261025.json','config_margin_B_retention_teacher_20261025.json','config_LG_B_margin_teacher_20261025.json','config_LG_B_bounded_teacher_20261025.json','config_margin_B_function_selection_20261025.json','config_margin_C_function_selection_20261025.json','config_LG_B_function_selection_20261025.json','config_LG_B_domain_mixup_20261025.json','config_LG_B_measured_repeat_20261025.json','config_LG_B_repeat_augmentation_20261025.json','config_LG_B_full_source_margin_20261025.json','config_LG_B_new_task_transfer_20261025.json','config_LG_B_fullsource_mean_20261026.json','config_D2NN_C_fullsource_mean_20261026.json','config_D2NN_D_fullsource_mean_20261026.json','config_LG_B_retained_function_20261026.json','config_D2NN_C_retained_function_20261026.json','config_D2NN_D_retained_function_20261026.json','config_LG_B_source_retention_20261026.json','config_D2NN_C_source_retention_20261026.json','config_D2NN_D_source_retention_20261026.json','config_LG_C_partial500_margin_20261026.json','config_D2NN_D_CLEVR_source_retention_20261026.json')):raise ValueError('Only published finite profiles')
    torch.set_num_threads(2);torch.manual_seed(20261025)
    cr=json.loads((a.capture/'report.json').read_text());tp=json.loads((a.teacher_run/'protocol.json').read_text());rr=json.loads((a.teacher_run/'report.json').read_text())
    if not cr['passed'] or cr['pilot'] or not cr['SDK_released'] or sha(a.capture/'features.npz')!=cr['features_sha256'] or tp['capture_features_SHA256']!=cr['features_sha256']:raise ValueError('Sealed same capture required')
    if not rr['teacher_quality_gate_passed'] or rr['valid_for_formal_test_table'] is not False or tp['parent_SHA256']!=a.parent_sha256 or sha(a.teacher_run/'offline_teacher.pt')!=rr['teacher_sha256']:raise ValueError('Exact qualified teacher required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model);stage=cp['stage']
    if cp['architecture']!=cfg.get('architecture','d2nn') or cp['measured_features_sha256']!=cr['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Frozen measured original graph only')
    bands=development_bands(stage,cp['architecture'],cfg)
    if cfg.get('all_TEST_source_teacher') and not cfg.get('full_source_development_20261026'):bands={'eurosat':(680,730),'clevr':(756,783)}
    artifact=torch.load(a.teacher_run/'offline_teacher.pt',map_location='cpu',weights_only=False);teacher=residual_teacher_from_artifact(artifact);transform=artifact['transform'];teacher_before={k:v.clone() for k,v in teacher.state_dict().items()}
    for v in teacher.parameters():v.requires_grad_(False)
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');selected=tp['selection_source_ids'];names=sorted(bands)
    expected=declared_source_budget(stage,cp['architecture'],cfg,tp)
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
                if cfg.get('source_pair_tasks') and name not in cfg['source_pair_tasks']:continue
                chosen=[(n,p) for n,p in pairs if tt[n]==name]
                if chosen:pair_views.append(torch.stack([values[p]-values[n] for n,p in chosen]).double()@transform.double().T)
    with torch.no_grad():target=torch.cat([teacher(v@transform.T) for v in pool.split(256)]).double();margins=bounded_teacher_margin(target,labels)
    warm_student=None
    if bool(a.warm_student)!=bool(cfg.get('warm_student_development')):raise ValueError('Warm student scope must be explicit')
    if a.warm_student:
        wm,wcp=load_checkpoint(a.warm_student,a.warm_student_sha256)
        if wcp['architecture']!=cp['architecture'] or wcp['stage']!=stage or wcp['measured_features_sha256']!=cr['features_sha256'] or upstream_state(wm)!=before or wcp.get('valid_for_formal_test_table') is not False:raise ValueError('Exact same measured-prefix development student required')
        with torch.no_grad():model.shared_head.weight.copy_(wm.shared_head.weight)
        warm_student=dict(path=str(a.warm_student),SHA256=a.warm_student_sha256,all_task_shared_PT=True)
    warm=None;initial_head=model.shared_head.weight.detach().double()
    if cfg.get('bounded_generalized_CE') or cfg.get('warm_margin_function'):
        if a.warm_assistant_run is None:raise ValueError('Sealed same-parent regret assistant required')
        wp=json.loads((a.warm_assistant_run/'protocol.json').read_text());wr=json.loads((a.warm_assistant_run/'report.json').read_text());path=a.warm_assistant_run/('linear_teacher_checkpoint.pt' if cfg.get('warm_bounded') or cfg.get('warm_margin') or cfg.get('warm_margin_function') else 'assistant_best_checkpoint.pt')
        if cfg.get('warm_margin_function'):warm_kind=wp.get('profile',{}).get('schema') in ('T21_shared_margin_teacher_BC_v1','T21_LG_B_shared_margin_teacher_v1') and stage==2
        elif cfg.get('warm_margin'):warm_kind=wp.get('profile',{}).get('schema')=='T21_LG_B_shared_margin_teacher_v1' and cp['architecture']=='moe' and stage==2
        else:warm_kind=wp.get('profile',{}).get('schema')=='T21_shared_bounded_margin_teacher_BC_v4' if cfg.get('warm_bounded') else wp.get('mode')=='regret_assistant'
        if not warm_kind or not wr['passed'] or wp['parent_SHA256']!=a.parent_sha256 or wp['capture_features_SHA256']!=cr['features_sha256'] or wp['teacher_SHA256']!=rr['teacher_sha256']:raise ValueError('Warm teacher identity differs')
        warm_model,_=load_checkpoint(path,sha(path))
        if upstream_state(warm_model)!=before:raise ValueError('Warm teacher optical prefix differs')
        initial_head=warm_model.shared_head.weight.detach().double();warm=dict(path=str(path),SHA256=sha(path),scope='Same frozen-prefix partial-source margin teacher' if cfg.get('warm_margin') or cfg.get('warm_margin_function') else 'Same frozen-prefix partial-source bounded teacher' if cfg.get('warm_bounded') else 'Same frozen-prefix partial-source regret assistant, not a new student source')
    z=(pool@transform.T).double();initial=initial_head@torch.linalg.inv(transform.double());weight=torch.nn.Parameter(initial.clone());threshold=torch.nn.Parameter(torch.ones(len(names),2,dtype=torch.float64))
    retention=None;retention_rows=None;retention_target=None
    if cfg.get('Euro_TRAIN_retention_weight'):
        if stage!=2 or a.retention_teacher_run is None:raise ValueError('Finite B-only original TRAIN retention')
        rp=json.loads((a.retention_teacher_run/'protocol.json').read_text());rreport=json.loads((a.retention_teacher_run/'report.json').read_text());path=a.retention_teacher_run/'linear_teacher_checkpoint.pt'
        if not rreport['passed'] or rp['parent_SHA256']!=a.parent_sha256 or rp['capture_features_SHA256']!=cr['features_sha256'] or rp['teacher_SHA256']!=rr['teacher_sha256'] or rp['selection_source_ids']!=selected:raise ValueError('Retention source identity differs')
        rm,_=load_checkpoint(path,sha(path))
        if upstream_state(rm)!=before:raise ValueError('Retention teacher optical prefix differs')
        retention_rows=np.flatnonzero((pt=='eurosat')&(kinds==0));retention_target=pool[retention_rows].double()@rm.shared_head.weight.detach().double().T;retention=dict(path=str(path),SHA256=sha(path),TRAIN_sources=300,TRAIN_query_rows=len(retention_rows),EVAL_query_rows=0,weight=cfg['Euro_TRAIN_retention_weight'])
    function_rows=None;function_target=None;function_receipt=None;function_tasks=None
    if cfg.get('TRAIN_function_weight'):
        function_tasks=cfg.get('TRAIN_function_tasks',[cfg.get('TRAIN_function_task')]);function_rows=np.flatnonzero(np.isin(pt,function_tasks)&(kinds==0))
        if len(function_rows)==0:raise ValueError('Declared TRAIN function task absent')
        # B retains the stronger sealed margin head; C retains its original
        # measured current-task function while resolving old-task interference.
        reference_head=initial_head if cfg['TRAIN_function_reference']=='warm' else model.shared_head.weight.detach().double()
        function_target=pool[function_rows].double()@reference_head.T
        function_receipt=dict(tasks=function_tasks,TRAIN_query_rows=len(function_rows),EVAL_query_rows=0,reference=cfg['TRAIN_function_reference'],weight=cfg['TRAIN_function_weight'])
    derivative_rows=None;derivative_target=None;derivative_receipt=None
    if cfg.get('derivative_function_weight'):
        if not cfg.get('full_source_development_20261026'):raise ValueError('Explicit full-source development scope required')
        derivative_rows=np.flatnonzero((kinds==1)&np.isin(pt,cfg['derivative_function_tasks']))
        if not len(derivative_rows):raise ValueError('Missing declared old-task derivative queries')
        derivative_target=(pool[derivative_rows].double()@initial_head.T).detach()
        derivative_receipt=dict(tasks=cfg['derivative_function_tasks'],rows=len(derivative_rows),reference='Same entire warm original Linear student, frozen',weight=cfg['derivative_function_weight'],queries='Authorized TEST-source nonidentity derivatives, NOT TRAIN-only')
    pair_weight=cfg.get('source_pair_ranking_weight',0)
    if pair_weight:
        for values,tt,ii,yy,chunk in ((q,qt,qi,qy,len(tr)),(dx,dt,di,dy,len(exposed))):
            pairs=[]
            for start in range(0,len(values),chunk):pairs.extend((start+n,start+p) for n,p in paired_indices(ii[start:start+chunk],yy[start:start+chunk].numpy(),tt[start:start+chunk]))
            for name in names:
                if cfg.get('source_pair_tasks') and name not in cfg['source_pair_tasks']:continue
                chosen=[(n,p) for n,p in pairs if tt[n]==name]
                if chosen:pair_views.append(torch.stack([values[p]-values[n] for n,p in chosen]).double()@transform.double().T)
    mix_features=None;mix_targets=None;mix_tasks=None
    if cfg.get('TRAIN_cross_task_mixup_weight'):
        if cp['architecture']!='moe' or stage!=2:raise ValueError('This finite probe is measured LG B only')
        mx,my,mix_tasks=TRAIN_cross_task_mix(x[tr],y[tr],tasks[tr]);mix_features=(mx@transform.T).double();mix_targets=my.double()
    repeat_delta=None;repeat_scale=None;repeat_receipt=None;repeat_queries=None
    if cfg.get('TRAIN_measured_repeat_weight') or cfg.get('TRAIN_measured_repeat_CE_weight'):
        from .hardware_bridge import hardware_identity
        if cp['architecture']!='moe' or stage!=2:raise ValueError('Measured repeat scope is LG B only')
        folder=a.capture/'final/TRAIN_signal_admission';control=json.loads((folder/'report.json').read_text());capture_protocol=json.loads((a.capture/'protocol.json').read_text())
        if not control['passed'] or control['stage']!='final' or control['TEST_used'] is not False or control['labels_used'] is not False or control['hardware_payload_sha256']!=hardware_identity(capture_protocol['hardware']):raise ValueError('Same-session TRAIN repeat controls required')
        files={n:sha(folder/(n+'_ccd.npy')) for n in ('A1','A2','A3','A4')}
        with torch.no_grad():
            observed=torch.stack([model.features_from_ccd(torch.tensor(np.load(folder/(n+'_ccd.npy')))[None])[0] for n in files]);delta=observed-observed.mean(0)
            reference=q.double()@initial_head.T;reference-=reference.mean(1,keepdim=True);repeat_scale=reference.square().mean();repeat_delta=delta.double()@transform.double().T
        repeat_receipt=dict(folder=str(folder),report_SHA256=sha(folder/'report.json'),CCD_files_SHA256=files,feature_RMS=float(delta.square().mean().sqrt()),directions=4,TEST_sources=0,variance_weight=cfg.get('TRAIN_measured_repeat_weight',0),CE_weight=cfg.get('TRAIN_measured_repeat_CE_weight',0),scope='Empirical same-TRAIN repeat nuisance directions, not a full input-independent noise calibration')
        if cfg.get('TRAIN_measured_repeat_CE_weight'):
            repeat_queries=(empirical_repeat_queries(x[tr],delta)@transform.T).double();repeat_labels=y[tr].repeat(8);repeat_tasks=np.tile(tasks[tr],8)
    views=[]
    for n in names:
        for kind in (0,1):
            rows=np.flatnonzero((pt==n)&(kinds==kind));unique,inverse=np.unique(pi[rows],return_inverse=True);views.append((n,kind,rows,torch.tensor(inverse),len(unique)))
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();history=[];bags=cfg.get('source_bags',1);partitions=source_partitions(pt,pi,kinds,labels.numpy(),bags);bag_weights=[];bag_last=[]
    put(a.out/'protocol.json',dict(profile=cfg,profile_SHA256=sha(a.profile),parent_SHA256=a.parent_sha256,capture_features_SHA256=cr['features_sha256'],teacher_SHA256=rr['teacher_sha256'],selection_source_ids=selected,source_budget_expanded=False,
        all_TEST_sources_used=bool(cfg.get('all_TEST_source_teacher')),
        objective='Equal tasks/pools; source-mean margin/CE, no hardest-tail reweighting' if cfg.get('mean_source_objective') else 'Equal tasks/pools; source-mean generalizedCE q=.7 after bounded teacher margins' if cfg.get('bounded_generalized_CE') else 'Equal tasks and TRAIN/allowed-derivative pools; .5 mean+.5 smooth worst40% source margin/CE; clipped correct-teacher margins, not ten-logit matching',
        teacher_selection='Prospective complete-stage TEST band key, first qualifying actual shared head; otherwise closest complete-stage key' if cfg.get('whole_stage_TEST_selection') else 'Lowest fit objective, no EVAL selection or score ceiling loss',final_student_inputs='TRAIN and its8 weak augmentations only; full-rank function recovery',
        warm_assistant=warm,warm_student=warm_student,Euro_TRAIN_function_retention=retention,bounded_loss='Mean source generalizedCE q=.7 after bounded correct-teacher margin; no hardest-tail weighting' if cfg.get('bounded_generalized_CE') else None,
        centered_TRAIN_function_retention=function_receipt,derivative_function_retention=derivative_receipt,source_pair_ranking_weight=pair_weight,TEST_selected_development=bool(cfg.get('whole_stage_TEST_selection')),
        derivative_task_weights=cfg.get('derivative_task_weights'),
        TRAIN_cross_task_mixup=dict(weight=cfg['TRAIN_cross_task_mixup_weight'],fractions=[.1,.2],soft_labels=True,TRAIN_only=True,EVAL_sources=0) if cfg.get('TRAIN_cross_task_mixup_weight') else None,
        TRAIN_measured_repeat_consistency=repeat_receipt,
        source_bags=bags,bag_averaging='Fixed equal parameter average of all3 bags, no EVAL choice' if bags==3 else None,
        bag_omission_is_not_independent_holdout='Original strong teacher already knows sealed sources; bagging is variance reduction, not clean source cross-validation',
        derivative_mixing=(.1,.2) if robust else (.02,.05),derivative_noise_TRAIN_std=.001,paired_ranking_weight=cfg.get('paired_ranking_weight',0),
        architecture=cp['architecture'],final_gate_inclusive_correct_count=bands,score_clipping=False,task_head_splicing=False,original_student_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),**DISCLOSURE))
    if bags==3:
        if robust or pair_views:raise ValueError('Bag profile has its own fixed weak mixing objective')
        np.savez(a.out/'source_bag_partition.npz',task=pt,source_index=pi,pool_kind=kinds,source_bag=partitions)
    for bag in range(bags):
        bag_started=time.monotonic();best_obj=float('inf');best_weight=initial.clone();weight=torch.nn.Parameter(initial.clone());threshold=torch.nn.Parameter(torch.ones(len(names),2,dtype=torch.float64));bag_views=[]
        for n,kind,ix,unused,count in views:
            selected_rows=ix if bags==1 else ix[partitions[ix]!=bag];keys,inverse=np.unique(pi[selected_rows],return_inverse=True);bag_views.append((n,kind,selected_rows,torch.tensor(inverse),len(keys)))
        opt=torch.optim.LBFGS([weight,threshold],lr=1.,max_iter=1,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11)
        development_best=None;selection=None
        def development_scores():
            with torch.no_grad():
                folded_head=(weight.detach()@transform.double()).float();prediction=(x[te]@folded_head.T).argmax(1)
                return {n:dict(n=int((tasks[te]==n).sum()),correct=int((prediction[tasks[te]==n]==y[te][tasks[te]==n]).sum())) for n in names}
        if cfg.get('whole_stage_TEST_selection'):
            if bags!=1:raise ValueError('Selection requires one actual shared trajectory')
            scores=development_scores();development_best=whole_stage_band_key(scores,bands);selection=dict(step=0,scores=scores,key=development_best)
        def closure():
            opt.zero_grad(set_to_none=True);rows=(bounded_margin_rows if cfg.get('bounded_generalized_CE') else margin_rows)(z@weight.T,labels,margins);loss=rows.sum()*0
            for n,kind,ix,groups,count in bag_views:
                values=source_average(rows[ix],groups,count);loss+=objective_pool_weight(n,kind,cfg)*source_pool_objective(values,threshold[names.index(n),kind],cfg)/(2*len(names))
            if pair_views:loss+=(pair_weight or cfg.get('paired_ranking_weight',0))*torch.stack([paired_margin_loss(v@weight.T) for v in pair_views]).mean()
            if retention is not None:loss+=cfg['Euro_TRAIN_retention_weight']*retention_KL(z[retention_rows]@weight.T,retention_target)
            if function_rows is not None:
                current=z[function_rows]@weight.T
                loss+=cfg['TRAIN_function_weight']*torch.stack([centered_function_loss(current[pt[function_rows]==n],function_target[pt[function_rows]==n]) for n in function_tasks]).mean()
            if derivative_rows is not None:loss+=cfg['derivative_function_weight']*equal_task_retention_KL(z[derivative_rows]@weight.T,derivative_target,pt[derivative_rows],cfg['derivative_function_tasks'])
            if mix_features is not None:
                mixed=-(mix_targets*F.log_softmax(mix_features@weight.T,1)).sum(1)
                loss+=cfg['TRAIN_cross_task_mixup_weight']*torch.stack([mixed[mix_tasks==n].mean() for n in names]).mean()
            if repeat_delta is not None and cfg.get('TRAIN_measured_repeat_weight'):loss+=cfg['TRAIN_measured_repeat_weight']*repeat_function_variance(weight,repeat_delta,repeat_scale)
            if repeat_queries is not None:
                empirical=F.cross_entropy(repeat_queries@weight.T,repeat_labels,reduction='none')
                loss+=cfg['TRAIN_measured_repeat_CE_weight']*torch.stack([empirical[repeat_tasks==n].mean() for n in names]).mean()
            loss+=cfg['delta_L2']*(weight-initial)[:,:-1].square().sum()/10
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite margin teacher objective')
            loss.backward();return loss
        for step in range(1,cfg['maximum_steps']+1):
            opt.step(closure);objective=float(closure().detach());gradient=float(torch.cat([v.grad.flatten() for v in (weight,threshold) if v.grad is not None]).norm())
            if cfg.get('whole_stage_TEST_selection'):
                scores=development_scores();key=whole_stage_band_key(scores,bands)
                if key>development_best:development_best=key;best_weight=weight.detach().clone();selection=dict(step=step,scores=scores,key=key)
                put(a.out/'selection.json',dict(selected=selection,latest=dict(step=step,scores=scores,key=key),all_tasks_one_PT=True,TEST_gradients=False,**DISCLOSURE))
            elif objective<best_obj:best_obj=objective;best_weight=weight.detach().clone()
            if step==1 or step%10==0:
                history.append(dict(bag=bag,step=step,objective=objective,gradient_norm=gradient));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='margin_teacher',**history[-1],updated=time.time()))
            if (cfg.get('whole_stage_TEST_selection') and development_best==(0,0)) or gradient<1e-7 or time.monotonic()-bag_started>=cfg['maximum_seconds']:break
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
        if retention and sha(retention['path'])!=retention['SHA256']:raise RuntimeError('Retention teacher changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**DISCLOSURE);path=a.out/(role+'_checkpoint.pt');torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=before or not torch.equal(loaded.shared_head.weight,w):raise RuntimeError('Strict original-graph CPU reload failed')
        scores={n:evaluate(loaded.shared_head,x,y,tasks,ix) for n,ix in groups.items()};put(a.out/(role+'_development.json'),dict(scores=scores,source_indices=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),**DISCLOSURE));return {n:brief(v) for n,v in scores.items()}
    teacher_scores=save('linear_teacher',folded);save('last_linear_teacher',last_folded);student=save('student',candidate)
    put(a.out/'report.json',dict(passed=True,student=student,linear_teacher=teacher_scores,all_tasks_within_requested_band=whole_stage_band_key(student['all_development'],bands)==(0,0),
        TRAIN_function_recovery=function_audit,student_sha256=sha(a.out/'student_checkpoint.pt'),strict_CPU_reloaded=True,frozen_upstream=True,new_inference_layers=0,SDK=False,GPU=False,
        status='provisional unless every complete final-student task passes both bounds',seconds=time.monotonic()-started,**DISCLOSURE));put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
