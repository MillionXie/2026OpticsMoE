"""Completed input100 rho pair plus preserved human-stopped control; no forward/test."""
import argparse,hashlib,inspect,math,os,subprocess,textwrap
from pathlib import Path
import numpy as np
import torch
import resisc20 as s
import resisc20_input_study as study
import audit_resisc20 as old
from audit_caltech_repair234 import exited

PAIR=('L4_tile100_rho03','L4_tile100_rho0')
STOP='L4_tile50_rho03'
OUT='cpu_remaining_pair_report'
FAIL='cpu_remaining_pair_report_failure.json'
RUN='resisc20_s17_mainL4_input50input100_rho_controls120_micro8_gpu256_20261014'

def barrier(base):
    assert base.name==RUN
    meta=s.read(base/'metadata.json');state=s.read(base/'status.json');cur=s.read(base/'current_processes.json')
    assert len(meta['sources'])==151 and meta['git_commit']=='d5158a53e4fc1a5a42fff54b25b8c06353f2c1e5'
    assert meta['sources']==study.sources();s.engine.verify_sources(meta['sources'],meta['git_commit'])
    receipt=s.read(base/'human_stop_a100_receipt_20261014.json');claim=s.read(base/'human_stop_a100_claim_20261014.json')
    assert receipt['passed'] and receipt['stopped_actual_exit'] and receipt['a100_project_gpu_release_confirmed']
    assert receipt['claim_sha256']==s.sha(base/'human_stop_a100_claim_20261014.json')
    assert receipt['source_commit']==claim['source_commit']==meta['git_commit'] and receipt['source_count']==151
    assert receipt['stopped_child']['case']==claim['case']==STOP and receipt['stopped_child']['pid']==claim['pid']
    assert receipt['stopped_child']['gpu_uuid']==claim['gpu_uuid']==s.ALLOWED[6] and not claim['resume_authorized']
    assert exited(claim['pid']) and receipt['remaining_gpu_indices']==[2,5] and receipt['last_complete_history_epoch']==26
    for name,identity in receipt['files_preserved_sha256'].items():assert s.sha(base/STOP/name)==identity
    assert len(state['failures'])==1
    f=state['failures'][0];assert f['case']==STOP and f['pid']==claim['pid'] and f['gpu_uuid']==s.ALLOWED[6] and f['exit_code']==-15
    assert all(j['case'] in PAIR and j['gpu_uuid'] in (s.ALLOWED[2],s.ALLOWED[5]) for j in state['completed']+cur['active'])
    ready=exited(meta['pid']) and state['active']==state['pending']==0 and not cur['active'] and not cur['pending'] and len(state['completed'])==2
    if ready:
        assert {j['case'] for j in state['completed']}==set(PAIR) and all(j['exit_code']==0 and exited(j['pid']) for j in state['completed'])
        assert all((base/k/'fit_result.json').exists() for k in PAIR)
    return dict(ready=ready,meta=meta,state=state,stop_receipt=receipt)

def expected_config(key,m):
    cfg=study.config(key);tile,rho=study.CASES[key]
    cfg.update(depth=4,residual_rho=rho,classes=m['classes'],contract=study.contract(4,rho,tile));return cfg

def prefix_hashes(cfg,hist,n):
    orders=[];transforms=[]
    for epoch,h in enumerate(hist,1):
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,20261013]));order=rng.permutation(n).astype('<i8')
        angle=np.deg2rad(rng.uniform(-20,20,n));scale=rng.uniform(.9,1.1,n);theta=np.zeros((n,2,3),np.float32)
        theta[:,0,0]=np.cos(angle)/scale;theta[:,1,1]=theta[:,0,0];theta[:,0,1]=-np.sin(angle)/scale;theta[:,1,0]=np.sin(angle)/scale;theta[:,:,2]=rng.uniform(-5,5,(n,2))*2/150
        mix=hashlib.sha256()
        for b,start in enumerate(range(0,n,16)):
            count=min(16,n-start);draw=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,b,20261013,4]));lam=float(draw.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);permutation=draw.permutation(count)
            mix.update(hashlib.sha256(np.r_[lam,permutation.astype(float)].astype('<f8').tobytes()).hexdigest().encode())
        assert mix.hexdigest()==h['mixup_sha256'];orders.append(hashlib.sha256(order.tobytes()).hexdigest());transforms.append(hashlib.sha256((hashlib.sha256(theta.tobytes()).hexdigest()+mix.hexdigest()).encode()).hexdigest())
    return orders,transforms

def history_metrics(metric,support):
    cm=np.asarray(metric['confusion_matrix']);support=np.asarray(support);assert cm.shape==(20,20) and np.issubdtype(cm.dtype,np.integer) and (cm>=0).all()
    assert np.array_equal(cm.sum(1),support) and np.array_equal(metric['support'],support) and metric['n']==int(support.sum())
    recall=np.diag(cm)/support
    assert np.allclose(metric['recall'],recall,rtol=0,atol=1e-12)
    assert math.isclose(metric['accuracy'],float(np.trace(cm)/support.sum()),rel_tol=0,abs_tol=1e-12)
    assert math.isclose(metric['balanced_accuracy'],float(recall.mean()),rel_tol=0,abs_tol=1e-12)
    assert math.isfinite(metric['balanced_nll']) and metric['balanced_nll']>=0

def scheduler_budget_checks(cfg,hist,last,stop_reason=None):
    sch=last['scheduler'];assert sch['mode']=='min' and sch['threshold_mode']=='rel' and sch['threshold']==1e-4
    assert sch['factor']==.5 and sch['patience']==6 and sch['cooldown']==0 and sch['eps']==1e-8 and sch['min_lrs']==[2e-5]
    rate=cfg['lr'];best=float('inf');bad=0;significant=float('inf');stale=0
    for i,h in enumerate(hist):
        assert math.isclose(h['lr'],rate,rel_tol=0,abs_tol=1e-15)
        if i:assert hist[i-1]['seconds']<cfg['maximum_seconds'] and h['seconds']>=hist[i-1]['seconds']
        value=h['val']['balanced_nll']
        if value<best*(1-1e-4):best=value;bad=0
        else:bad+=1
        if bad>6:
            lowered=max(rate*.5,2e-5)
            if rate-lowered>1e-8:rate=lowered
            bad=0
        if value<significant-cfg['early_min_delta']:significant=value;stale=0
        else:stale+=1
        if i<len(hist)-1:assert not (h['epoch']>=cfg['minimum_epochs'] and stale>=cfg['early_patience'])
    assert sch['last_epoch']==len(hist) and sch['num_bad_epochs']==bad and sch['cooldown_counter']==0 and sch['best']==best
    assert len(sch['_last_lr'])==len(last['optimizer']['param_groups'])==1
    assert math.isclose(sch['_last_lr'][0],rate,rel_tol=0,abs_tol=1e-15) and math.isclose(last['optimizer']['param_groups'][0]['lr'],rate,rel_tol=0,abs_tol=1e-15)
    if stop_reason is not None:
        assert stop_reason in ('maximum_epochs','time_budget','validation_plateau')
        if stop_reason=='maximum_epochs':assert len(hist)==cfg['maximum_epochs']
        if stop_reason=='validation_plateau':assert len(hist)>=cfg['minimum_epochs'] and stale>=cfg['early_patience']
        if stop_reason=='time_budget':assert len(hist)<cfg['maximum_epochs']
    return dict(passed=True,actual_epochs=len(hist),last_history_seconds=hist[-1]['seconds'],maximum_seconds=cfg['maximum_seconds'],epoch_start_budget_boundary_check=True,independent_LR_replay=True,independent_earlystop_replay=True,stop_reason=stop_reason)

def stopped_prefix(base,m,info):
    arm=base/STOP;cfg=s.read(arm/'config.json');meta=s.read(arm/'metadata.json');hist=s.read(arm/'history.json');grads=s.read(arm/'gradients.json')
    assert cfg==expected_config(STOP,m) and meta['sources']==info['meta']['sources'] and exited(meta['pid'])
    best=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    n=sum(m['supports']['train']);assert n==11198 and math.ceil(n/16)==700 and n%16==14
    assert len(hist)==len(grads)==last['epoch']==info['stop_receipt']['last_complete_history_epoch']==26
    assert [h['epoch'] for h in hist]==[g['epoch'] for g in grads]==list(range(1,27))
    assert best['sources']==last['sources']==meta['sources'] and best['config']==last['config']==cfg
    assert best['data_sha256']==last['data_sha256']==m['cache_sha256']
    chosen=min(hist,key=lambda h:h['val']['balanced_nll']);assert chosen['epoch']==best['epoch']==info['stop_receipt']['best_saved_epoch'] and chosen['val']==best['validation']
    assert best['selection_split']=='val' and best['selected_state_kind']=='ema'
    s.engine.seed(cfg['seed']);model=s.r.CorrectedMoE(4,.3);initial=s.digest(model.state_dict());assert initial==last['initial_parameter_sha256']
    shapes={k:tuple(v.shape) for k,v in model.named_parameters()};assert sum(v.numel() for v in model.parameters())==908552 and len(shapes)==11
    for state in (best['model'],last['model'],last['ema']):
        assert set(state)==set(shapes) and all(tuple(v.shape)==shapes[k] and torch.isfinite(v).all() for k,v in state.items())
    assert all(set(g['norms'])==set(shapes) and all(math.isfinite(v) and v>0 for v in g['norms'].values()) for g in grads)
    assert last['scheduler']['last_epoch']==26 and len(last['optimizer']['state'])==11
    assert all(int(v['step'])==700*26 for v in last['optimizer']['state'].values())
    orders,transforms=prefix_hashes(cfg,hist,n);assert orders==last['orders'] and transforms==last['transforms']
    for h in hist:
        history_metrics(h['val'],m['supports']['val'])
        if 'train' in h:history_metrics(h['train'],m['supports']['train'])
    assert s.read(arm/'smoke.json')['passed'] and not (arm/'failure.json').exists()
    schedule=scheduler_budget_checks(cfg,hist,last)
    with np.load(Path(cfg['data']),allow_pickle=False) as z:labels=z['val_labels'].copy()
    old.predictions(arm/'val_predictions.csv',best['validation'],m['split_ids']['val'],labels)
    return dict(passed=True,case=STOP,epochs=26,initial_parameter_sha256=initial,orders=orders,transforms=transforms,config=cfg,selected_saved_epoch=best['epoch'],validation_from_saved_history=chosen['val'],validation_prediction_ID_label_audit=True,validation_predictions_sha256=s.sha(arm/'val_predictions.csv'),scheduler_budget_checks=schedule,finalized_train_val_prediction_ID_audit=False,selected_train_metrics_available=False,eligible=False,complete_budget=False,scope='Human-stopped immutable prefix only; independent config/source/init/shape/Adam/order/affineMix/grad/saved CM checks, no finalized selected-train predictions or full-arm performance admission.')

def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==a.base/OUT and not a.out.exists() and not (a.base/FAIL).exists()
    info=barrier(a.base);assert info['ready'],'Both remaining fits and supervisor must actually exit before saved science review'
    study.prerequisites();m=s.manifest(a.data);assert a.data==Path(study.config(PAIR[0])['data'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.ROOT,text=True).strip();relative=Path(__file__).relative_to(s.ROOT).as_posix()
    assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+relative],cwd=s.ROOT)).hexdigest()==s.sha(__file__)
    source=textwrap.dedent(inspect.getsource(old.verify));assert source.count("f['epochs_completed']<=30")==1
    ns=dict(old.__dict__);exec(compile(source.replace("f['epochs_completed']<=30","f['epochs_completed']<=cfg['maximum_epochs']"),str(__file__)+':sealed_independent_verify','exec'),ns)
    fits={};schedule_checks={}
    for key in PAIR:
        f=ns['verify'](a.base,key,a.data,m);assert f['config']==expected_config(key,m) and f['sources']==info['meta']['sources']
        assert f['rho']==study.CASES[key][1] and f['config']['maximum_epochs']==120 and f['config']['maximum_seconds']==7200 and f['config']['microbatch']==8
        fits[key]=f
        history=s.read(a.base/key/'history.json');last=torch.load(a.base/key/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        schedule_checks[key]=scheduler_budget_checks(f['config'],history,last,f['stop_reason'])
    partial=stopped_prefix(a.base,m,info);assert len({f['initial_parameter_sha256'] for f in fits.values()}|{partial['initial_parameter_sha256']})==1
    normalized=[]
    for f in fits.values():
        cfg=dict(f['config'])
        for k in ('study_case','residual_rho','contract'):cfg.pop(k)
        normalized.append(cfg)
    assert normalized[0]==normalized[1]
    common=min(f['epochs_completed'] for f in fits.values());assert len({tuple(f['orders'][:common]) for f in fits.values()})==len({tuple(f['transforms'][:common]) for f in fits.values()})==1
    prefix=min(26,common);assert all(f['orders'][:prefix]==partial['orders'][:prefix] and f['transforms'][:prefix]==partial['transforms'][:prefix] for f in fits.values())
    gates={k:s.read(a.base/k/'validation_gate.json') for k in PAIR};a.out.mkdir(exist_ok=False)
    s.save(a.out/'audit.json',dict(passed=True,fits=fits,gates=gates,scheduler_budget_checks=schedule_checks,all_pair_eligible=all(g['passed'] for g in gates.values()),human_stopped_partial=partial,common_pair_epochs=common,common_control_prefix_epochs=prefix,stop_claim_sha256=s.sha(a.base/'human_stop_a100_claim_20261014.json'),stop_receipt_sha256=s.sha(a.base/'human_stop_a100_receipt_20261014.json'),training_source_commit=info['meta']['git_commit'],sources=info['meta']['sources'],audit_source_sha256=s.sha(__file__),audit_commit=commit,train_samples=11198,val_samples=1399,optimizer_steps_per_epoch=700,tail_batch=14,cpu_models_constructed_for_shape_seed=3,model_forward_calls=0,test_arrays_read=0,test_evaluations=0,formal_authorized_by_this_receipt=False,scope='Complete input100 rho pair at original bounded budgets; stopped A only independently checked immutable26 prefix, no full-budget input50-vs100 claim or automatic formal/native/test admission'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','out','data'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        marker=a.base/FAIL
        if not marker.exists():s.save(marker,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
