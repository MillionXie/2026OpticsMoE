"""Retained depth4/6, current TRAIN-calibrated public teacher temperatures."""
import types,hashlib
import caltech20_rich_L2_retained_soup_finetune as baseline
import caltech20_rich_electronic_midpoint46 as closure
import caltech20_rich_pair50_teacher as teacher_source
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','closure','original','old'):globals()[n]=v
original=baseline.original
FILES=('caltech20_rich_currentKD46.py','caltech20_rich_currentKD46_profile.json','check_caltech20_rich_currentKD46.py','audit_caltech20_rich_currentKD46.py','CALTECH20_RICH_CURRENTKD46_PROTOCOL_20261018.md')
def config():
    c=baseline.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_currentKD46_CPU_20261018';ACTIVE_DEPTH=4
def sources():
    z=closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','mixed','campaign','feature_loss','target_data','initialize','gradient_kernel','gate_kernel','complete_gate_contract','gate','fit_kernel'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=baseline.select;select_kernel=types.FunctionType(f.__code__,globals(),'select_kernel',f.__defaults__,f.__closure__)
def select(key):
    global ACTIVE_DEPTH
    ACTIVE_DEPTH=config()['cases'][key]['depth'];select_kernel(key)
@lru_cache(None)
def incumbent(depth):
    assert depth in (4,6)
    if depth==4:return baseline.incumbent(depth)
    module=closure.original;f=module.OUT/'L6';rec=s.read(module.OUT/'test_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(module.OUT/'test_science/audit.json');fit=s.read(f/'fit_result.json');path=f/'candidate.pt';assert fit['candidate_SHA256']==s.sha(path)=='4f9d5e1b6a9d0a3a7c15484236d53e82ad341d9c347ee09857629e26e2aa92ae';pt=torch.load(path,map_location='cpu',weights_only=False);pred={r:load_predictions(f/('candidate_'+r+'.npz')) for r in ('train','val','test')};q=dict(path=str(path),sha256=s.sha(path),module='caltech20_rich_dualview_hosthash_repair',class_name='Alpha40MoE',alpha_low=None,joint_budget_epochs=None,selected_optical_epoch=80,parameter_average=True,optical_phase_budget=80,parent_extra_electronic_epochs=8,source450_science_SHA256=rec['audit_sha256']);return q,pt,pred
f=baseline.prototype_data.__wrapped__;prototype_data=lru_cache(None)(types.FunctionType(f.__code__,globals(),'same_TRAIN_prototypes_current_parent_calibration',f.__defaults__,f.__closure__))
@lru_cache(None)
def teacher_data(depth):
    assert depth in (4,6);t=teacher_source;rec=s.read(t.OUT/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(t.OUT/'science/audit.json');fit=s.read(t.OUT/'fit_result.json');path=t.OUT/'selected_head.pt';assert fit['head_sha256']==s.sha(path);pt=torch.load(path,map_location='cpu',weights_only=False)
    with np.load(t.OUT/'train_features.npz',allow_pickle=False) as z:x,y,ids=[z[n].copy() for n in ('features','labels','ids')]
    assert ids.tolist()==base.manifest()['split_ids']['train'];x=((x-pt['mean'])/pt['std']).astype(np.float32).astype(float);logits=x@pt['model']['weight'].double().numpy().T+pt['model']['bias'].double().numpy();p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True);valid=((logits.argmax(1)==y)&(p.max(1)>=.5)).astype(np.float32);tau=config()['teacher_temperature_by_depth'][str(depth)];z=logits/tau;target=np.exp(z-z.max(1,keepdims=True));target/=target.sum(1,keepdims=True);return target.astype(np.float32),valid,dict(temperature=tau,teacher_science_SHA256=rec['audit_sha256'],head_SHA256=s.sha(path))
def teacher_targets():return teacher_data(ACTIVE_DEPTH)[:2]
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'same_MixUp_mass_new_calibrated_targets',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
f=baseline.gradients;gradient_record_kernel=types.FunctionType(f.__code__,globals(),'same_full_RMS_prototype_and_phasecode_gradients',f.__defaults__,f.__closure__)
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    result=gradient_record_kernel(model,x,y,q,mass,mask,weights,cfg,micro);model.view_info.update(teacher_temperature=cfg['teacher_temperature_by_depth'][str(model.depth)],teacher_q_raw_SHA256=hashlib.sha256(q.detach().cpu().numpy().astype('<f4').tobytes()).hexdigest(),teacher_mass_raw_SHA256=hashlib.sha256(mass.detach().cpu().numpy().astype('<f4').tobytes()).hexdigest());return result
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['independent_current_teacher_calibration'] and p['gate_producer_auditor_capture_key_contract_proved'];identity();old.controls()
    for d in (4,6):incumbent(d);prototype_data(d);teacher_data(d)
def fit(key,gpu):
    fit_kernel(key,gpu);p=OUT/key/'fit_result.json';q=s.read(p);d=config()['cases'][key]['depth'];q.update(local_added_epochs=5,local_adaptation_epoch_labels=True,not_fresh_initialization=True,parent_electronic_adaptation_epochs=8 if d==6 else 0,parent_optical_phase_budget=80 if d==6 else 100,teacher_temperature=config()['teacher_temperature_by_depth'][str(d)]);s.save(p,q)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
