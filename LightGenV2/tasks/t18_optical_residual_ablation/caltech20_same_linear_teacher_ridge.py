"""One convex teacher-logit correction to the existing linear head, no optical forward."""
import argparse,copy,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import caltech20_original_balancedKD as old
HERE,ROOT,s,engine=old.HERE,old.ROOT,old.s,old.engine
FILES=('caltech20_same_linear_teacher_ridge.py','caltech20_same_linear_teacher_ridge_profile.json','check_caltech20_same_linear_teacher_ridge.py','audit_caltech20_same_linear_teacher_ridge.py','CALTECH20_SAME_LINEAR_TEACHER_RIDGE_PROTOCOL_20261017.md')
CPU=HERE/'runs/smoke/caltech20_same_linear_teacher_ridge_cpu_20261017'
def config():
    c=s.read(HERE/FILES[1]);assert c['ridge_relative_regularization']==.01 and c['head_weight_trust_fraction']==.5 and c['new_inference_parameters']==0;c.update(classes=old.base.config()['classes'],class_count=20);return c
def folder(key):assert key in config()['parents'];return HERE/'runs/simulation'/config()['run']/key
def sources():
    src=old.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src
def identity():
    src=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,git);return dict(sources=src,git_commit=git)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def core_digest(state):return s.digest({n:v for n,v in state.items() if not n.startswith('linear.')})
def load_np(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def probability(logits):
    p=np.exp(logits-logits.max(1,keepdims=True));return p/p.sum(1,keepdims=True)
def logits(state,features):return features.astype(np.float64)@state['linear.weight'].numpy().astype(np.float64).T+state['linear.bias'].numpy().astype(np.float64)
def temperature(logits,y,bounds,iterations):
    weights=len(y)/(20*np.bincount(y,minlength=20)[y]);lo,hi=bounds
    def derivative(a):return float(np.mean(weights*((probability(a*logits)*logits).sum(1)-logits[np.arange(len(y)),y])))
    if derivative(lo)>=0:return float(lo)
    if derivative(hi)<=0:return float(hi)
    for _ in range(iterations):
        mid=(lo+hi)/2
        if derivative(mid)>0:hi=mid
        else:lo=mid
    return float((lo+hi)/2)
def admit(key):
    cfg=config();parent=HERE/'runs/simulation'/cfg['parents'][key];meta=s.read(parent/'metadata.json');assert meta['sources']==old.sources();engine.verify_sources(meta['sources'],meta['git_commit']);state=s.read(parent/'status.json');assert state['state']=='balancedKD20_science_and_reviewed_tests_complete' and not state['active'] and not state['pending'] and not state['failures']
    for receipt,where in [('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')]:
        r=s.read(parent/receipt);assert r['actual_audit_returncode']==0 and r['audit_sha256']==s.sha(parent/where/'audit.json') and s.read(parent/where/'audit.json')['passed']
    assert s.read(parent/'campaign_exit.json')['returncode']==0
    for pid in [meta['pid'],s.read(parent.with_name(parent.name+'.started.json'))['supervisor_pid']]:
        p=Path('/proc',str(pid),'cmdline');assert not p.exists() or str(parent).encode() not in p.read_bytes()
    fit=s.read(parent/key/'fit_result.json');assert fit['actual_joint_epochs']==80 and fit['steps']==6880 and fit['best_sha256']==s.sha(parent/key/'best.pt')
    started=s.read(parent/'once_tests'/key/'started.json');assert started['selected_sha256']==fit['best_sha256'] and started['selected_epoch']==fit['selected_epoch'];test=s.read(parent/'once_tests'/key/'result.json');assert test['passed']
    receipt=s.read(old.teacher.OUT/'audit_completed.json');assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(old.teacher.OUT/'science/audit.json')==cfg['teacher_science_sha256'] and s.sha(old.teacher.OUT/'selected_head.pt')==cfg['teacher_head_sha256'];return parent,fit
def cpu_ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();engine.verify_sources(p['sources'],p['git_commit'])

def fit(key):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cpu_ready();parent,fit=admit(key);root=folder(key);out=root/'fit';assert root.exists() and not out.exists();claim(out.with_name(out.name+'.claim.json'),'same_linear_one_TRAIN_solve_and_VAL_scalar_calibration');out.mkdir();torch.set_num_threads(4);cfg=config();pt=torch.load(parent/key/'best.pt',map_location='cpu',weights_only=False);state=pt['model'];arrays={role:load_np(parent/key/('selected_'+role+'.npz')) for role in ('train','val')};teacher=torch.load(old.teacher.OUT/'selected_head.pt',map_location='cpu',weights_only=False);tf=load_np(old.teacher.OUT/'train_features.npz');assert np.array_equal(tf['ids'],arrays['train']['ids']) and np.array_equal(tf['labels'],arrays['train']['labels'])
    for role,a in arrays.items():
        assert a['ids'].tolist()==old.base.manifest()['split_ids'][role];assert np.max(np.abs(probability(logits(state,a['features']))-a['probabilities']))<3e-6
    tx=((tf['features']-teacher['mean'])/teacher['std']).astype(np.float32);teacher_logits=(torch.from_numpy(tx.astype(np.float64))@teacher['model']['weight'].double().T+teacher['model']['bias'].double()).numpy();teacher_probability=probability(teacher_logits);valid=(teacher_logits.argmax(1)==tf['labels'])&(teacher_probability.max(1)>=cfg['teacher_correct_confidence_gate']);X=np.c_[arrays['train']['features'].astype(np.float64),np.ones(len(tf['labels']))];delta=teacher_logits-logits(state,arrays['train']['features']);delta-=delta.mean(1,keepdims=True);delta*=valid[:,None]
    xt=torch.from_numpy(X);yt=torch.from_numpy(delta);A=xt.T@xt;regularization=cfg['ridge_relative_regularization']*float(A.trace())/A.shape[0];correction=torch.linalg.solve(A+regularization*torch.eye(A.shape[0],dtype=torch.float64),xt.T@yt).numpy();old_theta=np.r_[state['linear.weight'].numpy().astype(np.float64).T,state['linear.bias'].numpy().astype(np.float64)[None]];theta=old_theta+cfg['head_weight_trust_fraction']*correction
    raw_logits=np.c_[arrays['val']['features'].astype(np.float64),np.ones(len(arrays['val']['labels']))]@theta;scale=temperature(raw_logits,arrays['val']['labels'],cfg['calibration_inverse_temperature_bounds'],cfg['calibration_iterations']);candidate={n:v.detach().cpu().clone() for n,v in state.items()};candidate['linear.weight']=torch.from_numpy((theta[:-1].T*scale).astype(np.float32));candidate['linear.bias']=torch.from_numpy((theta[-1]*scale).astype(np.float32));assert core_digest(candidate)==core_digest(state) and set(candidate)==set(state)
    metrics={role:dict(fit['metrics'][role],**old.base.metrics(a['labels'],probability(logits(candidate,a['features'])))) for role,a in arrays.items()};assert np.array_equal(raw_logits.argmax(1),probability(logits(candidate,arrays['val']['features'])).argmax(1));parent_val=fit['metrics']['val'];cv=metrics['val'];accepted=cv['accuracy']>parent_val['accuracy']+1e-12 or (abs(cv['accuracy']-parent_val['accuracy'])<1e-12 and cv['balanced_nll']<parent_val['balanced_nll']-1e-6)
    engine.save_pt(out/'candidate.pt',dict(model=candidate,config=cfg,sources=sources(),contract=pt['contract'],epoch=pt['epoch'],metrics=metrics));selected=candidate if accepted else state;selected_metrics=metrics if accepted else fit['metrics'];engine.save_pt(out/'selected.pt',dict(model=selected,config=cfg,sources=sources(),contract=pt['contract'],epoch=pt['epoch'],metrics=selected_metrics))
    for role,a in arrays.items():
        saved=dict(a);saved['probabilities']=probability(logits(selected,a['features']));old.base.save_np(out/('selected_'+role+'.npz'),saved)
    s.save(out/'fit_result.json',dict(passed=True,case=key,parent=str(parent),parent_pt_sha256=fit['best_sha256'],parent_core_digest=core_digest(state),parent_model_digest=s.digest(state),selected_core_digest=core_digest(selected),selected_model_digest=s.digest(selected),parent_sources=s.read(parent/'metadata.json')['sources'],optical_actual_epochs=80,optical_selected_epoch=pt['epoch'],analytical_head_steps=1,new_optical_optimizer_steps=0,new_inference_parameters=0,teacher_valid_samples=int(valid.sum()),ridge_lambda=regularization,calibration_inverse_temperature=scale,VAL_labels_used_for_scalar_calibration=True,candidate_metrics=metrics,metrics=selected_metrics,candidate_accepted=accepted,selection_before_TEST=True,TEST_arrays_read=0,selected_sha256=s.sha(out/'selected.pt'),candidate_sha256=s.sha(out/'candidate.pt'),parent_feature_sha256={role:s.sha(parent/key/('selected_'+role+'.npz')) for role in arrays}),exclusive=True);assert s.sha(parent/key/'best.pt')==fit['best_sha256']

def campaign(key):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cpu_ready();parent,_=admit(key);root=folder(key);assert not root.exists();root.parent.mkdir(parents=True,exist_ok=True);claim(root.with_name(root.name+'.campaign_claim.json'),'one_same_head_CPU_producer_and_own_audit');root.mkdir();s.save(root/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),case=key,parent=str(parent),TEST_arrays_read=0),exclusive=True)
    with (root/'fit.log').open('x') as log:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','fit','--case',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
    s.save(root/'producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0;claim(root/'audit_claim.json','own_saved_linear_ridge_and_temperature_math')
    with (root/'audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_same_linear_teacher_ridge.py'),'--case',key,'--phase','fit'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(root/'science/audit.json')['passed'];s.save(root/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(root/'science/audit.json')),exclusive=True);s.save(root/'status.json',dict(state='same_linear_fit_science_complete_awaiting_review_noTEST',case=key),exclusive=True)

def test(key):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';root=folder(key);parent,_=admit(key);rec=s.read(root/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(root/'science/audit.json') and s.read(root/'science/audit.json')['passed'] and s.read(root/'campaign_exit.json')['returncode']==0;out=root/'once_test';assert not out.exists();claim(out.with_name(out.name+'.claim.json'),'selected_same_head_one_cached_TEST_evaluation');out.mkdir();state=torch.load(root/'fit/selected.pt',map_location='cpu',weights_only=False)['model'];pt=torch.load(parent/key/'best.pt',map_location='cpu',weights_only=False);assert core_digest(state)==core_digest(pt['model']);reused=s.digest(state)==s.digest(pt['model']);a=load_np(parent/'once_tests'/key/'predictions.npz');assert a['ids'].tolist()==old.base.manifest()['split_ids']['test']
    if reused:shutil.copyfile(parent/'once_tests'/key/'predictions.npz',out/'predictions.npz');metrics=s.read(parent/'once_tests'/key/'result.json')['metrics']
    else:a['probabilities']=probability(logits(state,a['features']));old.base.save_np(out/'predictions.npz',a);metrics=old.base.metrics(a['labels'],a['probabilities'])
    s.save(out/'result.json',dict(passed=True,metrics=metrics,model_digest=s.digest(state),selected_sha256=s.sha(root/'fit/selected.pt'),test_reused=reused,production_head_TEST_evaluations=0 if reused else 1,new_optical_TEST_forwards=0,features_reused_from_parent=True,parent_TEST_prediction_sha256=s.sha(parent/'once_tests'/key/'predictions.npz'),test_used_for_selection=False,VAL_used_for_scalar_calibration=True,test_informed_development=True),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','test'],required=True);p.add_argument('--case',required=True);a=p.parse_args()
    try:globals()[a.phase](a.case)
    except BaseException as e:
        out=folder(a.case);mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
