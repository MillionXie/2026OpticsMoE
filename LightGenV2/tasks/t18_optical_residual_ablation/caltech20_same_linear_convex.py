"""Fully fit the existing linear readout on frozen optical features, CPU only."""
import argparse, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import caltech20_same_linear_teacher_ridge as old

HERE, ROOT, s, engine = old.HERE, old.ROOT, old.s, old.engine
FILES = ('caltech20_same_linear_convex.py', 'caltech20_same_linear_convex_profile.json',
         'check_caltech20_same_linear_convex.py', 'audit_caltech20_same_linear_convex.py',
         'CALTECH20_SAME_LINEAR_CONVEX_PROTOCOL_20261017.md')
CPU = HERE/'runs/smoke/caltech20_same_linear_convex_cpu_20261017'

def config():
    c=s.read(HERE/FILES[1]); assert c['teacher_weight']==.02 and c['CE_weight']==.98
    assert c['new_inference_parameters']==0 and c['core_frozen']
    return c

def folder(key):
    assert key in config()['cases']; return HERE/'runs/simulation'/config()['run']/key

def sources():
    z=old.sources()
    for n in FILES: z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

def identity():
    z=sources(); git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    engine.verify_sources(z,git); return dict(sources=z,git_commit=git)

def claim(path,phase): s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def gone(pid): return not Path('/proc',str(pid)).exists()

def admit(key):
    old.admit(key); root=old.folder(key); meta=s.read(root/'metadata.json')
    assert meta['sources']==old.sources(); engine.verify_sources(meta['sources'],meta['git_commit'])
    assert s.read(root/'status.json')['state']=='same_linear_fit_science_and_cached_TEST_complete'
    for receipt,where in [('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')]:
        r=s.read(root/receipt); assert r['actual_audit_returncode']==0 and r['audit_sha256']==s.sha(root/where/'audit.json') and s.read(root/where/'audit.json')['passed']
    for name in ('producer_exit.json','campaign_exit.json','test_exit.json'):
        ex=s.read(root/name); assert ex['returncode']==0 and ex['actual_exit'] and gone(ex['pid'])
    assert not any((root/n).exists() for n in ('failure.json','fit_audit_failure.json','test_audit_failure.json'))
    fit=s.read(root/'fit/fit_result.json'); assert fit['selected_sha256']==s.sha(root/'fit/selected.pt')
    assert fit['optical_actual_epochs']==80 and fit['new_optical_optimizer_steps']==0
    t=s.read(root/'once_test/result.json'); assert t['selected_sha256']==fit['selected_sha256'] and t['model_digest']==fit['selected_model_digest']
    return root,fit

def cpu_ready():
    z=s.read(CPU/'checks.json'); assert z['passed'] and z['sources']==sources()
    engine.verify_sources(z['sources'],z['git_commit'])

def training_data(parent,state):
    a={r:old.load_np(parent/'fit'/('selected_'+r+'.npz')) for r in ('train','val')}
    for role,z in a.items():
        assert z['ids'].tolist()==old.old.base.manifest()['split_ids'][role]
        assert np.max(np.abs(old.probability(old.logits(state,z['features']))-z['probabilities']))<3e-6
    teacher=torch.load(old.old.teacher.OUT/'selected_head.pt',map_location='cpu',weights_only=False)
    tf=old.load_np(old.old.teacher.OUT/'train_features.npz')
    assert np.array_equal(tf['ids'],a['train']['ids']) and np.array_equal(tf['labels'],a['train']['labels'])
    tx=((tf['features']-teacher['mean'])/teacher['std']).astype(np.float32).astype(np.float64)
    tz=tx@teacher['model']['weight'].double().numpy().T+teacher['model']['bias'].double().numpy()
    valid=(tz.argmax(1)==tf['labels'])&(old.probability(tz).max(1)>=config()['teacher_correct_confidence_gate'])
    return a,old.probability(tz/config()['teacher_temperature']),valid

def objective(theta,X,y,q,valid,anchor,cfg):
    n=len(y); z=X@theta; soft=F.one_hot(y,20).double()*(1-cfg['label_smoothing'])+cfg['label_smoothing']/20
    counts=torch.bincount(y,minlength=20).double(); weights=counts.rsqrt(); weights/=weights[y].mean()
    ce=-(soft*weights[None]*F.log_softmax(z,dim=1)).sum(1).mean()
    T=cfg['teacher_temperature']; kd=((q*(q.clamp_min(1e-300).log()-F.log_softmax(z/T,dim=1))).sum(1)*valid).mean()*T*T
    penalty=(theta-anchor).square().sum()/(2*n)
    return cfg['CE_weight']*ce+cfg['teacher_weight']*kd+penalty,(ce,kd,penalty)

def fit(key):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''; cpu_ready(); parent,pf=admit(key); root=folder(key); out=root/'fit'
    assert root.exists() and not out.exists(); claim(root/'fit_claim.json','single_CPU_LBFGS_existing_linear_only'); out.mkdir(); torch.set_num_threads(4)
    cfg=config(); pt=torch.load(parent/'fit/selected.pt',map_location='cpu',weights_only=False); state=pt['model']; a,q,valid=training_data(parent,state)
    X=np.c_[a['train']['features'].astype(np.float64),np.ones(len(q))]
    anchor=np.r_[state['linear.weight'].double().numpy().T,state['linear.bias'].double().numpy()[None]]
    parameter=torch.nn.Parameter(torch.tensor(anchor)); tensors=[torch.tensor(X),torch.tensor(a['train']['labels']).long(),torch.tensor(q),torch.tensor(valid).double(),torch.tensor(anchor)]
    optimizer=torch.optim.LBFGS([parameter],lr=1.,max_iter=cfg['max_iter'],max_eval=cfg['max_eval'],history_size=cfg['history_size'],line_search_fn='strong_wolfe',tolerance_grad=cfg['tolerance_grad'],tolerance_change=cfg['tolerance_change'])
    trace=[]
    def closure():
        optimizer.zero_grad(); loss,parts=objective(parameter,*tensors,cfg); loss.backward(); assert torch.isfinite(loss) and torch.isfinite(parameter.grad).all()
        trace.append(dict(theta=parameter.detach().numpy().copy(),loss=float(loss.detach()),parts=[float(x.detach()) for x in parts],grad_inf=float(parameter.grad.abs().max())))
        return loss
    optimizer.step(closure)
    final_loss,parts=objective(parameter,*tensors,cfg); gradient=torch.autograd.grad(final_loss,parameter)[0]
    stats=optimizer.state[parameter]; theta=parameter.detach().numpy().copy(); assert len(trace)==stats['func_evals']
    assert float(final_loss)<=trace[0]['loss']+1e-9
    old.old.base.save_np(out/'closure_trace.npz',dict(theta=np.stack([z['theta'] for z in trace]),loss=np.array([z['loss'] for z in trace]),parts=np.array([z['parts'] for z in trace]),grad_inf=np.array([z['grad_inf'] for z in trace])))
    engine.save_pt(out/'optimization.pt',dict(theta=parameter.detach(),anchor=torch.tensor(anchor),optimizer=optimizer.state_dict(),final_gradient=gradient.detach(),sources=sources()))
    vz=np.c_[a['val']['features'].astype(np.float64),np.ones(len(a['val']['labels']))]@theta
    scale=old.temperature(vz,a['val']['labels'],cfg['calibration_inverse_temperature_bounds'],cfg['calibration_iterations'])
    candidate={n:v.clone() for n,v in state.items()}; candidate['linear.weight']=torch.tensor((theta[:-1].T*scale).astype(np.float32)); candidate['linear.bias']=torch.tensor((theta[-1]*scale).astype(np.float32))
    assert set(candidate)==set(state) and old.core_digest(candidate)==old.core_digest(state)
    cm={r:dict(pf['metrics'][r],**old.old.base.metrics(z['labels'],old.probability(old.logits(candidate,z['features'])))) for r,z in a.items()}
    cv,pv=cm['val'],pf['metrics']['val']; accepted=cv['accuracy']>pv['accuracy']+1e-12 or (abs(cv['accuracy']-pv['accuracy'])<1e-12 and cv['balanced_nll']<pv['balanced_nll']-1e-6)
    selected=candidate if accepted else state; metrics=cm if accepted else pf['metrics']
    for name,model,met in [('candidate.pt',candidate,cm),('selected.pt',selected,metrics)]: engine.save_pt(out/name,dict(model=model,metrics=met,epoch=pt['epoch'],contract=pt['contract'],config=cfg,sources=sources()))
    for role,z in a.items():
        saved=dict(z); saved['probabilities']=old.probability(old.logits(selected,z['features'])); old.old.base.save_np(out/('selected_'+role+'.npz'),saved)
    s.save(out/'fit_result.json',dict(passed=True,case=key,metrics=metrics,candidate_metrics=cm,candidate_accepted=accepted,optical_actual_epochs=80,optical_selected_epoch=pt['epoch'],head_LBFGS_iterations=int(stats['n_iter']),closure_evaluations=int(stats['func_evals']),final_objective=float(final_loss.detach()),final_gradient_inf=float(gradient.abs().max()),stationary_gradient_1e5=bool(float(gradient.abs().max())<=1e-5),teacher_valid_samples=int(valid.sum()),calibration_inverse_temperature=scale,parent_pt_sha256=s.sha(parent/'fit/selected.pt'),parent_core_digest=old.core_digest(state),selected_core_digest=old.core_digest(selected),selected_model_digest=s.digest(selected),selected_sha256=s.sha(out/'selected.pt'),candidate_sha256=s.sha(out/'candidate.pt'),trace_sha256=s.sha(out/'closure_trace.npz'),optimization_sha256=s.sha(out/'optimization.pt'),parent_feature_sha256={r:s.sha(parent/'fit'/('selected_'+r+'.npz')) for r in a},new_optical_optimizer_steps=0,new_inference_parameters=0,TEST_arrays_read=0,VAL_labels_used_for_scalar_calibration=True,selection_before_TEST=True),exclusive=True)

def campaign(key):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''; cpu_ready(); parent,_=admit(key); root=folder(key); assert not root.exists(); root.parent.mkdir(parents=True,exist_ok=True); claim(root.with_name(root.name+'.campaign_claim.json'),'single_CPU_head_fit_own_science'); root.mkdir()
    s.save(root/'metadata.json',dict(identity(),pid=os.getpid(),case=key,config=config(),parent=str(parent)),exclusive=True)
    with (root/'fit.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','fit','--case',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT); rc=p.wait()
    s.save(root/'producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True); assert rc==0
    claim(root/'audit_claim.json','independent_saved_convex_objective_and_gradients')
    with (root/'audit.log').open('x') as log: rc=subprocess.run([sys.executable,str(HERE/FILES[3]),'--phase','fit','--case',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(root/'science/audit.json')['passed']; s.save(root/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(root/'science/audit.json')),exclusive=True)
    s.save(root/'status.json',dict(state='convex_head_science_complete_awaiting_review_noTEST',case=key))

def test(key):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''; parent,_=admit(key); root=folder(key); r=s.read(root/'audit_completed.json')
    assert r['actual_audit_returncode']==0 and r['audit_sha256']==s.sha(root/'science/audit.json') and s.read(root/'science/audit.json')['passed']
    assert s.read(root/'campaign_exit.json')['returncode']==0 and gone(s.read(root/'campaign_exit.json')['pid']); out=root/'once_test'; assert not out.exists(); claim(root/'test_claim.json','once_selected_linear_cached_TEST'); out.mkdir()
    state=torch.load(root/'fit/selected.pt',map_location='cpu',weights_only=False)['model']; ps=torch.load(parent/'fit/selected.pt',map_location='cpu',weights_only=False)['model']; assert old.core_digest(state)==old.core_digest(ps)
    reused=s.digest(state)==s.digest(ps); z=old.load_np(parent/'once_test/predictions.npz'); assert z['ids'].tolist()==old.old.base.manifest()['split_ids']['test']
    if reused: shutil.copyfile(parent/'once_test/predictions.npz',out/'predictions.npz'); metrics=s.read(parent/'once_test/result.json')['metrics']
    else: z['probabilities']=old.probability(old.logits(state,z['features'])); old.old.base.save_np(out/'predictions.npz',z); metrics=old.old.base.metrics(z['labels'],z['probabilities'])
    s.save(out/'result.json',dict(passed=True,metrics=metrics,test_reused=reused,production_head_TEST_evaluations=0 if reused else 1,new_optical_TEST_forwards=0,selected_sha256=s.sha(root/'fit/selected.pt'),model_digest=s.digest(state),parent_TEST_prediction_sha256=s.sha(parent/'once_test/predictions.npz'),test_used_for_selection=False,VAL_used_for_scalar_calibration=True,test_informed_development=True),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['fit','campaign','test'],required=True); p.add_argument('--case',required=True); a=p.parse_args()
    try: globals()[a.phase](a.case)
    except BaseException as e:
        out=folder(a.case); mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists(): s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
