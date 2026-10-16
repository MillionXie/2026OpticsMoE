"""Cross-fitted scale/intercept calibration, folded into the SAME terminal head."""
import argparse,copy,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
import caltech20_alt246_richsplit_headanchor as prior
temp=prior.adapted;HERE,ROOT,s,engine,b=temp.HERE,temp.ROOT,temp.s,temp.engine,temp.b
PROFILE=HERE/'caltech20_alt246_richsplit_crossbias_profile.json'
FILES=('caltech20_alt246_richsplit_crossbias.py','caltech20_alt246_richsplit_crossbias_profile.json','check_caltech20_alt246_richsplit_crossbias.py','audit_caltech20_alt246_richsplit_crossbias.py','CALTECH20_ALT246_RICHSPLIT_CROSSBIAS_PROTOCOL_20261017.md')
RUN='caltech20_rich20_L6_crossfit_bias_s17_20261017';CPU=HERE/'runs/smoke/caltech20_richsplit_crossbias_cpu_20261017'
def config():
    p=s.read(PROFILE);assert p['folds']==5 and p['ridge']==.1 and p['inverse_temperature_bounds']==[1/3,2.] and p['seed']==20261017 and not p['automatic_retry'];return p
def sources():
    q=prior.sources()
    for name in FILES:q[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return q
def identity():
    q=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(q,commit);return dict(sources=q,git_commit=commit,versions=temp.versions())
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def parent():
    p=temp.parent();root=HERE/'runs/simulation'/temp.RUN;assert s.read(root/'status.json')['state']=='fixed_LAST_temperature_science_test_complete';meta=s.read(root/'metadata.json');assert meta['sources']==temp.sources();engine.verify_sources(meta['sources'],meta['git_commit']);temp.previous.receipt(root,'audit_completed.json','science');temp.previous.receipt(root,'test_audit_completed.json','saved_test_audit');ex=s.read(root/'campaign_exit.json');assert ex['returncode']==0 and ex['actual_exit'];assert s.read(root/'comparison.json')['six_above_four'] is False;extraction=s.read(root/'extract/result.json');assert extraction['parent']==p and extraction['model_digest']==p['EMA_digest'];p=copy.deepcopy(p);p['cache_root']=str(root/'extract');p['cache_sha256']=extraction['prediction_sha256'];p['cache_science_sha256']=s.sha(root/'science/audit.json');p['test_registry']=dict(p['test_registry']);test=s.read(root/'test/result.json');p['test_registry'][test['model_digest']]=dict(result_path=str(root/'test/result.json'),predictions_path=test['predictions_path'],predictions_sha256=test['predictions_sha256'],metrics=test['metrics'])
    old=HERE/'runs/simulation'/prior.RUN;assert s.read(old/'status.json')['state']=='same_natural_head_science_test_complete' and s.read(old/'metadata.json')['sources']==prior.sources();temp.previous.receipt(old,'audit_completed.json','science');temp.previous.receipt(old,'test_audit_completed.json','saved_test_audit');assert s.read(old/'campaign_exit.json')['returncode']==0;test=s.read(old/'test/result.json');assert test['predictions_sha256']==s.sha(test['predictions_path']);p['test_registry'][test['model_digest']]=dict(result_path=str(old/'test/result.json'),predictions_path=test['predictions_path'],predictions_sha256=test['predictions_sha256'],metrics=test['metrics']);return p
def load(role,p):
    assert role in ('train','val');path=Path(p['cache_root'])/(role+'.npz');assert s.sha(path)==p['cache_sha256'][role];m=b.dataset.manifest()
    with np.load(path,allow_pickle=False) as z:q={k:z[k].copy() for k in z.files}
    assert q['ids'].tolist()==m['split_ids'][role] and np.bincount(q['labels'],minlength=20).tolist()==m['supports'][role];return q
def logits(data,state):return data['features'].astype(np.float64)@state['linear.weight'].numpy().astype(np.float64).T+state['linear.bias'].numpy().astype(np.float64)
def objective(theta,z,y):
    x=z*theta[0]+theta[1:];x-=x.max(1,keepdims=True);prob=np.exp(x);prob/=prob.sum(1,keepdims=True);logp=x-np.log(np.exp(x).sum(1,keepdims=True));delta=prob.copy();delta[np.arange(len(y)),y]-=1;delta/=len(y);penalty=theta.copy();penalty[0]-=1;ridge=config()['ridge'];loss=-float(logp[np.arange(len(y)),y].mean())+.5*ridge*float(penalty@penalty);grad=np.r_[(delta*z).sum(),delta.sum(0)]+ridge*penalty;return loss,grad
def solve(z,y):
    trace=[]
    def fun(theta):
        value,grad=objective(theta,z,y);trace.append(dict(parameters=theta.tolist(),objective=value));return value,grad
    result=minimize(fun,np.r_[1.,np.zeros(20)],method='L-BFGS-B',jac=True,bounds=[tuple(config()['inverse_temperature_bounds'])]+[(None,None)]*20,options=dict(maxiter=300,maxls=40,gtol=1e-10,ftol=1e-12));assert result.success,result.message;return dict(parameters=result.x.tolist(),objective=float(result.fun),gradient=result.jac.tolist(),iterations=int(result.nit),function_evaluations=int(result.nfev),trace=trace,success=bool(result.success))
def folded(state,theta):
    theta=np.asarray(theta,np.float64);q={k:v.clone() for k,v in state.items()};q['linear.weight']=torch.from_numpy((state['linear.weight'].numpy().astype(np.float64)*theta[0]).astype(np.float32));q['linear.bias']=torch.from_numpy((state['linear.bias'].numpy().astype(np.float64)*theta[0]+theta[1:]).astype(np.float32));assert temp.head.frozen_digest(q)==temp.head.frozen_digest(state);return q
def predict(data,state,reference):return temp.head.predict(data,state['linear.weight'].numpy(),state['linear.bias'].numpy(),reference)
def folds(labels):
    q=np.empty(len(labels),np.int64)
    for k in range(20):
        indices=np.flatnonzero(labels==k);rng=np.random.default_rng(np.random.SeedSequence([config()['seed'],k]));indices=rng.permutation(indices);q[indices]=np.arange(len(indices))%5
    return q
def cpu(out):
    p=parent();assert out==CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(out.with_name(out.name+'.claim.json'),'crossbias_CPU_math');state=temp.parent_state(p);torch.set_num_threads(4);rng=np.random.default_rng(17);z=rng.normal(size=(20,20));labels=np.arange(20);theta=np.r_[.9,rng.normal(0,.05,20)];value,grad=objective(theta,z,labels);numeric=[]
    for i in range(21):
        plus=theta.copy();minus=theta.copy();plus[i]+=1e-6;minus[i]-=1e-6;numeric.append((objective(plus,z,labels)[0]-objective(minus,z,labels)[0])/2e-6)
    err=float(np.abs(grad-numeric).max());assert err<1e-8 and np.isfinite(grad).all() and np.linalg.norm(grad)>0;data=load('val',p);assignment=folds(data['labels']);assert set(assignment)==set(range(5)) and all(np.bincount(data['labels'][assignment!=f],minlength=20).min()>0 for f in range(5));out.mkdir();s.save(out/'checks.json',dict(identity(),passed=True,parent=p,analytic_gradient_maxabs=err,calibration_parameters=21,inference_parameters_added=0,real_model_forwards=0,optimizer_updates=0,TEST_images_read=0),exclusive=True)
def cpu_ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();engine.verify_sources(q['sources'],q['git_commit'])
def cv(out):
    cpu_ready();p=parent();assert not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(out.with_name(out.name+'.claim.json'),'crossbias_out_of_fold_calibration');out.mkdir();torch.set_num_threads(4);state=temp.parent_state(p);data=load('val',p);z=logits(data,state);assignment=folds(data['labels']);baseline,base_arrays=predict(data,state,p['metrics']['val']);prob=np.empty((len(z),20),np.float64);records=[]
    for fold in range(5):
        train=np.flatnonzero(assignment!=fold);heldout=np.flatnonzero(assignment==fold);solved=solve(z[train],data['labels'][train]);model=folded(state,solved['parameters']);local={k:(v[heldout] if k!='probe_amplitudes' else v) for k,v in data.items()};_,arrays=predict(local,model,p['metrics']['val']);prob[heldout]=arrays['probabilities'];records.append(dict(fold=fold,train_indices_sha256=s.digest(train),heldout_indices_sha256=s.digest(heldout),TRAIN_n=len(train),heldout_n=len(heldout),fit=solved))
    measured=engine.metrics(data['labels'],prob);eligible=measured['accuracy']>baseline['accuracy'] and measured['nll']<=baseline['nll']-.001;b.save_np(out/'out_of_fold.npz',dict(probabilities=prob,baseline_probabilities=base_arrays['probabilities'],ids=data['ids'],labels=data['labels'],fold=assignment));s.save(out/'result.json',dict(identity(),parent=p,folds=records,baseline=baseline,OOF=measured,eligible=eligible,criterion='OOF accuracy strictly above fixed unscaled LAST EMA120 and naturalNLL improves>=.001; never TEST',OOF_sha256=s.sha(out/'out_of_fold.npz'),new_TEST_evaluations=0),exclusive=True)
def fit(out):
    cpu_ready();p=parent();root=out.parent;receipt(root,'cv');assert s.read(root/'cv/result.json')['eligible'] and not out.exists();claim(out.with_name(out.name+'.claim.json'),'crossbias_full_VAL_calibration');out.mkdir();state=temp.parent_state(p);data=load('val',p);solved=solve(logits(data,state),data['labels']);model=folded(state,solved['parameters']);metrics={};saved={}
    for role in ('train','val'):
        raw=load(role,p);metrics[role],saved[role]=predict(raw,model,p['metrics'][role]);b.save_np(out/('selected_'+role+'.npz'),saved[role])
    checkpoint=dict(identity(),model=model,epoch=120,contract=p['contract'],config=config(),parent=p,calibration_uses_VAL_labels=True,calibration_parameters=solved['parameters'],metrics=metrics);engine.save_pt(out/'best.pt',checkpoint);engine.save_pt(out/'last.pt',dict(checkpoint,solver=solved));s.save(out/'fit_result.json',dict(passed=True,actual_epochs=120,selected_epoch=120,calibration_degrees_of_freedom=21,inference_parameters_added=0,calibration_uses_VAL_labels=True,additional_optical_epochs=0,parameters=solved['parameters'],optimizer_iterations=solved['iterations'],metrics=metrics,best_path=str(out/'best.pt'),best_sha256=s.sha(out/'best.pt'),last_sha256=s.sha(out/'last.pt'),selected_train_path=str(out/'selected_train.npz'),selected_val_path=str(out/'selected_val.npz')),exclusive=True)
def receipt(root,phase):
    folder={'cv':'cv_science','fit':'science','test':'saved_test_audit'}[phase];record={'cv':'cv_audit_completed.json','fit':'audit_completed.json','test':'test_audit_completed.json'}[phase];q=s.read(root/record);assert q['actual_audit_returncode']==0 and q['audit_sha256']==s.sha(root/folder/'audit.json') and s.read(root/folder/'audit.json')['passed'];return q
def test(a):
    root=a.out.parent;receipt(root,'fit');p=parent();f=s.read(root/'fit/fit_result.json');assert not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'crossbias_selected_once_test');a.out.mkdir();state=torch.load(f['best_path'],map_location='cpu',weights_only=False)['model'];digest=s.digest(state);s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),model_digest=digest,best_sha256=f['best_sha256'],selected_epoch=120),exclusive=True)
    if digest in p['test_registry']:
        old=p['test_registry'][digest];assert old['predictions_sha256']==s.sha(old['predictions_path']);result=dict(metrics=old['metrics'],predictions_path=old['predictions_path'],predictions_sha256=old['predictions_sha256'],reused=True,new_test_evaluations=0)
    else:
        assert a.gpu in b.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);model=b.RegularizedMoE(6,.3).cuda();model.load_state_dict(state,strict=True);metrics,arrays=b.evaluate(model,b.load('test'));b.save_np(a.out/'predictions.npz',arrays);result=dict(metrics=metrics,predictions_path=str(a.out/'predictions.npz'),predictions_sha256=s.sha(a.out/'predictions.npz'),reused=False,new_test_evaluations=1)
    s.save(a.out/'result.json',dict(result,model_digest=digest,best_sha256=f['best_sha256'],selected_epoch=120,test_used_for_selection=False,VAL_used_for_calibration=True,test_informed_development=True),exclusive=True)
def child(root,phase,gpu=''):
    with (root/(phase+'.log')).open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--out',str(root/phase),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);rc=proc.wait()
    s.save(root/(phase+'_exit.json'),dict(pid=proc.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Stage failed: preserve/no retry'
def own_audit(root,phase):
    claim(root/(phase+'_audit_claim.json'),'crossbias_own_'+phase);folder={'cv':'cv_science','fit':'science','test':'saved_test_audit'}[phase];record={'cv':'cv_audit_completed.json','fit':'audit_completed.json','test':'test_audit_completed.json'}[phase]
    with (root/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_richsplit_crossbias.py'),'--base',str(root),'--out',str(root/folder),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(root/folder/'audit.json')['passed'],'Own audit failed: no retry';s.save(root/record,dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(root/folder/'audit.json')),exclusive=True)
def campaign(a):
    cpu_ready();p=parent();root=a.out;assert root==HERE/'runs/simulation'/RUN and not root.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(root.with_name(root.name+'.campaign_claim.json'),'crossbias_gated_campaign');root.mkdir();s.save(root/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),parent=p,actual_command=sys.argv),exclusive=True);s.save(root/'status.json',dict(state='crossfit_calibration_running'));child(root,'cv');own_audit(root,'cv');cv=s.read(root/'cv/result.json')
    if not cv['eligible']:s.save(root/'status.json',dict(state='crossfit_rejected_no_test',reason='No held-out-fold accuracy/NLL gain',new_TEST_evaluations=0));return
    child(root,'fit');own_audit(root,'fit');gpu=None;s.gpu_budget.ALLOWED.update(b.GPU.values())
    for card in (2,5,4,0):
        if b.slots()['counts'][b.GPU[card]]:continue
        try:s.gpu_budget.preflight(b.GPU[card],shared=True,memory_gib=2);gpu=b.GPU[card];break
        except AssertionError:continue
    assert gpu is not None;child(root,'test',gpu);own_audit(root,'test');f=s.read(root/'fit/fit_result.json');t=s.read(root/'test/result.json');reference=temp.previous.BASE;control=s.read(reference/'once_tests/L6_rho0/result.json')['metrics'];four=s.read(reference/'once_tests/L4_rho03/result.json')['metrics'];s.save(root/'comparison.json',dict(actual_epochs=120,selected_epoch=120,train=f['metrics']['train'],val=f['metrics']['val'],OOF=cv['OOF'],test=t['metrics'],control100=control,reference_four100=four,test_gap_pp=100*(t['metrics']['accuracy']-control['accuracy']),six_above_four=t['metrics']['accuracy']>four['accuracy'],VAL_used_for_calibration=True,test_informed_development=True),exclusive=True);s.save(root/'status.json',dict(state='crossfit_bias_science_test_complete',comparison_sha256=s.sha(root/'comparison.json')))
def supervise(a):
    cpu_ready();parent();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(a.out.with_name(a.out.name+'.launch_claim.json'),'finite_crossbias_supervision')
    with a.out.with_name(a.out.name+'.campaign.log').open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(a.out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);s.save(a.out.with_name(a.out.name+'.started.json'),dict(identity(),supervisor_pid=os.getpid(),campaign_pid=proc.pid),exclusive=True);rc=proc.wait()
    path=a.out/'campaign_exit.json' if a.out.exists() else a.out.with_name(a.out.name+'.campaign_exit.json');s.save(path,dict(pid=proc.pid,supervisor_pid=os.getpid(),returncode=rc,actual_exit=True),exclusive=True);assert rc==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','cv','fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu');a=p.parse_args()
    try:cpu(a.out) if a.phase=='cpu' else cv(a.out) if a.phase=='cv' else fit(a.out) if a.phase=='fit' else globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
