"""Fixed LAST EMA120 candidate: one confidence scalar folded into SAME linear head."""
import argparse,copy,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
import torch
import caltech20_alt246_richsplit_cachedkd as previous
HERE,ROOT,s,engine,b=previous.HERE,previous.ROOT,previous.s,previous.engine,previous.b
head=previous.rejected.calibrated
PROFILE=HERE/'caltech20_alt246_richsplit_tempcal_profile.json'
FILES=('caltech20_alt246_richsplit_tempcal.py','caltech20_alt246_richsplit_tempcal_profile.json','check_caltech20_alt246_richsplit_tempcal.py','audit_caltech20_alt246_richsplit_tempcal.py','CALTECH20_ALT246_RICHSPLIT_TEMPCAL_PROTOCOL_20261017.md')
RUN='caltech20_rich20_L6_lastEMA120_temperature_s17_20261017'
CPU=HERE/'runs/smoke/caltech20_richsplit_tempcal_cpu_20261017'

def config():
    p=s.read(PROFILE);assert p['main_depth']==6 and p['temperature_bounds']==[.5,3.] and p['calibration_uses_VAL_labels'] and p['inference_parameters_added']==0 and not p['automatic_retry'];return p
def sources():
    q=previous.sources()
    for name in FILES:q[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return q
def versions():return dict(previous.versions(),scipy=scipy.__version__)
def identity():
    q=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(q,commit);return dict(sources=q,git_commit=commit,versions=versions())
def claim(path,phase):s.save(path,dict(identity(),phase=phase,pid=os.getpid(),automatic_retry=False),exclusive=True)
def parent():
    root=previous.expected_root(6,previous.BASE);assert s.read(root/'status.json')['state']=='cachedkd_science_test_complete';meta=s.read(root/'metadata.json');assert meta['sources']==previous.sources();engine.verify_sources(meta['sources'],meta['git_commit']);previous.receipt(root,'audit_completed.json','science');previous.receipt(root,'test_audit_completed.json','saved_test_audit');ex=s.read(root/'campaign_exit.json');assert ex['returncode']==0 and ex['actual_exit'];proc=Path('/proc',str(meta['pid']),'cmdline');assert not proc.exists() or str(root).encode() not in proc.read_bytes();f=s.read(root/'fit/fit_result.json');history=s.read(root/'fit/history.json');assert f['actual_epochs']==len(history)==120 and f['steps']==23640 and f['history_sha256']==s.sha(root/'fit/history.json') and f['last_sha256']==s.sha(root/'fit/last.pt');last=torch.load(root/'fit/last.pt',map_location='cpu',weights_only=False);assert last['epoch']==120 and s.digest(last['ema'])==history[-1]['ema_digest'];reference=s.read(previous.BASE/'L6_rho03/fit_result.json');assert history[-1]['val']['accuracy']>=reference['metrics']['val']['accuracy']+.01,'Calibration hypothesis requires the fixed LAST state to improve validation classification';assert s.read(root/'comparison.json')['six_above_baseline_four'] is False,'Primary depth target already met: no automatic confidence candidate needed';contract=torch.load(f['best_path'],map_location='cpu',weights_only=False)['contract'];assert contract['main_depth']==6 and contract['terminal_linear_parameters']==5140
    return dict(root=str(root),last=str(root/'fit/last.pt'),last_sha256=f['last_sha256'],EMA_digest=history[-1]['ema_digest'],actual_epochs=120,steps=23640,metrics=dict(train=history[-1]['train'],val=history[-1]['val']),contract=contract,test_registry=s.read(root/'comparison.json')['test_registry'])
def parent_state(p):
    assert s.sha(p['last'])==p['last_sha256'];state=torch.load(p['last'],map_location='cpu',weights_only=False)['ema'];assert s.digest(state)==p['EMA_digest'];return state
def cpu(out):
    assert out==CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';p=parent();claim(out.with_name(out.name+'.claim.json'),'folded_temperature_CPU_math');torch.set_num_threads(4);logits=torch.linspace(-3,3,400,dtype=torch.float64).reshape(20,20);labels=torch.arange(20);u=torch.tensor(.15,dtype=torch.float64,requires_grad=True);value=-torch.nn.functional.log_softmax(logits/torch.exp(u),1)[labels,labels].mean();value.backward();T=float(torch.exp(u).detach());q=(logits.detach().numpy()/T);q-=q.max(1,keepdims=True);q=np.exp(q);q/=q.sum(1,keepdims=True);expected=float(np.mean((logits.numpy()[np.arange(20),np.arange(20)]-(q*logits.numpy()).sum(1))/T));err=abs(float(u.grad)-expected);assert err<1e-10 and torch.isfinite(u.grad) and u.grad.abs()>0;state=parent_state(p);assert len(state)==30 and head.frozen_digest(state)==head.frozen_digest(copy.deepcopy(state));out.mkdir();s.save(out/'checks.json',dict(identity(),passed=True,parent=p,analytic_log_temperature_gradient_maxabs=err,trainable_degrees_of_freedom=1,terminal_parameters=5140,inference_parameters_added=0,real_model_forwards=0,optimizer_updates=0,TEST_images_read=0),exclusive=True)
def cpu_ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['versions']==versions();engine.verify_sources(q['sources'],q['git_commit'])
def extract(a):
    cpu_ready();p=parent();assert not a.out.exists() and a.gpu in b.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'fixed_LAST_EMA_TRAIN_VAL_extract');a.out.mkdir();s.gpu_budget.ALLOWED.update(b.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);state=parent_state(p);model=b.RegularizedMoE(6,.3).cuda();model.load_state_dict(state,strict=True);model.requires_grad_(False);model.eval();train=b.load('train');idx=np.array([int((train[1]==k).nonzero()[0]) for k in range(20)]);x=b.encode(train[0][idx].cuda(),None,100);parts=[];captures=[]
    with torch.no_grad():
        for micro in (1,4):
            ps=[];cs=[]
            for off in range(0,20,micro):q,c=model(x[off:off+micro]);ps.append(q);cs.append(c)
            parts.append(torch.cat(ps));captures.append(torch.cat(cs))
    err=max(float((parts[0]-parts[1]).abs().max()),float((captures[0]-captures[1]).abs().max()));assert err<=2e-6 and s.digest(model.state_dict())==p['EMA_digest'] and all(v.grad is None for v in model.parameters());s.save(a.out/'numeric_gate.json',dict(passed=True,micro1_vs4_probability_capture_maxabs=err,parameters_frozen=True,optimizer_updates=0,test_arrays_read=0),exclusive=True);metrics={};files={}
    for role,data in (('train',train),('val',b.load('val'))):
        metrics[role],arrays=b.evaluate(model,data);b.save_np(a.out/(role+'.npz'),arrays);files[role]=s.sha(a.out/(role+'.npz'))
    assert s.digest(model.state_dict())==p['EMA_digest'];s.save(a.out/'result.json',dict(identity(),parent=p,pid=os.getpid(),gpu_uuid=a.gpu,policy=policy,metrics=metrics,prediction_sha256=files,model_digest=p['EMA_digest'],TRAIN_images=3141,VAL_images=383,numeric_gate_images=40,TEST_images=0,optimizer_updates=0),exclusive=True)
def fit(out):
    cpu_ready();p=parent();root=out.parent;ex=s.read(root/'extract_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(out.with_name(out.name+'.claim.json'),'VAL_temperature_scalar_fit');out.mkdir();torch.set_num_threads(4);cfg=config();state=parent_state(p);data={}
    for role in ('train','val'):
        path=root/'extract'/(role+'.npz');assert s.sha(path)==s.read(root/'extract/result.json')['prediction_sha256'][role]
        with np.load(path,allow_pickle=False) as z:data[role]={k:z[k].copy() for k in z.files}
    W=state['linear.weight'].numpy().astype(np.float64);bias=state['linear.bias'].numpy().astype(np.float64);logits=data['val']['features'].astype(np.float64)@W.T+bias;labels=data['val']['labels'];weight=1/(20*np.bincount(labels,minlength=20)[labels]);trace=[];started=time.monotonic()
    def objective(logT):
        z=logits/np.exp(logT);z-=z.max(1,keepdims=True);logp=z-np.log(np.exp(z).sum(1,keepdims=True));loss=float(-np.sum(weight*logp[np.arange(len(labels)),labels]));trace.append(dict(log_temperature=float(logT),balanced_VAL_NLL=loss));return loss
    baseline=objective(0.);result=minimize_scalar(objective,bounds=tuple(np.log(cfg['temperature_bounds'])),method='bounded',options=dict(xatol=1e-10,maxiter=200));assert result.success and len(trace)==result.nfev+1;T=float(np.exp(result.x));assert .5<=T<=3 and time.monotonic()-started<cfg['maximum_seconds'];candidate={k:v.clone() for k,v in state.items()};candidate['linear.weight']=torch.from_numpy((W/T).astype(np.float32));candidate['linear.bias']=torch.from_numpy((bias/T).astype(np.float32));metrics={};saved={}
    for role in ('train','val'):metrics[role],saved[role]=head.predict(data[role],candidate['linear.weight'].numpy(),candidate['linear.bias'].numpy(),p['metrics'][role])
    native_baseline,_=head.predict(data['val'],W.astype(np.float32),bias.astype(np.float32),p['metrics']['val'])
    if metrics['val']['balanced_nll']>=native_baseline['balanced_nll']:
        T=1.;candidate=state
        for role in ('train','val'):metrics[role],saved[role]=head.predict(data[role],W.astype(np.float32),bias.astype(np.float32),p['metrics'][role])
    for role in ('train','val'):assert np.array_equal(saved[role]['probabilities'].argmax(1),data[role]['probabilities'].argmax(1)),'Confidence calibration must preserve classification on TRAIN/VAL';b.save_np(out/('selected_'+role+'.npz'),saved[role])
    assert head.frozen_digest(candidate)==head.frozen_digest(state);checkpoint=dict(identity(),model=candidate,epoch=120,optical_epochs_spent=120,temperature=T,calibration_uses_VAL_labels=True,contract=p['contract'],config=cfg,metrics=metrics,parent=p,actual_command=sys.argv,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'));engine.save_pt(out/'best.pt',checkpoint);engine.save_pt(out/'last.pt',dict(checkpoint,scalar_trace=trace,solver=dict(success=bool(result.success),nfev=int(result.nfev),log_temperature=float(result.x)),baseline_float64_balanced_VAL_NLL=baseline));s.save(out/'fit_result.json',dict(passed=True,actual_epochs=120,selected_epoch=120,temperature=T,unconstrained_native_temperature=float(np.exp(result.x)),scalar_objective_calls=len(trace),trainable_degrees_of_freedom=1,trainable_head_parameters_modified=5140,inference_parameters_added=0,calibration_uses_VAL_labels=True,additional_optical_epochs=0,optical_updates=0,fit_seconds=time.monotonic()-started,metrics=metrics,best_path=str(out/'best.pt'),best_sha256=s.sha(out/'best.pt'),last_sha256=s.sha(out/'last.pt'),selected_train_path=str(out/'selected_train.npz'),selected_val_path=str(out/'selected_val.npz'),frozen_digest=head.frozen_digest(state)),exclusive=True)
def test(a):
    root=a.out.parent;previous.receipt(root,'audit_completed.json','science');p=parent();f=s.read(root/'fit/fit_result.json');assert not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'folded_temperature_selected_once_test');a.out.mkdir();state=torch.load(f['best_path'],map_location='cpu',weights_only=False)['model'];digest=s.digest(state);s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),model_digest=digest,best_sha256=f['best_sha256'],selected_epoch=120,temperature=f['temperature']),exclusive=True)
    if digest in p['test_registry']:
        old=p['test_registry'][digest];assert old['predictions_sha256']==s.sha(old['predictions_path']);result=dict(metrics=old['metrics'],predictions_path=old['predictions_path'],predictions_sha256=old['predictions_sha256'],reused=True,new_test_evaluations=0)
    else:
        assert a.gpu in b.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);model=b.RegularizedMoE(6,.3).cuda();model.load_state_dict(state,strict=True);metrics,arrays=b.evaluate(model,b.load('test'));b.save_np(a.out/'predictions.npz',arrays);result=dict(metrics=metrics,predictions_path=str(a.out/'predictions.npz'),predictions_sha256=s.sha(a.out/'predictions.npz'),reused=False,new_test_evaluations=1)
    s.save(a.out/'result.json',dict(result,model_digest=digest,best_sha256=f['best_sha256'],selected_epoch=120,temperature=f['temperature'],test_used_for_selection=False,test_informed_development=True,VAL_used_for_calibration=True),exclusive=True)
def audit_child(root,phase):
    claim(root/(phase+'_audit_claim.json'),'own_fixed_LAST_temperature_'+phase);out=root/('science' if phase=='fit' else 'saved_test_audit')
    with (root/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_richsplit_tempcal.py'),'--base',str(root),'--out',str(out),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Own science failed: preserve, never retry';s.save(root/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/'audit.json')),exclusive=True)
def child(root,phase,gpu=''):
    with (root/(phase+'.log')).open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--out',str(root/phase),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);rc=proc.wait()
    s.save(root/(phase+'_exit.json'),dict(pid=proc.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Actual stage failure: no automatic retry'
def campaign(a):
    cpu_ready();p=parent();out=a.out;assert not out.exists() and out==HERE/'runs/simulation'/RUN and os.environ.get('CUDA_VISIBLE_DEVICES')=='';b.slots();claim(out.with_name(out.name+'.campaign_claim.json'),'fixed_LAST_EMA_temperature_campaign');out.mkdir();s.save(out/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),parent=p),exclusive=True);gpu=None;s.gpu_budget.ALLOWED.update(b.GPU.values())
    for card in (2,5,4,0):
        if b.slots()['counts'][b.GPU[card]]:continue
        try:s.gpu_budget.preflight(b.GPU[card],shared=True,memory_gib=2);gpu=b.GPU[card];break
        except AssertionError:continue
    assert gpu is not None;s.save(out/'status.json',dict(state='temperature_extract_running'));child(out,'extract',gpu);child(out,'fit');audit_child(out,'fit');s.save(out/'status.json',dict(state='temperature_test_running'));child(out,'test',gpu);audit_child(out,'test');f=s.read(out/'fit/fit_result.json');t=s.read(out/'test/result.json');control=s.read(previous.BASE/'once_tests/L6_rho0/result.json')['metrics'];four=s.read(previous.BASE/'once_tests/L4_rho03/result.json')['metrics'];s.save(out/'comparison.json',dict(actual_epochs=120,selected_epoch=120,temperature=f['temperature'],train=f['metrics']['train'],val=f['metrics']['val'],test=t['metrics'],control100=control,reference_four100=four,test_gap_pp=100*(t['metrics']['accuracy']-control['accuracy']),six_above_four=t['metrics']['accuracy']>four['accuracy'],temperature_does_not_create_accuracy_gain=True,VAL_used_for_calibration=True,test_informed_development=True),exclusive=True);s.save(out/'status.json',dict(state='fixed_LAST_temperature_science_test_complete',comparison_sha256=s.sha(out/'comparison.json')))
def supervise(a):
    cpu_ready();parent();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(a.out.with_name(a.out.name+'.launch_claim.json'),'finite_fixed_LAST_temperature_supervision')
    with a.out.with_name(a.out.name+'.campaign.log').open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(a.out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);s.save(a.out.with_name(a.out.name+'.started.json'),dict(identity(),supervisor_pid=os.getpid(),campaign_pid=proc.pid),exclusive=True);rc=proc.wait()
    path=a.out/'campaign_exit.json' if a.out.exists() else a.out.with_name(a.out.name+'.campaign_exit.json');s.save(path,dict(pid=proc.pid,supervisor_pid=os.getpid(),returncode=rc,actual_exit=True),exclusive=True);assert rc==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','extract','fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu');a=p.parse_args()
    try:cpu(a.out) if a.phase=='cpu' else fit(a.out) if a.phase=='fit' else globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
