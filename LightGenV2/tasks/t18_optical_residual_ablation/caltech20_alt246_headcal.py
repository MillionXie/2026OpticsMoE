"""CPU refinement of the existing six-layer linear head; frozen optical identity."""
import argparse,copy,hashlib,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import caltech20_alt246_genreg as g

HERE,ROOT,s,engine=g.HERE,g.ROOT,g.s,g.engine
PROFILE=HERE/'caltech20_alt246_headcal_profile.json'
FILES=('caltech20_alt246_headcal.py','caltech20_alt246_headcal_profile.json','audit_caltech20_alt246_headcal.py','CALTECH20_ALT246_HEADCAL_PROTOCOL_20261016.md')
RUN='caltech20_alt246_L6_same_linear_ridge_s17_20261016'
CPU=HERE/'runs/smoke/caltech20_alt246_headcal_cpu_20261016'

def config():
    p=s.read(PROFILE);assert p['case']=='L6_rho03' and p['maximum_head_updates']==20 and p['trainable_parameters']==5140 and p['new_inference_parameters']==0
    return p

def sources():
    src=g.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src

def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def admit():
    p=config();g.baseline();root=HERE/'runs/simulation'/g.RUN;state=s.read(root/'status.json');assert state['state']=='all_genreg_stopped_targets_or_review'
    pid=s.read(root/'metadata.json')['pid'];proc=Path('/proc',str(pid),'cmdline');assert not proc.exists() or g.RUN.encode() not in proc.read_bytes()
    for block,expected in p['regularization_receipts'].items():
        folder=root/'blocks'/block
        for phase,sub,rec in (('fit','science','audit_completed.json'),('test','saved_test_audit','test_audit_completed.json')):
            receipt=s.read(folder/rec);assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==expected[phase]==s.sha(folder/sub/'audit.json')
    for key in ('best','train','val','test'):assert s.sha(p['parent'][key+'_path'])==p['parent'][key+'_sha256']
    assert s.sha(p['reference_four']['best_path'])==p['reference_four']['best_sha256'] and s.read(p['reference_four']['test_result_path'])['metrics']['accuracy']==p['reference_four']['test_accuracy']
    return p

def load(role):
    assert role in ('train','val'),'No TEST features while fitting/admitting'
    p=config()
    with np.load(p['parent'][role+'_path'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
    manifest=g.b.old.manifest(Path(g.b.config()['data']));assert data['ids'].tolist()==manifest['split_ids'][role] and np.bincount(data['labels'],minlength=20).tolist()==manifest['supports'][role]
    assert data['features'].shape==(len(data['labels']),256)
    return data

def frozen_digest(state):return s.digest({k:v for k,v in state.items() if not k.startswith('linear.')})

def coordinates(data,parent,p):
    x=data['features'].astype(np.float64);mu=x.mean(0);sigma=np.maximum(x.std(0),p['feature_std_floor']);z=(x-mu)/sigma
    w=parent['linear.weight'].numpy().astype(np.float64);b=parent['linear.bias'].numpy().astype(np.float64)
    return z,mu,sigma,w*sigma[None],b+w@mu

def fold(w,b,mu,sigma):
    native=w/sigma[None];return native,b-native@mu

def target(y,smoothing,weights):return (F.one_hot(y,20).to(torch.float64)*(1-smoothing)+smoothing/20)*weights[None]

def objective(head,z,t,regularization):return -(t*F.log_softmax(head(z),1)).sum(1).mean()+regularization*.5*head.weight.square().sum()

def predict(data,w,b,alphas):
    with torch.no_grad():prob=F.softmax(F.linear(torch.from_numpy(data['features']).float(),torch.from_numpy(w).float(),torch.from_numpy(b).float()),1).numpy().astype(np.float64)
    metrics=engine.metrics(data['labels'],prob);metrics['alphas']=alphas
    saved={k:v for k,v in data.items() if k!='probabilities'};saved['probabilities']=prob
    return metrics,saved

def cpu(out):
    assert out==CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';p=admit();assert g.b.slots()['total']==0;claim(out.with_name(out.name+'.claim.json'),'same_head_CPU_math');torch.set_num_threads(4)
    data=load('train');val=load('val');parent=torch.load(p['parent']['best_path'],map_location='cpu',weights_only=False);state=parent['model'];z,mu,sigma,w,b=coordinates(data,state,p);native,bias=fold(w,b,mu,sigma)
    assert parent['contract']['main_depth']==6 and parent['contract']['rho']==.3 and parent['contract']['terminal_linear_parameters']==5140
    assert np.allclose(z@w.T+b,data['features'].astype(np.float64)@native.T+bias,atol=1e-10,rtol=0)
    metrics,saved=predict(data,state['linear.weight'].numpy(),state['linear.bias'].numpy(),parent['metrics']['val']['alphas']);assert np.abs(saved['probabilities']-data['probabilities']).max()<3e-6
    idx=np.asarray([np.flatnonzero(data['labels']==k)[0] for k in range(20)]);Z=torch.from_numpy(z[idx]);y=torch.from_numpy(data['labels'][idx]).long();weights=len(data['labels'])/(20*np.bincount(data['labels'],minlength=20));tw=torch.from_numpy(weights)
    head=torch.nn.Linear(256,20).double();head.weight.data.copy_(torch.from_numpy(w));head.bias.data.copy_(torch.from_numpy(b));t=target(y,p['label_smoothing'],tw);loss=objective(head,Z,t,.002);loss.backward()
    logits=z[idx]@w.T+b;prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);T=(np.eye(20)[data['labels'][idx]]*(1-p['label_smoothing'])+p['label_smoothing']/20)*weights[None];delta=(prob*T.sum(1,keepdims=True)-T)/20
    assert np.allclose(delta.T@z[idx]+.002*w,head.weight.grad.numpy(),atol=1e-10,rtol=0) and np.allclose(delta.sum(0),head.bias.grad.numpy(),atol=1e-10,rtol=0)
    assert all(torch.isfinite(q.grad).all() and q.grad.norm()>0 for q in head.parameters()) and sum(q.numel() for q in head.parameters())==5140
    out.mkdir();s.save(out/'checks.json',dict(identity(),passed=True,linear_parameters=5140,inference_parameters_added=0,optimizer_steps=0,optical_model_forwards=0,test_arrays_read=0,parent_frozen_digest=frozen_digest(state),folding_maxabs=float(np.abs((z@w.T+b)-(data['features'].astype(np.float64)@native.T+bias)).max()),parent_probability_maxabs=float(np.abs(saved['probabilities']-data['probabilities']).max()),weighted_smoothed_gradient_analytic=True,statistics_role='TRAIN only'),exclusive=True)

def cpu_ready():
    proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();engine.verify_sources(proof['sources'],proof['git_commit'])

def fit(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();p=admit();cpu_ready();claim(out.with_name(out.name+'.claim.json'),'same_linear_head_fit');out.mkdir();torch.set_num_threads(4)
    data=load('train');val=load('val');parent=torch.load(p['parent']['best_path'],map_location='cpu',weights_only=False);state=parent['model'];frozen=frozen_digest(state);z,mu,sigma,w0,b0=coordinates(data,state,p)
    head=torch.nn.Linear(256,20).double();head.weight.data.copy_(torch.from_numpy(w0));head.bias.data.copy_(torch.from_numpy(b0));Z=torch.from_numpy(z);y=torch.from_numpy(data['labels']).long();weights=torch.from_numpy(len(y)/(20*np.bincount(data['labels'],minlength=20)));t=target(y,p['label_smoothing'],weights)
    initial_ce=float((-(t*F.log_softmax(head(Z),1)).sum(1).mean()).detach());regularization=float(np.clip(2*p['initial_penalty_fraction']*initial_ce/np.square(w0).sum(),*p['l2_bounds']))
    optimizer=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=1,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12);ema_w=w0.copy();ema_b=b0.copy();history=[];closures=0;best=parent['metrics']['val']['balanced_nll'];selected=0;best_path=p['parent']['best_path'];train_path=p['parent']['train_path'];val_path=p['parent']['val_path'];best_metrics=parent['metrics'];alphas=parent['metrics']['val']['alphas'];stop='maximum_head_updates';started=time.monotonic()
    s.save(out/'metadata.json',dict(identity(),pid=os.getpid(),config=p,frozen_digest=frozen,regularization=regularization,initial_weighted_ce=initial_ce,optimizer='CPU float64 fullbatch LBFGS, raw variable in standardized training coordinates; native W/b folded'),exclusive=True)
    def closure():
        nonlocal closures
        closures+=1;optimizer.zero_grad(set_to_none=True);value=objective(head,Z,t,regularization);assert torch.isfinite(value);value.backward();assert all(torch.isfinite(q.grad).all() for q in head.parameters());return value
    for step in range(1,p['maximum_head_updates']+1):
        before=s.digest(head.state_dict());optimizer.step(closure)
        if s.digest(head.state_dict())==before:stop='optimizer_no_parameter_change';break
        raw_w=head.weight.detach().numpy().copy();raw_b=head.bias.detach().numpy().copy();ema_w=p['head_ema']*ema_w+(1-p['head_ema'])*raw_w;ema_b=p['head_ema']*ema_b+(1-p['head_ema'])*raw_b;native,bias=fold(ema_w,ema_b,mu,sigma);native=native.astype(np.float32);bias=bias.astype(np.float32)
        tm,tr=predict(data,native,bias,alphas);vm,vr=predict(val,native,bias,alphas)
        row=dict(step=step,raw_w=torch.from_numpy(raw_w),raw_b=torch.from_numpy(raw_b),ema_w=torch.from_numpy(ema_w.copy()),ema_b=torch.from_numpy(ema_b.copy()),train=tm,val=vm,train_probabilities=torch.from_numpy(tr['probabilities'].astype(np.float32)),val_probabilities=torch.from_numpy(vr['probabilities'].astype(np.float32)),raw_objective=float(objective(head,Z,t,regularization).detach()),optimizer_n_iter=int(optimizer.state[head.weight]['n_iter']),closure_calls=closures)
        history.append(row)
        if vm['balanced_nll']<best-p['selection_tie_tolerance']:
            best=vm['balanced_nll'];selected=step;candidate=dict(state);candidate['linear.weight']=torch.from_numpy(native);candidate['linear.bias']=torch.from_numpy(bias);assert frozen_digest(candidate)==frozen
            best_path=str(out/'best.pt');train_path=str(out/'selected_train.npz');val_path=str(out/'selected_val.npz');best_metrics=dict(train=tm,val=vm)
            engine.save_pt(Path(best_path),dict(model=candidate,epoch=p['parent']['selected_epoch'],head_step=step,contract=parent['contract'],config=p,sources=sources(),metrics=best_metrics,parent_sha256=p['parent']['best_sha256'],frozen_digest=frozen));g.b.save_np(Path(train_path),tr);g.b.save_np(Path(val_path),vr)
    engine.save_pt(out/'head_history.pt',dict(history=history,mu=torch.from_numpy(mu),sigma=torch.from_numpy(sigma),initial_w=torch.from_numpy(w0),initial_b=torch.from_numpy(b0),optimizer=optimizer.state_dict()))
    s.save(out/'fit_result.json',dict(passed=True,head_updates=len(history),selected_head_step=selected,selected_optical_epoch=p['parent']['selected_epoch'],optical_model_forwards=0,optical_updates=0,trainable_parameters=5140,regularization=regularization,closure_calls=closures,stop_reason=stop,fit_seconds=time.monotonic()-started,best_path=best_path,best_sha256=s.sha(best_path),selected_train_path=train_path,selected_val_path=val_path,metrics=best_metrics,frozen_digest=frozen,history_sha256=s.sha(out/'head_history.pt')),exclusive=True)

def test(root):
    p=admit();f=s.read(root/'fit/fit_result.json');receipt=s.read(root/'audit_completed.json');assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(root/'science/audit.json');out=root/'test';assert not out.exists();claim(root/'test_claim.json','same_head_once_test_or_true_reuse');out.mkdir()
    if f['selected_head_step']==0:
        with np.load(p['parent']['test_path'],allow_pickle=False) as z:prob=z['probabilities'];labels=z['labels']
        result=dict(metrics=engine.metrics(labels,prob),predictions_path=p['parent']['test_path'],predictions_sha256=p['parent']['test_sha256'],reused=True,new_test_evaluations=0,optical_forwards=0,head_forward_batches=0)
    else:
        with np.load(p['parent']['test_path'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
        checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);state=checkpoint['model'];metrics,saved=predict(data,state['linear.weight'].numpy(),state['linear.bias'].numpy(),f['metrics']['val']['alphas']);g.b.save_np(out/'predictions.npz',saved)
        result=dict(metrics=metrics,predictions_path=str(out/'predictions.npz'),predictions_sha256=s.sha(out/'predictions.npz'),reused=False,new_test_evaluations=1,optical_forwards=0,head_forward_batches=1)
    s.save(out/'result.json',dict(result,best_sha256=f['best_sha256'],selected_head_step=f['selected_head_step'],test_informed_development=True,test_used_for_update_selection=False,reference_four=p['reference_four']),exclusive=True)

def campaign(out):
    assert out==HERE/'runs/simulation'/RUN and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';admit();cpu_ready();assert g.b.slots()['total']==0;claim(out.with_name(out.name+'.campaign_claim.json'),'same_linear_head_campaign');out.mkdir();s.save(out/'metadata.json',dict(identity(),pid=os.getpid(),config=config()),exclusive=True)
    with (out/'fit.log').open('x') as log:child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','fit','--out',str(out/'fit')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);rc=child.wait()
    s.save(out/'fit_exit.json',dict(pid=child.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0
    for phase,folder in (('fit','science'),('test','saved_test_audit')):
        if phase=='test':test(out)
        claim(out/(phase+'_audit_claim.json'),'independent_same_head_'+phase)
        with (out/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_headcal.py'),'--base',str(out),'--out',str(out/folder),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0 and s.read(out/folder/'audit.json')['passed'],'Own head audit failed; preserve and stop'
        s.save(out/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/folder/'audit.json')),exclusive=True)
    result=s.read(out/'test/result.json');p=config();s.save(out/'status.json',dict(state='same_head_fit_science_test_complete',test=result['metrics'],selected_head_step=result['selected_head_step'],target_six_above_four=result['metrics']['accuracy']>p['reference_four']['test_accuracy'],performance_maintained=result['metrics']['accuracy']>=p['parent']['test_accuracy']),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','fit','campaign'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:globals()[a.phase](a.out)
    except BaseException as e:
        path=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not path.exists():s.save(path,dict(error=repr(e),automatic_retry=False,pid=os.getpid()),exclusive=True)
        raise
