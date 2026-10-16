"""Same5140-parameter head, natural-distribution goal; immutable optical backbone."""
import argparse,copy,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import caltech20_alt246_richsplit_viewreg as v
import caltech20_alt246_richsplit_tempcal as adapted
HERE,ROOT,s,engine,b=v.HERE,v.ROOT,v.s,v.engine,v.b
PROFILE=HERE/'caltech20_alt246_richsplit_headanchor_profile.json'
FILES=('caltech20_alt246_richsplit_headanchor.py','caltech20_alt246_richsplit_headanchor_profile.json','audit_caltech20_alt246_richsplit_headanchor.py','CALTECH20_ALT246_RICHSPLIT_HEADANCHOR_PROTOCOL_20261017.md')
RUN='caltech20_rich20_L6_same_head_anchor01_s17_20261017'
CPU=HERE/'runs/smoke/caltech20_richsplit_headanchor_cpu_20261017'
BASE=HERE/'runs/simulation'/adapted.RUN
REFERENCE=adapted.previous.BASE

def config():
    p=s.read(PROFILE);assert p['maximum_head_updates']==20 and p['case']=='L6_rho03' and p['trainable_parameters']==5140 and p['new_inference_parameters']==0 and p['automatic_retry'] is False;return p

def sources():
    src=adapted.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src

def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def frozen_digest(state):return s.digest({k:value for k,value in state.items() if not k.startswith('linear.')})

def admit():
    p=config();assert s.read(BASE/'status.json')['state']=='fixed_LAST_temperature_science_test_complete';meta=s.read(BASE/'metadata.json');assert meta['sources']==adapted.sources();engine.verify_sources(meta['sources'],meta['git_commit']);assert v.receipt(BASE,'audit_completed.json','science')==p['parent_science_sha256'] and v.receipt(BASE,'test_audit_completed.json','saved_test_audit')==p['parent_test_audit_sha256'];ex=s.read(BASE/'campaign_exit.json');assert ex['returncode']==0 and ex['actual_exit'];proc=Path('/proc',str(meta['pid']),'cmdline');assert not proc.exists() or str(BASE).encode() not in proc.read_bytes();f=s.read(BASE/'fit/fit_result.json');test=s.read(BASE/'test/result.json');assert f['best_sha256']==p['parent_checkpoint_sha256']==s.sha(f['best_path']) and test['best_sha256']==f['best_sha256'] and test['predictions_sha256']==s.sha(test['predictions_path']);registry=dict(adapted.parent()['test_registry']);registry[test['model_digest']]=dict(result_path=str(BASE/'test/result.json'),predictions_path=test['predictions_path'],predictions_sha256=test['predictions_sha256'],metrics=test['metrics']);cfg=b.config();assert cfg['cache_sha256']==p['cache_sha256'] and cfg['manifest_sha256']==p['manifest_sha256'];return dict(actual_epochs=120,selected_epoch=120,best=f['best_path'],train=f['selected_train_path'],val=f['selected_val_path'],metrics=f['metrics'],test_registry=registry,ancestor_VAL_temperature_calibration=True)

def load(role):
    assert role in ('train','val');parent=admit();m=b.dataset.manifest();path=Path(parent[role]);audit=s.read(BASE/'science/audit.json');assert audit['checks'][role]['prediction_sha256']==s.sha(path)
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    assert data['ids'].tolist()==m['split_ids'][role] and np.bincount(data['labels'],minlength=20).tolist()==m['supports'][role] and data['features'].shape==(len(data['labels']),256);return data

def targets(y,smoothing):return F.one_hot(y,20).to(torch.float64)*(1-smoothing)+smoothing/20
def objective(head,x,y,w0,b0,p):return -(y*F.log_softmax(head(x),1)).sum(1).mean()+.5*p['anchor_l2']*((head.weight-w0).square().sum()+(head.bias-b0).square().sum())

def predict(data,w,bias,reference):
    with torch.no_grad():prob=F.softmax(F.linear(torch.from_numpy(data['features']).float(),torch.from_numpy(w).float(),torch.from_numpy(bias).float()),1).numpy().astype(np.float64)
    metrics=engine.metrics(data['labels'],prob)
    for key in ('alphas','detector_capture','routing_weight_mean','routing_weight_std'):
        if key in reference:metrics[key]=reference[key]
    saved={k:value for k,value in data.items() if k!='probabilities'};saved['probabilities']=prob;return metrics,saved

def cpu(out):
    assert out==CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';parent=admit();claim(out.with_name(out.name+'.claim.json'),'natural_same_head_CPU_math');torch.set_num_threads(4);p=config();state=torch.load(parent['best'],map_location='cpu',weights_only=False)['model'];data,val=load('train'),load('val')
    _,saved=predict(data,state['linear.weight'].numpy(),state['linear.bias'].numpy(),parent['metrics']['train']);err=float(np.abs(saved['probabilities']-data['probabilities']).max());assert err<=3e-6
    torch.manual_seed(p['seed']);head=torch.nn.Linear(256,20).double();w0=head.weight.detach().clone()+.01;b0=head.bias.detach().clone()-.02;x=torch.randn(20,256,dtype=torch.float64);y=targets(torch.arange(20),p['label_smoothing']);value=objective(head,x,y,w0,b0,p);value.backward()
    W=head.weight.detach().numpy();B=head.bias.detach().numpy();logits=x.numpy()@W.T+B;prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);delta=(prob-y.numpy())/20;gw=delta.T@x.numpy()+p['anchor_l2']*(W-w0.numpy());gb=delta.sum(0)+p['anchor_l2']*(B-b0.numpy());graderr=max(float(np.abs(gw-head.weight.grad.numpy()).max()),float(np.abs(gb-head.bias.grad.numpy()).max()));assert graderr<=1e-10 and all(torch.isfinite(q.grad).all() and q.grad.norm()>0 for q in head.parameters()) and sum(q.numel() for q in head.parameters())==5140
    full={n:q.grad.clone() for n,q in head.named_parameters()};head.zero_grad(set_to_none=True)
    for off in range(20):(objective(head,x[off:off+1],y[off:off+1],w0,b0,p)/20).backward()
    microerr=max(float((q.grad-full[n]).abs().max()) for n,q in head.named_parameters());assert microerr<=1e-10
    out.mkdir();s.save(out/'checks.json',dict(identity(),passed=True,parent_sha256=s.sha(parent['best']),parent_frozen_digest=frozen_digest(state),trainable_parameters=5140,inference_parameters_added=0,parent_probability_maxabs=err,analytic_gradient_maxabs=graderr,micro1_vs20_gradient_maxabs=microerr,optical_model_objects=0,optical_forwards=0,optimizer_updates=0,TEST_images_read=0,TEST_feature_arrays_read=0),exclusive=True)

def cpu_ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();engine.verify_sources(q['sources'],q['git_commit'])

def fit(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();parent=admit();cpu_ready();p=config();claim(out.with_name(out.name+'.claim.json'),'natural_same_linear_fit');out.mkdir();torch.set_num_threads(4);data,val=load('train'),load('val');checkpoint=torch.load(parent['best'],map_location='cpu',weights_only=False);state=checkpoint['model'];frozen=frozen_digest(state)
    head=torch.nn.Linear(256,20).double();head.weight.data.copy_(state['linear.weight'].double());head.bias.data.copy_(state['linear.bias'].double());w0=head.weight.detach().clone();b0=head.bias.detach().clone();x=torch.from_numpy(data['features']).double();y=targets(torch.from_numpy(data['labels']).long(),p['label_smoothing']);optimizer=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=1,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12);history=[];closures=0;best=parent['metrics']['val']['nll'];selected=0;chosen=dict(best_path=parent['best'],train_path=parent['train'],val_path=parent['val'],metrics=parent['metrics']);started=time.monotonic()
    s.save(out/'metadata.json',dict(identity(),pid=os.getpid(),config=p,parent=parent,parent_sha256=s.sha(parent['best']),frozen_digest=frozen,initial_head_digest=s.digest({k:q for k,q in state.items() if k.startswith('linear.')}),optical_forwards=0),exclusive=True);attempts=0;stop='maximum20_head_updates'
    def closure():
        nonlocal closures
        closures+=1;optimizer.zero_grad(set_to_none=True);loss=objective(head,x,y,w0,b0,p);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(q.grad).all() for q in head.parameters());return loss
    for step in range(1,21):
        assert time.monotonic()-started<p['maximum_seconds'];before=s.digest(head.state_dict());optimizer.step(closure);attempts+=1
        if s.digest(head.state_dict())==before:stop='optimizer_no_parameter_change';break
        raw_w=head.weight.detach().clone();raw_b=head.bias.detach().clone();native=raw_w.float().numpy();bias=raw_b.float().numpy();tm,tr=predict(data,native,bias,parent['metrics']['train']);vm,vr=predict(val,native,bias,parent['metrics']['val']);row=dict(step=step,raw_w=raw_w,raw_b=raw_b,train=tm,val=vm,train_probabilities=torch.from_numpy(tr['probabilities'].astype(np.float32)),val_probabilities=torch.from_numpy(vr['probabilities'].astype(np.float32)),objective=float(objective(head,x,y,w0,b0,p).detach()),optimizer_n_iter=int(optimizer.state[head.weight]['n_iter']),closure_calls=closures);history.append(row)
        if vm['nll']<best:
            best=vm['nll'];selected=step;candidate=dict(state);candidate['linear.weight']=torch.from_numpy(native.copy());candidate['linear.bias']=torch.from_numpy(bias.copy());assert frozen_digest(candidate)==frozen;chosen=dict(best_path=str(out/'best.pt'),train_path=str(out/'selected_train.npz'),val_path=str(out/'selected_val.npz'),metrics=dict(train=tm,val=vm));engine.save_pt(Path(chosen['best_path']),dict(model=candidate,epoch=parent['selected_epoch'],optical_epochs_spent=120,head_step=step,sources=sources(),config=p,contract=checkpoint['contract'],metrics=chosen['metrics'],parent_sha256=s.sha(parent['best']),frozen_digest=frozen));b.save_np(Path(chosen['train_path']),tr);b.save_np(Path(chosen['val_path']),vr)
        s.save(out/'status.json',dict(state='head_training',head_step=step,train=tm,val=vm,selected_head_step=selected))
    last_model=dict(state);last_model['linear.weight']=head.weight.detach().float().clone();last_model['linear.bias']=head.bias.detach().float().clone();engine.save_pt(out/'last.pt',dict(model=last_model,history=history,optimizer=optimizer.state_dict(),initial_w=w0,initial_b=b0,sources=sources(),config=p));identity();s.save(out/'fit_result.json',dict(passed=True,optical_epochs_spent=120,selected_optical_epoch=parent['selected_epoch'],head_updates=len(history),head_attempts=attempts,selected_head_step=selected,trainable_parameters=5140,inference_parameters_added=0,optical_forwards=0,optical_updates=0,closure_calls=closures,fit_seconds=time.monotonic()-started,stop_reason=stop,best_path=chosen['best_path'],best_sha256=s.sha(chosen['best_path']),selected_train_path=chosen['train_path'],selected_val_path=chosen['val_path'],metrics=chosen['metrics'],frozen_digest=frozen,last_sha256=s.sha(out/'last.pt')),exclusive=True)

def test(a):
    root=a.out.parent;v.receipt(root,'audit_completed.json','science');parent=admit();f=s.read(root/'fit/fit_result.json');assert not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'natural_head_selected_once_test');a.out.mkdir();checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);digest=s.digest(checkpoint['model']);assert frozen_digest(checkpoint['model'])==f['frozen_digest']
    s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),selected_head_step=f['selected_head_step'],model_digest=digest,best_sha256=f['best_sha256']),exclusive=True)
    if digest in parent['test_registry']:
        known=parent['test_registry'][digest];assert s.sha(known['predictions_path'])==known['predictions_sha256'];result=dict(metrics=known['metrics'],predictions_path=known['predictions_path'],predictions_sha256=known['predictions_sha256'],reused=True,new_test_evaluations=0,new_optical_TEST_evaluations=0)
    else:
        assert a.gpu in b.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);model=b.RegularizedMoE(6,.3).cuda();model.load_state_dict(checkpoint['model'],strict=True);metrics,arrays=b.evaluate(model,b.load('test'));b.save_np(a.out/'predictions.npz',arrays);result=dict(metrics=metrics,predictions_path=str(a.out/'predictions.npz'),predictions_sha256=s.sha(a.out/'predictions.npz'),reused=False,new_test_evaluations=1,new_optical_TEST_evaluations=1,policy=policy)
    s.save(a.out/'result.json',dict(result,model_digest=digest,best_sha256=f['best_sha256'],selected_head_step=f['selected_head_step'],test_used_for_epoch_selection=False,test_informed_development=True),exclusive=True)

def audit_child(root,phase):
    claim(root/(phase+'_audit_claim.json'),'own_natural_head_'+phase+'_saved_math');out=root/('science' if phase=='fit' else 'saved_test_audit')
    with (root/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_richsplit_headanchor.py'),'--base',str(root),'--out',str(out),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Own saved audit failed: preserve, no retry';s.save(root/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/'audit.json')),exclusive=True)

def campaign(a):
    out=a.out;assert out==HERE/'runs/simulation'/RUN and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';admit();cpu_ready();b.slots();claim(out.with_name(out.name+'.campaign_claim.json'),'same_existing_head_natural_campaign');out.mkdir();s.save(out/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),parent=admit()),exclusive=True)
    with (out/'fit.log').open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','fit','--out',str(out/'fit')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);rc=proc.wait()
    s.save(out/'fit_exit.json',dict(pid=proc.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0;v.receipt(BASE,'audit_completed.json','science');audit_child(out,'fit');s.gpu_budget.ALLOWED.update(b.GPU.values());gpu=None
    while gpu is None:
        for card in (2,5,4,0):
            if b.slots()['counts'][b.GPU[card]]:continue
            try:s.gpu_budget.preflight(b.GPU[card],shared=True,memory_gib=2);gpu=b.GPU[card];break
            except AssertionError:continue
        if gpu is None:time.sleep(5)
    with (out/'test.log').open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','test','--out',str(out/'test'),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);rc=proc.wait()
    s.save(out/'test_exit.json',dict(pid=proc.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0;audit_child(out,'test');f=s.read(out/'fit/fit_result.json');t=s.read(out/'test/result.json');control=s.read(REFERENCE/'once_tests/L6_rho0/result.json')['metrics'];four=s.read(REFERENCE/'once_tests/L4_rho03/result.json')['metrics'];s.save(out/'comparison.json',dict(optical_epochs_spent=120,selected_optical_epoch=f['selected_optical_epoch'],head_updates=f['head_updates'],selected_head_step=f['selected_head_step'],train=f['metrics']['train'],val=f['metrics']['val'],test=t['metrics'],control100=control,reference_four100=four,test_gap_pp=100*(t['metrics']['accuracy']-control['accuracy']),six_above_four=t['metrics']['accuracy']>four['accuracy'],test_informed_development=True,extra_electronic_objective_not_pure_residual_causality=True),exclusive=True);s.save(out/'status.json',dict(state='same_natural_head_science_test_complete',comparison_sha256=s.sha(out/'comparison.json')))

def supervise(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';admit();cpu_ready();claim(a.out.with_name(a.out.name+'.launch_claim.json'),'finite_natural_head_supervision')
    with a.out.with_name(a.out.name+'.campaign.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(a.out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);s.save(a.out.with_name(a.out.name+'.started.json'),dict(identity(),supervisor_pid=os.getpid(),campaign_pid=proc.pid),exclusive=True);rc=proc.wait()
    path=a.out/'campaign_exit.json' if a.out.exists() else a.out.with_name(a.out.name+'.campaign_exit.json');s.save(path,dict(pid=proc.pid,supervisor_pid=os.getpid(),returncode=rc,actual_exit=True),exclusive=True);assert rc==0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu');a=p.parse_args()
    try:cpu(a.out) if a.phase=='cpu' else fit(a.out) if a.phase=='fit' else globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
