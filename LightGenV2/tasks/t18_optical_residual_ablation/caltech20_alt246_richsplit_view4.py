"""Four-layer view invariance from the accepted calibrated head, cached TRAIN selfteacher."""
import argparse,copy,hashlib,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_selfview2 as prior
import caltech20_alt246_richsplit_crossview6 as support
import caltech20_alt246_richsplit_fresh_head as head6
fresh=prior.fresh
b,k,v,g,s,engine,HERE,ROOT=fresh.b,fresh.k,fresh.v,fresh.g,fresh.s,fresh.engine,fresh.HERE,fresh.ROOT
RegularizedMoE=fresh.RegularizedMoE
PROFILE=HERE/'caltech20_alt246_richsplit_view4_profile.json'
FILES=('caltech20_alt246_richsplit_view4.py','caltech20_alt246_richsplit_view4_profile.json','check_caltech20_alt246_richsplit_view4.py','audit_caltech20_alt246_richsplit_view4.py','CALTECH20_ALT246_RICHSPLIT_VIEW4_PROTOCOL_20261017.md')
RUN='caltech20_rich20_L4_view_sqrt_selfteacher_s17_tune20_20261017'
CPU=HERE/'runs/smoke/caltech20_richsplit_view4_cpu_20261017'
OUT=HERE/'runs/simulation'/RUN
receipt=k.receipt
def sources():
    q=support.sources()
    for name in FILES:q[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return q
def identity():
    q=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(q,commit);return dict(sources=q,git_commit=commit)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def config():
    p=s.read(PROFILE);assert p['additional_epochs']==20 and p['maximum_seconds']==3600 and p['depth']==4 and p['rho']==.3 and p['automatic_retry'] is False;q=b.config();q.update(p['training']);q.update(additional_epochs=20,maximum_seconds=3600,selection=p['selection'],class_weight_exponent=.5);return q
def completed(root,module):
    meta=s.read(root/'metadata.json');assert meta['sources']==module.sources();engine.verify_sources(meta['sources'],meta['git_commit']);receipt(root,'audit_completed.json','science');receipt(root,'test_audit_completed.json','saved_test_audit');ex=s.read(root/'campaign_exit.json');assert ex['returncode']==0 and ex['actual_exit']
    for pid in (ex['pid'],ex['supervisor_pid']):
        path=Path('/proc',str(pid),'cmdline');assert not path.exists() or str(root).encode() not in path.read_bytes()
    f=s.read(root/'fit/fit_result.json');t=s.read(root/'test/result.json');assert f['best_sha256']==s.sha(f['best_path']) and t['predictions_sha256']==s.sha(t['predictions_path']);return f,t,meta

def teacher_info():
    module=prior.prior;root=HERE/'runs/simulation'/module.RUN;f,t,meta=completed(root,module);p=s.read(PROFILE);assert f['best_sha256']==p['teacher_checkpoint_sha256'] and s.sha(f['selected_train_path'])==p['teacher_TRAIN_prediction_sha256'] and f['selected_head_step']==4;return dict(checkpoint=f['best_path'],checkpoint_sha256=f['best_sha256'],train=f['selected_train_path'],TRAIN_prediction_sha256=s.sha(f['selected_train_path']),source_commit=meta['git_commit'])

def teacher_data():
    info=teacher_info();manifest=b.dataset.manifest()
    with np.load(info['train'],allow_pickle=False) as z:probs=z['probabilities'].copy();labels=z['labels'].copy();ids=z['ids'].copy()
    assert ids.tolist()==manifest['split_ids']['train'] and np.bincount(labels,minlength=20).tolist()==manifest['supports']['train'];return probs,labels

def parent():
    p=s.read(PROFILE);module=prior.prior;root=HERE/'runs/simulation'/module.RUN;f,t,meta=completed(root,module);assert f['optical_epochs_spent']==100 and f['selected_optical_epoch']==86 and f['head_updates']==20 and f['selected_head_step']==4 and f['best_sha256']==p['parent_checkpoint_sha256'];registry=dict(module.admit()['test_registry']);registry[t['model_digest']]=dict(result_path=str(root/'test/result.json'),predictions_path=t['predictions_path'],predictions_sha256=t['predictions_sha256'],metrics=t['metrics'])
    return dict(best=f['best_path'],best_sha256=f['best_sha256'],train=f['selected_train_path'],val=f['selected_val_path'],metrics=f['metrics'],actual_optical_epochs_spent=100,selected_optical_epoch=86,inherited_head_updates=20,selected_head_step=4,parent_sources=meta['sources'],teacher=teacher_info(),test_registry=registry)

def class_weights(data,device):return (len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).sqrt().to(device)

def cpu_ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['parent']==parent();engine.verify_sources(q['sources'],q['git_commit'])
def gate(state,data,cfg,device):
    model=RegularizedMoE(4,.3).to(device);model.load_state_dict(state,strict=True);initial=s.digest(model.state_dict());weights=class_weights(data,device);teacher=teacher_data();order,theta=engine.epoch_plan(101,len(data[1]),cfg);checks=[]
    for name,idx,batch in [('all20',np.array([int((data[1]==i).nonzero()[0]) for i in range(20)]),0),('macro16',order[:16],0),('tail5',order[-5:],196)]:
        x,y,mask,_=v.mixed(data,idx,theta,101,batch,cfg,device);target,eligible=k.kd_targets(teacher,idx,101,batch,cfg);qt=torch.as_tensor(target,device=device);gt=torch.as_tensor(eligible,device=device)
        for method in ('ordinary','strong_view_cached_selfteacher'):
            used=None if method=='ordinary' else mask;a,c,ga=k.gradients(model,x,y,used,weights,cfg,1,None if method=='ordinary' else qt,None if method=='ordinary' else gt)
            if method=='ordinary':z,d,gb=k.gradients(model,x,y,used,weights,cfg,4)
            else:z,d=fresh.training_backward(model,x,y,mask,weights,cfg,qt,gt);gb={n:p.grad.detach().cpu().numpy().astype(np.float64) for n,p in model.named_parameters()}
            norms={n:float(np.linalg.norm(a.ravel())) for n,a in gb.items()};assert len(norms)==21 and all(np.isfinite(a).all() and norms[n]>0 for n,a in gb.items());rel=max(float(np.linalg.norm((ga[n]-a).ravel())/norms[n]) for n,a in gb.items());cos=min(float(ga[n].ravel()@a.ravel()/(np.linalg.norm(ga[n])*norms[n])) for n,a in gb.items());err=max(float((a-z).abs().max()),float((c-d).abs().max()));assert err<=2e-6 and rel<=.02 and cos>=.999,(name,method,err,rel,cos);checks.append(dict(samples=name,method=method,n=len(idx),maxabs=err,gradient_relative_l2=rel,gradient_cosine=cos,gradient_norms=norms,teacher_eligible=int(eligible.sum())))
    assert s.digest(model.state_dict())==initial;del model;torch.cuda.empty_cache();return dict(passed=True,initial_digest=initial,checks=checks,optimizer_steps=0,TEST_images_read=0)
def fit(a):
    cpu_ready();p=parent();cfg=config();assert a.out==OUT/'fit' and not a.out.exists() and a.gpu==b.GPU[2] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'selected_parent_new_optimizer_strong_view_selftargets');a.out.mkdir();s.gpu_budget.ALLOWED.update(b.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);checkpoint=torch.load(p['best'],map_location='cpu',weights_only=False);data,val=b.load('train'),b.load('val');teacher=teacher_data();proof=gate(checkpoint['model'],data,cfg,'cuda');s.save(a.out/'numeric_gate.json',proof,exclusive=True);model=RegularizedMoE(4,.3).cuda();model.load_state_dict(checkpoint['model'],strict=True);ema=copy.deepcopy(model).eval();phase=[];electronic=[]
    for n,w in model.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(w)
    opt=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr']);assert len(opt.state)==0;meta=dict(identity(),pid=os.getpid(),depth=4,parent=p,config=cfg,gpu_uuid=a.gpu,policy=policy,initial_digest=s.digest(model.state_dict()),optimizer_restored=False,initial_optimizer_states=0,teacher_forward_count=0);s.save(a.out/'metadata.json',meta,exclusive=True);history=[dict(tune_epoch=0,augmentation_epoch=100,train=p['metrics']['train'],val=p['metrics']['val'],EMA_digest=s.digest(checkpoint['model']))];key=(-p['metrics']['val']['accuracy'],p['metrics']['val']['nll'],0);chosen=dict(tune_epoch=0,best_path=p['best'],train=p['train'],val=p['val'],metrics=p['metrics']);trace={};steps=0;started=time.monotonic();weights=class_weights(data,'cuda')
    for tune in range(1,21):
        assert time.monotonic()-started<3600,'Incomplete20: retain/no retry';epoch=100+tune;begin=time.monotonic();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);record={n:[] for n in ('mix','flip','mask','view','pixels','teacher_target','teacher_gate')};count=0;support=np.zeros(20,np.int64);kl=0.;predictions=[];model.train()
        for batch,off in enumerate(range(0,len(order),16)):
            idx=order[off:off+16];x,y,mask,hashes=v.mixed(data,idx,theta,epoch,batch,cfg,'cuda');target,eligible=k.kd_targets(teacher,idx,epoch,batch,cfg);hashes.update(teacher_target=s.digest(target),teacher_gate=s.digest(eligible));count+=int(eligible.sum());support+=np.bincount(data[1][idx].numpy(),weights=eligible,minlength=20).astype(np.int64)
            for name,value in hashes.items():record[name].append(value)
            prob,_=fresh.training_backward(model,x,y,mask,weights,cfg,torch.as_tensor(target,device='cuda'),torch.as_tensor(eligible,device='cuda'));raw=prob.cpu().numpy().astype(np.float64);predictions.append(torch.from_numpy(raw.astype(np.float32)));kl+=k.kl_sum(raw,target,eligible,cfg);torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);previous={n:w.detach().cpu().clone() for n,w in ema.named_parameters()} if tune==20 and batch==196 else None;opt.step();steps+=1
            with torch.no_grad():
                for e,w in zip(ema.parameters(),model.parameters()):e.mul_(cfg['ema_decay']).add_(w,alpha=1-cfg['ema_decay'])
        trace[tune]=torch.cat(predictions);tm,tr=b.evaluate(ema,data);vm,vr=b.evaluate(ema,val);row=dict(tune_epoch=tune,augmentation_epoch=epoch,train=tm,val=vm,lr=[q['lr'] for q in opt.param_groups],adam_steps=steps,order_sha256=s.digest(order),affine_sha256=s.digest(theta),**{n+'_sha256':hashlib.sha256(''.join(z).encode()).hexdigest() for n,z in record.items()},raw_digest=s.digest(model.state_dict()),EMA_digest=s.digest(ema.state_dict()),kd_eligible_count=count,kd_eligible_support=support.tolist(),kd_KL_T2_sum=kl,epoch_seconds=time.monotonic()-begin);history.append(row);current=(-vm['accuracy'],vm['nll'],tune)
        if current<key:
            key=current;engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,tune_epoch=tune,parent_selected_optical_epoch=86,parent_selected_head_step=4,sources=meta['sources'],config=cfg,contract=ema.contract(),metrics=dict(train=tm,val=vm)));b.save_np(a.out/'selected_train.npz',tr);b.save_np(a.out/'selected_val.npz',vr);chosen=dict(tune_epoch=tune,best_path=str(a.out/'best.pt'),train=str(a.out/'selected_train.npz'),val=str(a.out/'selected_val.npz'),metrics=dict(train=tm,val=vm))
        engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous,optimizer=opt.state_dict(),sources=meta['sources'],config=cfg,parent=p,tune_epoch=tune,new_steps=steps,student_probabilities=trace));s.save(a.out/'history.json',history);s.save(a.out/'status.json',dict(state='view4_training',tune_epoch=tune,selected_tune_epoch=chosen['tune_epoch'],train=tm,val=vm));print(dict(tune_epoch=tune,train=tm['accuracy'],val=vm['accuracy']),flush=True)
    assert steps==3940;identity();s.save(a.out/'fit_result.json',dict(passed=True,actual_optical_epochs_spent=120,additional_epochs=20,parent_selected_optical_epoch=86,inherited_head_updates=20,inherited_selected_head_step=4,selected_tune_epoch=chosen['tune_epoch'],best_path=chosen['best_path'],best_sha256=s.sha(chosen['best_path']),selected_train_path=chosen['train'],selected_val_path=chosen['val'],metrics=chosen['metrics'],new_adam_steps=3940,optimizer_restored=False,fit_seconds=time.monotonic()-started,history_sha256=s.sha(a.out/'history.json'),last_sha256=s.sha(a.out/'last.pt'),stop_reason='registered20_epoch_selfview_review'),exclusive=True)
def test(a):
    receipt(OUT,'audit_completed.json','science');assert a.out==OUT/'test' and not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'selfview_selected_digest_once_test');a.out.mkdir();p=parent();f=s.read(OUT/'fit/fit_result.json');checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);digest=s.digest(checkpoint['model']);s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),model_digest=digest,best_sha256=f['best_sha256'],selected_tune_epoch=f['selected_tune_epoch']),exclusive=True)
    if digest in p['test_registry']:
        known=p['test_registry'][digest];assert known['predictions_sha256']==s.sha(known['predictions_path']);result=dict(metrics=known['metrics'],predictions_path=known['predictions_path'],predictions_sha256=known['predictions_sha256'],reused=True,new_test_evaluations=0)
    else:
        assert a.gpu==b.GPU[2] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);model=RegularizedMoE(4,.3).cuda();model.load_state_dict(checkpoint['model'],strict=True);metrics,arrays=b.evaluate(model,b.load('test'));b.save_np(a.out/'predictions.npz',arrays);result=dict(metrics=metrics,predictions_path=str(a.out/'predictions.npz'),predictions_sha256=s.sha(a.out/'predictions.npz'),reused=False,new_test_evaluations=1)
    s.save(a.out/'result.json',dict(result,model_digest=digest,best_sha256=f['best_sha256'],selected_tune_epoch=f['selected_tune_epoch'],test_used_for_epoch_selection=False,test_informed_development=True),exclusive=True)
def child(a,phase):
    with (OUT/(phase+'.log')).open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--out',str(OUT/phase),'--gpu',a.gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu),stdout=log,stderr=subprocess.STDOUT);rc=proc.wait()
    s.save(OUT/(phase+'_exit.json'),dict(pid=proc.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Keep failure; no retry'
def audit_child(phase):
    claim(OUT/(phase+'_audit_claim.json'),'own_selfview_'+phase+'_saved_math');folder='science' if phase=='fit' else 'saved_test_audit'
    with (OUT/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_richsplit_view4.py'),'--base',str(OUT),'--out',str(OUT/folder),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(OUT/folder/'audit.json')['passed'];s.save(OUT/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(OUT/folder/'audit.json')),exclusive=True)
def campaign(a):
    cpu_ready();assert a.out==OUT and not OUT.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.gpu==b.GPU[2];b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.preflight(a.gpu,shared=True,memory_gib=2);claim(OUT.with_name(OUT.name+'.campaign_claim.json'),'single_view4_candidate');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),parent=parent(),config=config(),gpu_uuid=a.gpu),exclusive=True);s.save(OUT/'status.json',dict(state='view4_running'));child(a,'fit');audit_child('fit');child(a,'test');audit_child('test');f=s.read(OUT/'fit/fit_result.json');t=s.read(OUT/'test/result.json');control=s.read(k.BASE/'once_tests/L4_rho0/result.json')['metrics'];s.save(OUT/'comparison.json',dict(train=f['metrics']['train'],val=f['metrics']['val'],test=t['metrics'],control100=control,test_gap_pp=100*(t['metrics']['accuracy']-control['accuracy']),retain_parent_if_test_lower=t['metrics']['accuracy']<parent()['test_registry'][s.digest(torch.load(parent()['best'],map_location='cpu',weights_only=False)['model'])]['metrics']['accuracy'],test_informed_development=True),exclusive=True);s.save(OUT/'status.json',dict(state='view4_science_test_complete',comparison_sha256=s.sha(OUT/'comparison.json')))
def supervise(a):
    cpu_ready();assert a.out==OUT and not OUT.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(OUT.with_name(OUT.name+'.launch_claim.json'),'finite_training_no_scheduler')
    with OUT.with_name(OUT.name+'.campaign.log').open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(OUT),'--gpu',a.gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);s.save(OUT.with_name(OUT.name+'.started.json'),dict(identity(),supervisor_pid=os.getpid(),campaign_pid=proc.pid),exclusive=True);rc=proc.wait()
    path=OUT/'campaign_exit.json' if OUT.exists() else OUT.with_name(OUT.name+'.campaign_exit.json');s.save(path,dict(pid=proc.pid,supervisor_pid=os.getpid(),returncode=rc,actual_exit=True),exclusive=True);assert rc==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
