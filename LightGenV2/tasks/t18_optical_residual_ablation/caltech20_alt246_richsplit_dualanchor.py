"""Preserve clean supervision while learning a lightly weighted mild TRAIN view."""
import argparse,copy,hashlib,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_view4 as prior
import caltech20_alt246_richsplit_selfview2 as recent2
fresh=prior.fresh
b,k,v,g,s,engine,HERE,ROOT=fresh.b,fresh.k,fresh.v,fresh.g,fresh.s,fresh.engine,fresh.HERE,fresh.ROOT
RegularizedMoE=fresh.RegularizedMoE
PROFILE=HERE/'caltech20_alt246_richsplit_dualanchor_profile.json'
FILES=('caltech20_alt246_richsplit_dualanchor.py','caltech20_alt246_richsplit_dualanchor_profile.json','check_caltech20_alt246_richsplit_dualanchor.py','audit_caltech20_alt246_richsplit_dualanchor.py','CALTECH20_ALT246_RICHSPLIT_DUALANCHOR_PROTOCOL_20261017.md')
DEPTH=None
CPU=HERE/'runs/smoke/caltech20_richsplit_dualanchor_cpu_20261017'
def configure(depth):
    global DEPTH,RUN,OUT
    assert depth in (2,4);DEPTH=depth;RUN='caltech20_rich20_L'+str(depth)+'_dual_clean_weakview_s17_tune10_20261017';OUT=HERE/'runs/simulation'/RUN
def card():return 5 if DEPTH==2 else 2
receipt=k.receipt
def sources():
    q=prior.sources()
    for name in FILES:q[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return q
def identity():
    q=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(q,commit);return dict(sources=q,git_commit=commit)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def config():
    p=s.read(PROFILE);assert DEPTH in (2,4) and p['additional_epochs']==10 and p['maximum_seconds']==3600 and p['strong_branch_weight']==.25 and p['automatic_retry'] is False;q=b.config();q.update(p['training']);q.update(additional_epochs=10,maximum_seconds=3600,selection=p['selection'],class_weight_exponent=0. if DEPTH==2 else .5,strong_branch_weight=.25);return q

def parent():
    assert DEPTH in (2,4);p=prior.prior.parent() if DEPTH==2 else prior.parent();profile=s.read(PROFILE)['parents'][str(DEPTH)];assert p['best_sha256']==profile['checkpoint_sha256'] and s.sha(p['train'])==profile['TRAIN_prediction_sha256'];registry=dict(p['test_registry'])
    if DEPTH==2:
        root=recent2.OUT;f,t,meta=prior.completed(root,recent2);registry[t['model_digest']]=dict(result_path=str(root/'test/result.json'),predictions_path=t['predictions_path'],predictions_sha256=t['predictions_sha256'],metrics=t['metrics'])
    p=dict(p);p.update(test_registry=registry,teacher=dict(checkpoint=p['best'],checkpoint_sha256=p['best_sha256'],train=p['train'],TRAIN_prediction_sha256=s.sha(p['train'])));return p

def teacher_data():
    p=parent();manifest=b.dataset.manifest()
    with np.load(p['train'],allow_pickle=False) as z:probs=z['probabilities'].copy();labels=z['labels'].copy();ids=z['ids'].copy()
    assert ids.tolist()==manifest['split_ids']['train'] and np.bincount(labels,minlength=20).tolist()==manifest['supports']['train'];return probs,labels

def class_weights(data,device):return torch.ones(20,device=device) if DEPTH==2 else (len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).sqrt().to(device)

def clean_batch(data,idx,cfg,device):
    return b.encode(data[0][idx].to(device),None,100),engine.F_onehot(data[1][idx].to(device),cfg['label_smoothing'])

def dual_gradients(model,x0,y0,x,y,weights,cfg,target,eligible,micro,collect_gradients=True):
    model.train();model.zero_grad(set_to_none=True);ps=[];clean=[];cs=[];model.linear.mask=None;beta=cfg['strong_branch_weight']
    for off in range(0,len(x),micro):
        z,c0=model(x0[off:off+micro]);((1-beta)*k.loss_core(model,z,c0,y0[off:off+micro],weights,cfg)*len(z)/len(x)).backward();clean.append(z.detach())
        p,c=model(x[off:off+micro]);(beta*k.loss_core(model,p,c,y[off:off+micro],weights,cfg,target[off:off+micro],eligible[off:off+micro])*len(p)/len(x)).backward();ps.append(p.detach());cs.append(c.detach())
    return torch.cat(ps),torch.cat(cs),torch.cat(clean),({n:p.grad.detach().cpu().numpy().astype(np.float64) for n,p in model.named_parameters()} if collect_gradients else None)

def cpu_ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['parents'][str(DEPTH)]==parent();engine.verify_sources(q['sources'],q['git_commit'])
def gate(state,data,cfg,device):
    model=RegularizedMoE(DEPTH,.3).to(device);model.load_state_dict(state,strict=True);initial=s.digest(model.state_dict());weights=class_weights(data,device);teacher=teacher_data();order,theta=engine.epoch_plan(201,len(data[1]),cfg);checks=[]
    for name,idx,batch in [('all20',np.array([int((data[1]==i).nonzero()[0]) for i in range(20)]),0),('macro16',order[:16],0),('tail5',order[-5:],196)]:
        x,y,mask,_=v.mixed(data,idx,theta,201,batch,cfg,device);x0,y0=clean_batch(data,idx,cfg,device);q,gate=k.kd_targets(teacher,idx,201,batch,cfg);qt=torch.as_tensor(q,device=device);gt=torch.as_tensor(gate,device=device);assert torch.all(mask==1)
        p,c,p0,ga=dual_gradients(model,x0,y0,x,y,weights,cfg,qt,gt,1);z,d,z0,gb=dual_gradients(model,x0,y0,x,y,weights,cfg,qt,gt,4);norms={n:float(np.linalg.norm(a.ravel())) for n,a in gb.items()};assert len(norms)==(12 if DEPTH==2 else 21) and all(np.isfinite(a).all() and norms[n]>0 for n,a in gb.items());rel=max(float(np.linalg.norm((ga[n]-a).ravel())/norms[n]) for n,a in gb.items());cos=min(float(ga[n].ravel()@a.ravel()/(np.linalg.norm(ga[n])*norms[n])) for n,a in gb.items());err=max(float((p-z).abs().max()),float((c-d).abs().max()),float((p0-z0).abs().max()));assert err<=2e-6 and rel<=.02 and cos>=.999,(name,err,rel,cos);checks.append(dict(samples=name,method='dual_clean075_weak025',n=len(idx),maxabs=err,gradient_relative_l2=rel,gradient_cosine=cos,gradient_norms=norms,teacher_eligible=int(gate.sum())))
    assert s.digest(model.state_dict())==initial;del model;torch.cuda.empty_cache();return dict(passed=True,initial_digest=initial,checks=checks,optimizer_steps=0,TEST_images_read=0)

def fit(a):
    cpu_ready();p=parent();cfg=config();assert a.out==OUT/'fit' and not a.out.exists() and a.gpu==b.GPU[card()] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'selected_parent_new_optimizer_strong_view_selftargets');a.out.mkdir();s.gpu_budget.ALLOWED.update(b.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);checkpoint=torch.load(p['best'],map_location='cpu',weights_only=False);data,val=b.load('train'),b.load('val');teacher=teacher_data();proof=gate(checkpoint['model'],data,cfg,'cuda');s.save(a.out/'numeric_gate.json',proof,exclusive=True);model=RegularizedMoE(DEPTH,.3).cuda();model.load_state_dict(checkpoint['model'],strict=True);ema=copy.deepcopy(model).eval();phase=[];electronic=[]
    for n,w in model.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(w)
    opt=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr']);assert len(opt.state)==0;meta=dict(identity(),pid=os.getpid(),depth=DEPTH,parent=p,config=cfg,gpu_uuid=a.gpu,policy=policy,initial_digest=s.digest(model.state_dict()),optimizer_restored=False,initial_optimizer_states=0,teacher_forward_count=0);s.save(a.out/'metadata.json',meta,exclusive=True);history=[dict(tune_epoch=0,augmentation_epoch=100,train=p['metrics']['train'],val=p['metrics']['val'],EMA_digest=s.digest(checkpoint['model']))];key=(-p['metrics']['val']['accuracy'],p['metrics']['val']['nll'],0);chosen=dict(tune_epoch=0,best_path=p['best'],train=p['train'],val=p['val'],metrics=p['metrics']);trace={};steps=0;started=time.monotonic();weights=class_weights(data,'cuda')
    for tune in range(1,11):
        assert time.monotonic()-started<3600,'Incomplete20: retain/no retry';epoch=200+tune;begin=time.monotonic();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);record={n:[] for n in ('mix','flip','mask','view','pixels','teacher_target','teacher_gate','clean_pixels')};count=0;support=np.zeros(20,np.int64);kl=0.;predictions=[];clean_predictions=[];clean_ce=0.;strong_ce=0.;model.train()
        for batch,off in enumerate(range(0,len(order),16)):
            idx=order[off:off+16];x,y,mask,hashes=v.mixed(data,idx,theta,epoch,batch,cfg,'cuda');target,eligible=k.kd_targets(teacher,idx,epoch,batch,cfg);hashes.update(teacher_target=s.digest(target),teacher_gate=s.digest(eligible),clean_pixels=s.digest(data[0][idx].numpy()));count+=int(eligible.sum());support+=np.bincount(data[1][idx].numpy(),weights=eligible,minlength=20).astype(np.int64)
            for name,value in hashes.items():record[name].append(value)
            x0,y0=clean_batch(data,idx,cfg,'cuda');prob,_,p0,_=dual_gradients(model,x0,y0,x,y,weights,cfg,torch.as_tensor(target,device='cuda'),torch.as_tensor(eligible,device='cuda'),4,False);raw=prob.cpu().numpy().astype(np.float64);raw0=p0.cpu().numpy().astype(np.float64);predictions.append(torch.from_numpy(raw.astype(np.float32)));clean_predictions.append(torch.from_numpy(raw0.astype(np.float32)));clean_ce+=float(-(y0.cpu().numpy().astype(np.float64)*weights.cpu().numpy()[None]*np.log(np.maximum(raw0,1e-12))).sum());strong_ce+=float(-(y.cpu().numpy().astype(np.float64)*weights.cpu().numpy()[None]*np.log(np.maximum(raw,1e-12))).sum());kl+=k.kl_sum(raw,target,eligible,cfg);torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);previous={n:w.detach().cpu().clone() for n,w in ema.named_parameters()} if tune==10 and batch==196 else None;opt.step();steps+=1
            with torch.no_grad():
                for e,w in zip(ema.parameters(),model.parameters()):e.mul_(cfg['ema_decay']).add_(w,alpha=1-cfg['ema_decay'])
        trace[tune]=dict(strong=torch.cat(predictions),clean=torch.cat(clean_predictions));tm,tr=b.evaluate(ema,data);vm,vr=b.evaluate(ema,val);row=dict(tune_epoch=tune,augmentation_epoch=epoch,train=tm,val=vm,lr=[q['lr'] for q in opt.param_groups],adam_steps=steps,order_sha256=s.digest(order),affine_sha256=s.digest(theta),**{n+'_sha256':hashlib.sha256(''.join(z).encode()).hexdigest() for n,z in record.items()},raw_digest=s.digest(model.state_dict()),EMA_digest=s.digest(ema.state_dict()),kd_eligible_count=count,kd_eligible_support=support.tolist(),kd_KL_T2_sum=kl,clean_CE_mean=clean_ce/len(data[1]),strong_CE_mean=strong_ce/len(data[1]),epoch_seconds=time.monotonic()-begin);history.append(row);current=(-vm['accuracy'],vm['nll'],tune)
        if current<key:
            key=current;engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,tune_epoch=tune,parent_selected_optical_epoch=p['selected_optical_epoch'],parent_selected_head_step=p['selected_head_step'],sources=meta['sources'],config=cfg,contract=ema.contract(),metrics=dict(train=tm,val=vm)));b.save_np(a.out/'selected_train.npz',tr);b.save_np(a.out/'selected_val.npz',vr);chosen=dict(tune_epoch=tune,best_path=str(a.out/'best.pt'),train=str(a.out/'selected_train.npz'),val=str(a.out/'selected_val.npz'),metrics=dict(train=tm,val=vm))
        engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous,optimizer=opt.state_dict(),sources=meta['sources'],config=cfg,parent=p,tune_epoch=tune,new_steps=steps,student_probabilities=trace));s.save(a.out/'history.json',history);s.save(a.out/'status.json',dict(state='dualanchor_training',tune_epoch=tune,selected_tune_epoch=chosen['tune_epoch'],train=tm,val=vm));print(dict(tune_epoch=tune,train=tm['accuracy'],val=vm['accuracy']),flush=True)
    assert steps==1970;identity();s.save(a.out/'fit_result.json',dict(passed=True,actual_optical_epochs_spent=110,additional_epochs=10,parent_selected_optical_epoch=p['selected_optical_epoch'],inherited_head_updates=p['inherited_head_updates'],inherited_selected_head_step=p['selected_head_step'],selected_tune_epoch=chosen['tune_epoch'],best_path=chosen['best_path'],best_sha256=s.sha(chosen['best_path']),selected_train_path=chosen['train'],selected_val_path=chosen['val'],metrics=chosen['metrics'],new_adam_steps=1970,optimizer_restored=False,fit_seconds=time.monotonic()-started,history_sha256=s.sha(a.out/'history.json'),last_sha256=s.sha(a.out/'last.pt'),stop_reason='registered10_epoch_dual_clean_anchor_review'),exclusive=True)
def test(a):
    receipt(OUT,'audit_completed.json','science');assert a.out==OUT/'test' and not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'selfview_selected_digest_once_test');a.out.mkdir();p=parent();f=s.read(OUT/'fit/fit_result.json');checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);digest=s.digest(checkpoint['model']);s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),model_digest=digest,best_sha256=f['best_sha256'],selected_tune_epoch=f['selected_tune_epoch']),exclusive=True)
    if digest in p['test_registry']:
        known=p['test_registry'][digest];assert known['predictions_sha256']==s.sha(known['predictions_path']);result=dict(metrics=known['metrics'],predictions_path=known['predictions_path'],predictions_sha256=known['predictions_sha256'],reused=True,new_test_evaluations=0)
    else:
        assert a.gpu==b.GPU[card()] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);model=RegularizedMoE(DEPTH,.3).cuda();model.load_state_dict(checkpoint['model'],strict=True);metrics,arrays=b.evaluate(model,b.load('test'));b.save_np(a.out/'predictions.npz',arrays);result=dict(metrics=metrics,predictions_path=str(a.out/'predictions.npz'),predictions_sha256=s.sha(a.out/'predictions.npz'),reused=False,new_test_evaluations=1)
    s.save(a.out/'result.json',dict(result,model_digest=digest,best_sha256=f['best_sha256'],selected_tune_epoch=f['selected_tune_epoch'],test_used_for_epoch_selection=False,test_informed_development=True),exclusive=True)
def child(a,phase):
    with (OUT/(phase+'.log')).open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--out',str(OUT/phase),'--gpu',a.gpu,'--depth',str(DEPTH)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu),stdout=log,stderr=subprocess.STDOUT);rc=proc.wait()
    s.save(OUT/(phase+'_exit.json'),dict(pid=proc.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Keep failure; no retry'
def audit_child(phase):
    claim(OUT/(phase+'_audit_claim.json'),'own_selfview_'+phase+'_saved_math');folder='science' if phase=='fit' else 'saved_test_audit'
    with (OUT/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_richsplit_dualanchor.py'),'--base',str(OUT),'--out',str(OUT/folder),'--phase',phase,'--depth',str(DEPTH)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(OUT/folder/'audit.json')['passed'];s.save(OUT/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(OUT/folder/'audit.json')),exclusive=True)
def campaign(a):
    cpu_ready();assert a.out==OUT and not OUT.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.gpu==b.GPU[card()];b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.preflight(a.gpu,shared=True,memory_gib=2);claim(OUT.with_name(OUT.name+'.campaign_claim.json'),'single_dualanchor_candidate');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),parent=parent(),config=config(),gpu_uuid=a.gpu),exclusive=True);s.save(OUT/'status.json',dict(state='dualanchor_running'));child(a,'fit');audit_child('fit');child(a,'test');audit_child('test');f=s.read(OUT/'fit/fit_result.json');t=s.read(OUT/'test/result.json');control=s.read(k.BASE/('once_tests/L'+str(DEPTH)+'_rho0/result.json'))['metrics'];s.save(OUT/'comparison.json',dict(train=f['metrics']['train'],val=f['metrics']['val'],test=t['metrics'],control100=control,test_gap_pp=100*(t['metrics']['accuracy']-control['accuracy']),retain_parent_if_test_lower=t['metrics']['accuracy']<parent()['test_registry'][s.digest(torch.load(parent()['best'],map_location='cpu',weights_only=False)['model'])]['metrics']['accuracy'],test_informed_development=True),exclusive=True);s.save(OUT/'status.json',dict(state='dualanchor_science_test_complete',comparison_sha256=s.sha(OUT/'comparison.json')))
def supervise(a):
    cpu_ready();assert a.out==OUT and not OUT.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(OUT.with_name(OUT.name+'.launch_claim.json'),'finite_training_no_scheduler')
    with OUT.with_name(OUT.name+'.campaign.log').open('x') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(OUT),'--gpu',a.gpu,'--depth',str(DEPTH)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);s.save(OUT.with_name(OUT.name+'.started.json'),dict(identity(),supervisor_pid=os.getpid(),campaign_pid=proc.pid),exclusive=True);rc=proc.wait()
    path=OUT/'campaign_exit.json' if OUT.exists() else OUT.with_name(OUT.name+'.campaign_exit.json');s.save(path,dict(pid=proc.pid,supervisor_pid=os.getpid(),returncode=rc,actual_exit=True),exclusive=True);assert rc==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--depth',type=int,choices=[2,4],required=True);a=p.parse_args();configure(a.depth)
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
