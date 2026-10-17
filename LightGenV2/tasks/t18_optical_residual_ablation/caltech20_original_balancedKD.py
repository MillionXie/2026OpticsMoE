"""Residual-only teacher gradient budgets; identical deployed electronic graph."""
import argparse,copy,hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_original_equal60 as base
import caltech20_original_resnet_teacher as teacher
from functools import lru_cache

HERE,ROOT,s,engine=base.HERE,base.ROOT,base.s,base.engine
FILES=('caltech20_original_balancedKD.py', 'caltech20_original_balancedKD_profile.json', 'check_caltech20_original_balancedKD.py', 'audit_caltech20_original_balancedKD.py', 'CALTECH20_ORIGINAL_BALANCEDKD_PROTOCOL_20261017.md')
RUN='caltech20_original_equal60_residual_balancedKD20_'+base.GROUP+'_20261017'
OUT=HERE/'runs/simulation'/RUN;CPU=HERE/'runs/smoke'/('caltech20_original_balancedKD20_'+base.GROUP+'_cpu_20261017');BASE=base.OUT
CASES={f'L{d}_rho03':(d,.3) for d in base.DEPTHS}
GPU=base.GPU

def config():
    p=dict(base.config());p.update(s.read(HERE/FILES[1]));assert p['tune_epochs']==20 and p['class_weight_exponent']==1. and p['temperature']==4. and p['KD_gradient_fraction']==.25 and p['KD_scale_cap']==.1 and p['new_inference_parameters']==0;return p
def sources():
    src=base.sources()
    for name in FILES:src[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return src
def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def moments(state):return s.digest({str(i)+':'+k:v for i,values in state['state'].items() for k,v in values.items() if isinstance(v,torch.Tensor)})
def admit():
    cfg=config();assert cfg['class_count']==20 and cfg['maximum_epochs']==60
    state=s.read(BASE/'status.json');assert state['state']=='equal60_science_and_reviewed_tests_complete' and not state['active'] and not state['pending'] and not state['failures']
    for receipt,folder in [('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')]:
        q=s.read(BASE/receipt);assert q['actual_audit_returncode']==0 and q['audit_sha256']==s.sha(BASE/folder/'audit.json') and s.read(BASE/folder/'audit.json')['passed']
    meta=s.read(BASE/'metadata.json');assert meta['sources']==base.sources();engine.verify_sources(meta['sources'],meta['git_commit'])
    assert s.read(BASE/'campaign_exit.json')['returncode']==0
    controls={}
    for key in base.CASES:
        f=s.read(BASE/key/'fit_result.json');assert f['epochs_completed']==60 and f['steps']==5160 and f['last_sha256']==s.sha(BASE/key/'last.pt') and f['best_sha256']==s.sha(BASE/key/'best.pt') and f['history_sha256']==s.sha(BASE/key/'history.json')
        if key.endswith('rho0'):controls[key]=dict(best_sha256=f['best_sha256'],last_sha256=f['last_sha256'],test_result_sha256=s.sha(BASE/'once_tests'/key/'result.json'))
    teacher_ready();return controls
def cpu_ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();engine.verify_sources(p['sources'],p['git_commit'])
def class_weights(cfg,device):
    counts=np.asarray(base.dataset.manifest()['supports']['train'],np.float64);weights=np.power(counts.sum()/(20*counts),cfg['class_weight_exponent']);weights*=counts.sum()/np.dot(counts,weights)
    return torch.tensor(weights,dtype=torch.float32,device=device)
def teacher_ready():
    cfg=config();rec=s.read(teacher.OUT/'audit_completed.json');f=s.read(teacher.OUT/'fit_result.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(teacher.OUT/'science/audit.json')==cfg['teacher_science_sha256'] and s.read(teacher.OUT/'science/audit.json')['passed']
    assert s.sha(teacher.OUT/'selected_head.pt')==cfg['teacher_head_sha256'] and f['metrics']['val']['accuracy']>=.90 and f['backbone_optimizer_steps']==f['TEST_images_read']==0 and f['backbone_before']==f['backbone_after'];return f
@lru_cache(None)
def teacher_targets():
    teacher_ready();state=torch.load(teacher.OUT/'selected_head.pt',map_location='cpu',weights_only=False)
    with np.load(teacher.OUT/'train_features.npz',allow_pickle=False) as z:features,labels,ids=[z[k] for k in ('features','labels','ids')]
    assert ids.tolist()==base.manifest()['split_ids']['train'] and np.bincount(labels,minlength=20).tolist()==base.manifest()['supports']['train']
    x=((features-state['mean'])/state['std']).astype(np.float32).astype(np.float64);logits=x@state['model']['weight'].numpy().astype(np.float64).T+state['model']['bias'].numpy().astype(np.float64);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True);valid=((logits.argmax(1)==labels)&(p.max(1)>=config()['teacher_correct_confidence_gate'])).astype(np.float32)
    z=logits/config()['temperature'];q=np.exp(z-z.max(1,keepdims=True));q/=q.sum(1,keepdims=True);return q.astype(np.float32),valid
def targets(idx,epoch,batch,device):
    cfg=config();rg=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261013,4]));lam=float(rg.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rg.permutation(len(idx));qq,valid=teacher_targets();q=qq[idx];mass=valid[idx];weight=lam*mass+(1-lam)*mass[perm];target=(lam*mass[:,None]*q+(1-lam)*mass[perm,None]*q[perm])/weight[:,None].clip(1e-12);return torch.as_tensor(target,device=device),torch.as_tensor(weight,device=device),s.digest(np.r_[target.ravel(),weight].astype('<f4'))
def KD_loss(p,target,mass,cfg):
    T=cfg['temperature'];logs=torch.log_softmax(p.clamp_min(1e-12).log()/T,1);return T*T*(mass*(target*(target.clamp_min(1e-12).log()-logs)).sum(1)).mean()
def balanced_gradients(model,x,y,target,mass,mask,weights,cfg,micro,distill=True,assign=False,capture_boundary=False):
    model.train();model.zero_grad(set_to_none=True);named=list(model.named_parameters());params=[p for _,p in named];ceg=[torch.zeros_like(p) for p in params];kdg=[torch.zeros_like(p) for p in params];ps=[];cs=[];ce_value=0.;kd_value=0.
    for off in range(0,len(x),micro):
        model.linear.mask=None if mask is None else mask[off:off+micro];p,c=model(x[off:off+micro]);factor=len(p)/len(x);ce=base.loss(model,p,c,y[off:off+micro],weights,cfg)*factor;kd=KD_loss(p,target[off:off+micro],mass[off:off+micro],cfg)*factor
        gc=torch.autograd.grad(ce,params,retain_graph=True);gk=torch.autograd.grad(kd,params)
        for a,b,u,v in zip(ceg,kdg,gc,gk):a.add_(u);b.add_(v)
        ps.append(p.detach());cs.append(c.detach());ce_value+=float(ce.detach());kd_value+=float(kd.detach())
    model.linear.mask=None;groups=[[i for i,(n,_) in enumerate(named) if n=='raw_router' or n.startswith('phase_sets.')],[i for i,(n,_) in enumerate(named) if n!='raw_router' and not n.startswith('phase_sets.')]];scales=[];norm_ce=[];norm_kd=[];cos=[]
    for ix in groups:
        cn=float(sum(ceg[i].double().square().sum() for i in ix).sqrt());kn=float(sum(kdg[i].double().square().sum() for i in ix).sqrt());dot=float(sum((ceg[i].double()*kdg[i].double()).sum() for i in ix));norm_ce.append(cn);norm_kd.append(kn);scales.append(min(cfg['KD_scale_cap'],cfg['KD_gradient_fraction']*cn/(kn+1e-12)) if distill else 0.);cos.append(dot/max(cn*kn,1e-30))
    grads={}
    for group,ix in enumerate(groups):
        for i in ix:
            g=ceg[i]+scales[group]*kdg[i]
            if not assign:grads[named[i][0]]=g.detach().cpu().numpy().astype(np.float64)
            if assign:params[i].grad=g
    info=dict(scales=scales,CE_norm=norm_ce,KD_norm=norm_kd,gradient_cosine=cos,CE_with_regularizer=ce_value,KD=kd_value,teacher_mass_mean=float(mass.mean()))
    boundary=dict(CE_gradients={n:g.detach().cpu().clone() for (n,_),g in zip(named,ceg)},KD_gradients={n:g.detach().cpu().clone() for (n,_),g in zip(named,kdg)}) if capture_boundary else None
    return torch.cat(ps),torch.cat(cs),grads,info,boundary

def gate(depth,state,data,cfg):
    engine.seed(17);model=base.RegularizedMoE(depth,.3).cuda();model.load_state_dict(state,strict=True);initial=s.digest(model.state_dict());weights=class_weights(cfg,'cuda');order,theta=engine.epoch_plan(61,len(data[1]),cfg);checks=[];tail=cfg['tail_samples']
    for name,idx,batch in [('all20',np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]),0),('macro16',order[:16],0),('tail12',order[-tail:],85)]:
        x,y,mask,_=base.mixed(data,idx,theta,61,batch,cfg,'cuda');target,mass,target_sha=targets(idx,61,batch,'cuda')
        for distill in (False,True):
            for method in ('ordinary','dropout'):
                used=None if method=='ordinary' else mask;p,c,g,ia,_=balanced_gradients(model,x,y,target,mass,used,weights,cfg,1,distill);q,d,h,ib,_=balanced_gradients(model,x,y,target,mass,used,weights,cfg,4,distill)
                norms={n:float(np.linalg.norm(v.ravel())) for n,v in h.items()};assert all(np.isfinite(v).all() and norms[n]>0 for n,v in h.items())
                rel=max(float(np.linalg.norm((g[n]-v).ravel())/np.linalg.norm(v.ravel())) for n,v in h.items());cos=min(float(np.dot(g[n].ravel(),v.ravel())/(np.linalg.norm(g[n].ravel())*np.linalg.norm(v.ravel()))) for n,v in h.items());pe=float((p-q).abs().max());ce=float((c-d).abs().max());assert pe<=2e-6 and ce<=2e-6 and rel<=.02 and cos>=.999
                checks.append(dict(samples=name,method=method,distillation=distill,target_sha256=target_sha,group_gradient_info=ib,probability_maxabs=pe,capture_maxabs=ce,gradient_max_relative_l2=rel,gradient_min_cosine=cos,gradient_norms=norms))
    assert s.digest(model.state_dict())==initial;del model;torch.cuda.empty_cache();return dict(passed=True,parent_raw_digest=initial,class_weights=weights.cpu().tolist(),checks=checks,optimizer_steps=0,TEST_images_read=0)

def fit(a):
    controls=admit();cpu_ready();cfg=config();assert a.key in CASES and a.gpu in base.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and not a.out.exists();base.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'own_residual_group_balanced_teacherKD20');a.out.mkdir()
    s.gpu_budget.ALLOWED.update(base.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);depth,_=CASES[a.key];parent=BASE/a.key;last=torch.load(parent/'last.pt',map_location='cpu',weights_only=False);oldbest=torch.load(parent/'best.pt',map_location='cpu',weights_only=False);data,val=base.load('train'),base.load('val')
    proof=gate(depth,last['model'],data,cfg);s.save(a.out/'numeric_gate.json',proof,exclusive=True);model=base.RegularizedMoE(depth,.3).cuda();model.load_state_dict(last['model'],strict=True);ema=copy.deepcopy(model).eval();ema.load_state_dict(last['ema'],strict=True)
    phase=[];electronic=[]
    for name,p in model.named_parameters():(phase if name=='raw_router' or name.startswith('phase_sets.') else electronic).append(p)
    opt=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['phase_lr']);opt.load_state_dict(last['optimizer']);restored=moments(opt.state_dict());assert restored==moments(last['optimizer'])
    for group,lr,decay in zip(opt.param_groups,(cfg['phase_lr'],cfg['electronic_lr']),(0,cfg['electronic_weight_decay'])):group.update(lr=lr,initial_lr=lr,weight_decay=decay)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['tune_epochs'],eta_min=cfg['minimum_tune_lr']);weights=class_weights(cfg,'cuda');started=time.monotonic();history=[];trace_probs=[];trace_caps=[];trace_y=[];trace_q=[];trace_offsets=[0];trace_tv=[];trace_losses=[];trace_kd_losses=[];trace_mass=[];trace_info=[];steps=last['steps'];best=float(oldbest['metrics']['val']['balanced_nll']);selected_epoch=oldbest['epoch'];selected_tune=0
    meta=dict(identity(),pid=os.getpid(),case=a.key,depth=depth,rho=.3,gpu_uuid=a.gpu,policy=policy,config=cfg,parent_last_sha256=s.sha(parent/'last.pt'),parent_best_sha256=s.sha(parent/'best.pt'),parent_history_sha256=s.sha(parent/'history.json'),parent_raw_digest=s.digest(last['model']),parent_ema_digest=s.digest(last['ema']),restored_optimizer_moment_digest=restored,initial_digest=last['initial_digest'],frozen_controls=controls,class_weights=weights.cpu().tolist(),TEST_images_read=0)
    s.save(a.out/'metadata.json',meta,exclusive=True);incumbent=dict(model=oldbest['model'],epoch=selected_epoch,tune_epoch=0,config=cfg,sources=meta['sources'],metrics=oldbest['metrics'],contract=oldbest['contract']);engine.save_pt(a.out/'best.pt',incumbent)
    for role in ('train','val'):shutil.copyfile(parent/('selected_'+role+'.npz'),a.out/('selected_'+role+'.npz'))
    for tune in range(1,cfg['tune_epochs']+1):
        assert time.monotonic()-started<cfg['maximum_seconds'];epoch=60+tune;begin=time.monotonic();model.train();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);hashes={k:[] for k in ('mix','flip','mask','teacher')};losses=[];lr=[g['lr'] for g in opt.param_groups]
        for batch,start in enumerate(range(0,len(order),16)):
            idx=order[start:start+16];x,y,mask,h=base.mixed(data,idx,theta,epoch,batch,cfg,'cuda')
            for kind,value in h.items():hashes[kind].append(value)
            target,mass,thash=targets(idx,epoch,batch,'cuda');hashes['teacher'].append(thash);batch_probs=[];batch_caps=[];tv=float(model.phase_smoothness().detach())
            final=tune==cfg['tune_epochs'] and batch==cfg['steps_per_epoch']-1
            probabilities,captures,_,info,boundary=balanced_gradients(model,x,y,target,mass,mask,weights,cfg,4,True,True,final)
            trace_probs.append(probabilities.cpu().numpy());trace_caps.append(captures.cpu().numpy());trace_y.append(y.cpu().numpy());trace_q.append(target.cpu().numpy());trace_mass.append(mass.cpu().numpy());trace_offsets.append(trace_offsets[-1]+len(idx));trace_tv.append(tv);trace_losses.append(info['CE_with_regularizer']);trace_kd_losses.append(info['KD']);trace_info.append(info)
            final=tune==cfg['tune_epochs'] and batch==cfg['steps_per_epoch']-1
            if final:boundary.update(raw_before={n:p.detach().cpu().clone() for n,p in model.named_parameters()},optimizer_before=copy.deepcopy(opt.state_dict()),group_info=info)
            clip_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));previous={n:p.detach().cpu().clone() for n,p in ema.named_parameters()} if final else None
            opt.step();steps+=1
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(cfg['ema_decay']).add_(mp,alpha=1-cfg['ema_decay'])
            if final:boundary['clip_norm']=clip_norm;engine.save_pt(a.out/'final_optimizer_boundary.pt',boundary)
            value_sum=info['CE_with_regularizer']+float(np.mean(info['scales']))*info['KD']
            losses.append(value_sum)
        tm,tr=base.evaluate(ema,data);vm,vr=base.evaluate(ema,val);entry=dict(epoch=epoch,tune_epoch=tune,train=tm,val=vm,lr=lr,loss=float(np.mean(losses)),order_sha256=s.digest(order),affine_sha256=s.digest(theta),adam_steps=steps,raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin,seconds=time.monotonic()-started)
        entry.update({kind+'_sha256':hashlib.sha256(''.join(values).encode()).hexdigest() for kind,values in hashes.items()});history.append(entry)
        if vm['balanced_nll']<best:
            best=vm['balanced_nll'];selected_epoch=epoch;selected_tune=tune;engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,tune_epoch=tune,config=cfg,sources=meta['sources'],metrics=dict(train=tm,val=vm),contract=ema.contract()));base.save_np(a.out/'selected_train.npz',tr);base.save_np(a.out/'selected_val.npz',vr)
        schedule.step();engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous,optimizer=opt.state_dict(),scheduler=schedule.state_dict(),epoch=epoch,tune_epoch=tune,steps=steps,config=cfg,sources=meta['sources'],initial_digest=last['initial_digest']));s.save(a.out/'history.json',history);s.save(a.out/'status.json',dict(state='training',epoch=epoch,tune_epoch=tune,train=tm,val=vm,selected_epoch=selected_epoch,selected_tune=selected_tune));print(json.dumps(dict(case=a.key,epoch=epoch,tune_epoch=tune,train=tm['accuracy'],val=vm['accuracy'],NLL=vm['nll'],balancedNLL=vm['balanced_nll'])),flush=True)
    assert steps==6880;identity();admit();base.save_np(a.out/'method_trace.npz',dict(probabilities=np.concatenate(trace_probs),capture=np.concatenate(trace_caps),labels_soft=np.concatenate(trace_y),teacher_soft=np.concatenate(trace_q),offsets=np.asarray(trace_offsets),phaseTV=np.asarray(trace_tv),loss=np.asarray(trace_losses),KD_loss=np.asarray(trace_kd_losses),teacher_mass=np.concatenate(trace_mass),group_scales=np.asarray([q['scales'] for q in trace_info]),CE_norm=np.asarray([q['CE_norm'] for q in trace_info]),KD_norm=np.asarray([q['KD_norm'] for q in trace_info]),gradient_cosine=np.asarray([q['gradient_cosine'] for q in trace_info]))); best_state=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);s.save(a.out/'fit_result.json',dict(passed=True,actual_joint_epochs=80,extra_epochs=20,selected_epoch=selected_epoch,selected_tune=selected_tune,metrics=best_state['metrics'],steps=steps,best_sha256=s.sha(a.out/'best.pt'),last_sha256=s.sha(a.out/'last.pt'),history_sha256=s.sha(a.out/'history.json'),method_trace_sha256=s.sha(a.out/'method_trace.npz'),final_optimizer_boundary_sha256=s.sha(a.out/'final_optimizer_boundary.pt'),fit_seconds=time.monotonic()-started,stop_reason='strategy_review_after20',test_evaluations=0),exclusive=True);s.save(a.out/'status.json',dict(state='objective10_complete_awaiting_own_science',epoch=80,extra_epochs=20))
def campaign(a):
    admit();cpu_ready();assert a.out==OUT and not OUT.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and base.slots()['total']==0;claim(OUT.with_name(OUT.name+'.campaign_claim.json'),'one_short_objective_refinement');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),frozen_controls=admit()),exclusive=True);active=[]
    for (key,_),gpu in zip(CASES.items(),base.GPU.values()):
        s.gpu_budget.ALLOWED.update(base.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2);folder=OUT/key;log=(OUT/(key+'.fit.log')).open('x');p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','fit','--key',key,'--gpu',gpu,'--out',str(folder)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);active.append(dict(key=key,pid=p.pid,gpu=gpu,out=str(folder),process=p,log=log))
    done=[];failures=[]
    while active:
        for job in list(active):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();rec={k:v for k,v in job.items() if k not in ('process','log')};rec.update(returncode=rc,actual_exit=True);s.save(Path(job['out']).with_name(job['key']+'.fit_exit.json'),rec,exclusive=True);(failures if rc else done).append(rec);active.remove(job)
        s.save(OUT/'status.json',dict(state='fit_failed' if failures else 'fit_running' if active else 'fit_complete',active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=[],complete=done,failures=failures))
        if active:time.sleep(2)
    assert not failures and len(done)==len(CASES),failures;claim(OUT/'fit_audit_claim.json','own_independent_objective_audit')
    with (OUT/'fit_audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_original_balancedKD.py'),'--base',str(OUT),'--out',str(OUT/'science')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(OUT/'science/audit.json')['passed'];s.save(OUT/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(OUT/'science/audit.json')),exclusive=True);s.save(OUT/'status.json',dict(state='balancedKD20_science_complete_awaiting_review_no_TEST',active=[],pending=[],complete=done,failures=[]))
def test(a):
    admit();assert a.key in CASES and a.out==OUT/'once_tests'/a.key and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in GPU.values();proof=s.read(OUT/'audit_completed.json');assert proof['actual_audit_returncode']==0 and proof['audit_sha256']==s.sha(OUT/'science/audit.json') and s.read(OUT/'science/audit.json')['passed'];base.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'own_once_selected_TEST_after_review');a.out.mkdir();torch.set_num_threads(4)
    best=torch.load(OUT/a.key/'best.pt',map_location='cpu',weights_only=False);digest=s.digest(best['model']);old=s.read(BASE/'once_tests'/a.key/'started.json');reused=digest==old['model_digest']
    if reused:
        result=s.read(BASE/'once_tests'/a.key/'result.json');metrics=result['metrics'];shutil.copyfile(BASE/'once_tests'/a.key/'predictions.npz',a.out/'predictions.npz');policy=dict(reused_no_model_forward=True)
    else:
        s.gpu_budget.ALLOWED.update(GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);depth,_=CASES[a.key];model=base.RegularizedMoE(depth,.3).cuda();model.load_state_dict(best['model'],strict=True);metrics,saved=base.evaluate(model,base.load('test'));base.save_np(a.out/'predictions.npz',saved)
    s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),selected_sha256=s.sha(OUT/a.key/'best.pt'),model_digest=digest,selected_epoch=best['epoch'],test_evaluations=0 if reused else 1,test_reused=reused,policy=policy,reused_result_sha256=s.sha(BASE/'once_tests'/a.key/'result.json') if reused else None),exclusive=True)
    s.save(a.out/'result.json',dict(passed=True,metrics=metrics,selected_epoch=best['epoch'],test_evaluations=0 if reused else 1,test_reused=reused,test_used_for_selection=False,test_informed_development=True),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','test'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise
