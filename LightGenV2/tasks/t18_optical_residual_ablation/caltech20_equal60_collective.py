"""One short residual-only collective phase optimizer review; same inference graph."""
import argparse,copy,hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_equal60_logrepair as base
import caltech20_alt246_genreg as augment
import caltech20_equal60_cutmix as prior
import caltech20_equal60_collective_phase_diagnosis as diagnosis
HERE,ROOT,s,engine=base.HERE,base.ROOT,base.s,base.engine
FILES=('caltech20_equal60_collective.py', 'caltech20_equal60_collective_profile.json', 'check_caltech20_equal60_collective.py', 'audit_caltech20_equal60_collective.py', 'CALTECH20_EQUAL60_COLLECTIVE_PROTOCOL_20261017.md')
RUN='caltech20_rich20_equal60_residual_collective10_s17_20261017'
OUT=HERE/'runs/simulation'/RUN;CPU=HERE/'runs/smoke/caltech20_equal60_collective_cpu_20261017';BASE=prior.OUT;INCUMBENT=prior.prior.OUT
CASES={f'L{d}_rho03':(d,.3) for d in (6,4,2)}
GPU=base.GPU

def config():
    p=dict(prior.config());p.update(s.read(HERE/FILES[1]));assert p['tune_epochs']==10 and p['class_weight_exponent']==.5 and p['new_inference_parameters']==0;return p
def sources():
    src=diagnosis.sources()
    for name in FILES:src[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return src
def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def moments(state):return s.digest({str(i)+':'+k:v for i,values in state['state'].items() for k,v in values.items() if isinstance(v,torch.Tensor)})
def admit():
    controls=prior.admit();cfg=config();assert cfg['parent_run']==prior.RUN and cfg['incumbent_run']==INCUMBENT.name
    state=s.read(BASE/'status.json');assert state['state']=='cutmix10_science_and_reviewed_tests_complete' and not state['active'] and not state['pending'] and not state['failures']
    for receipt,folder in [('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')]:
        q=s.read(BASE/receipt);assert q['actual_audit_returncode']==0 and q['audit_sha256']==s.sha(BASE/folder/'audit.json') and s.read(BASE/folder/'audit.json')['passed']
    meta=s.read(BASE/'metadata.json');assert meta['sources']==prior.sources();engine.verify_sources(meta['sources'],meta['git_commit']);assert s.read(BASE/'campaign_exit.json')['returncode']==0
    for key in CASES:
        f=s.read(BASE/key/'fit_result.json');ex=s.read(BASE/(key+'.fit_exit.json'));assert f['actual_joint_epochs']==80 and f['steps']==15760 and ex['returncode']==0 and ex['actual_exit']
        assert f['last_sha256']==s.sha(BASE/key/'last.pt') and f['best_sha256']==s.sha(BASE/key/'best.pt') and f['history_sha256']==s.sha(BASE/key/'history.json')
    return controls

def cpu_ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();engine.verify_sources(p['sources'],p['git_commit'])
def class_weights(cfg,device):
    counts=np.asarray(base.dataset.manifest()['supports']['train'],np.float64);weights=np.power(counts.sum()/(20*counts),cfg['class_weight_exponent']);weights*=counts.sum()/np.dot(counts,weights)
    return torch.tensor(weights,dtype=torch.float32,device=device)
def collective_gradient(model,gradients=None):
    values=[]
    for lo,hi in model.stage_slices:
        total=0.
        for i in range(lo,hi):
            raw=model.phase_sets[i];u=raw.detach().double().sigmoid();jac=2*math.pi*u*(1-u)
            g=raw.grad.detach().double() if gradients is None else torch.as_tensor(gradients['phase_sets.'+str(i)],device=raw.device,dtype=torch.float64)
            total+=float((g/jac).sum())
        values.append(total)
    return np.asarray(values,np.float64)

def collective_update(model,opt,state,gradient,rate,cfg,capture=False):
    g=np.asarray(gradient,np.float64);assert g.shape==(model.depth,) and np.isfinite(g).all();g=g*min(1.,1/(np.linalg.norm(g)+1e-12));state['t']+=1;state['m']=.9*state['m']+.1*g;state['v']=.999*state['v']+.001*g*g;t=state['t'];move=-rate*(state['m']/(1-.9**t))/(np.sqrt(state['v']/(1-.999**t))+1e-8);new=np.clip(state['cumulative']+move,-cfg['collective_limit'],cfg['collective_limit']);delta=new-state['cumulative'];state['cumulative']=new;before={};clipped=[]
    with torch.no_grad():
        for k,(lo,hi) in enumerate(model.stage_slices):
            clipped_count=0
            for i in range(lo,hi):
                raw=model.phase_sets[i];item=opt.state[raw]
                if capture:before['phase_sets.'+str(i)]=dict(raw=raw.detach().cpu().clone(),exp_avg=item['exp_avg'].detach().cpu().clone(),exp_avg_sq=item['exp_avg_sq'].detach().cpu().clone())
                u=raw.double().sigmoid();q=u+float(delta[k])/(2*math.pi);clipped_count+=int(((q<1e-6)|(q>1-1e-6)).sum());q=q.clamp(1e-6,1-1e-6);old_j=2*math.pi*u*(1-u);raw.copy_(torch.logit(q).to(raw.dtype));v=raw.double().sigmoid();ratio=(2*math.pi*v*(1-v))/old_j;item['exp_avg'].mul_(ratio.to(item['exp_avg'].dtype));item['exp_avg_sq'].mul_((ratio*ratio).to(item['exp_avg_sq'].dtype))
            clipped.append(clipped_count)
    return dict(t=t,input_gradient=np.asarray(gradient).tolist(),used_gradient=g.tolist(),m=state['m'].tolist(),v=state['v'].tolist(),delta=delta.tolist(),cumulative=new.tolist(),rate=rate,clipped_pixels=clipped),before

def cutmix_plan(n,epoch,batch,cfg,force=False):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261017,6]));apply=bool(rng.random()<cfg['cutmix_probability']);fraction=float(rng.beta(cfg['cutmix_alpha'],cfg['cutmix_alpha']));perm=rng.permutation(n);width=int(112*math.sqrt(1-fraction));cx,cy=rng.integers(0,112,size=2);apply=apply or force
    x1,x2=max(int(cx)-width//2,0),min(int(cx)+width//2,112);y1,y2=max(int(cy)-width//2,0),min(int(cy)+width//2,112)
    if not apply:x1=x2=y1=y2=0
    lam=1-(x2-x1)*(y2-y1)/(112*112)
    return dict(apply=apply,perm=perm,box=(x1,y1,x2,y2),lam=lam)

def cutmix_apply(x,y,plan):
    assert x.shape[1:]==(224,224) and len(x)==len(plan['perm']);q=torch.as_tensor(plan['perm'],device=x.device);z=x.clone();x1,y1,x2,y2=plan['box']
    for top in (0,112):
        for left in (0,112):z[:,top+y1:top+y2,left+x1:left+x2]=x[q,top+y1:top+y2,left+x1:left+x2]
    z=z*(.25/z.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6));lam=plan['lam'];target=lam*y+(1-lam)*y[q]
    return z,target

def mixed(data,idx,theta,epoch,batch,cfg,device,force=False):
    flip,mask=augment.augmentation_masks(len(idx),epoch,batch,cfg);images=data[0][idx].to(device);flag=torch.as_tensor(flip,device=device);images=torch.where(flag[:,None,None,None],images.flip(-1),images);x=base.b.encode(images,torch.as_tensor(theta[idx],device=device),100);y=engine.F_onehot(data[1][idx].to(device),cfg['label_smoothing']);plan=cutmix_plan(len(idx),epoch,batch,cfg,force);x,y=cutmix_apply(x,y,plan);packed=np.r_[float(plan['apply']),plan['lam'],plan['box'],plan['perm'].astype(float)].astype('<f8')
    return x,y,torch.as_tensor(mask,device=device),dict(mix=s.digest(packed),flip=s.digest(flip),mask=s.digest(mask))

def gate(depth,state,data,cfg):
    engine.seed(17);model=base.RegularizedMoE(depth,.3).cuda();model.load_state_dict(state,strict=True);initial=s.digest(model.state_dict());weights=class_weights(cfg,'cuda');order,theta=engine.epoch_plan(81,len(data[1]),cfg);checks=[];tail=cfg['tail_samples']
    for name,idx,batch in [('all20',np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]),0),('macro16',order[:16],0),('tail5',order[-tail:],cfg['steps_per_epoch']-1)]:
        x,y,mask,_=mixed(data,idx,theta,81,batch,cfg,'cuda',force=True)
        for method in ('ordinary','dropout'):
            used=None if method=='ordinary' else mask;p,c,g=augment.gradients(model,x,y,used,weights,cfg,1);q,d,h=augment.gradients(model,x,y,used,weights,cfg,4)
            norms={n:float(np.linalg.norm(v.ravel())) for n,v in h.items()};assert all(np.isfinite(v).all() and norms[n]>0 for n,v in h.items())
            rel=max(float(np.linalg.norm((g[n]-v).ravel())/np.linalg.norm(v.ravel())) for n,v in h.items());cos=min(float(np.dot(g[n].ravel(),v.ravel())/(np.linalg.norm(g[n].ravel())*np.linalg.norm(v.ravel()))) for n,v in h.items());pe=float((p-q).abs().max());ce=float((c-d).abs().max());assert pe<=2e-6 and ce<=2e-6 and rel<=.02 and cos>=.999
            cg=collective_gradient(model,g);ch=collective_gradient(model,h);cr=float(np.linalg.norm(cg-ch)/np.linalg.norm(ch));cc=float(cg@ch/(np.linalg.norm(cg)*np.linalg.norm(ch)));assert np.isfinite(ch).all() and np.linalg.norm(ch)>0 and cr<=.02 and cc>=.999
            checks.append(dict(samples=name,method=method,probability_maxabs=pe,capture_maxabs=ce,gradient_max_relative_l2=rel,gradient_min_cosine=cos,gradient_norms=norms,collective_gradient_relative_l2=cr,collective_gradient_cosine=cc))
    assert s.digest(model.state_dict())==initial;del model;torch.cuda.empty_cache();return dict(passed=True,parent_raw_digest=initial,class_weights=weights.cpu().tolist(),checks=checks,optimizer_steps=0,TEST_images_read=0,forced_cutmix=True)
def fit(a):
    controls=admit();cpu_ready();cfg=config();assert a.key in CASES and a.gpu in base.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and not a.out.exists();base.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'own_residual_sqrt_collective10');a.out.mkdir()
    s.gpu_budget.ALLOWED.update(base.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);depth,_=CASES[a.key];parent=BASE/a.key;last=torch.load(parent/'last.pt',map_location='cpu',weights_only=False);oldbest=torch.load(INCUMBENT/a.key/'best.pt',map_location='cpu',weights_only=False);data,val=base.load('train'),base.load('val')
    proof=gate(depth,last['model'],data,cfg);s.save(a.out/'numeric_gate.json',proof,exclusive=True);model=base.RegularizedMoE(depth,.3).cuda();model.load_state_dict(last['model'],strict=True);ema=copy.deepcopy(model).eval();ema.load_state_dict(last['ema'],strict=True)
    phase=[];electronic=[]
    for name,p in model.named_parameters():(phase if name=='raw_router' or name.startswith('phase_sets.') else electronic).append(p)
    opt=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['phase_lr']);opt.load_state_dict(last['optimizer']);restored=moments(opt.state_dict());assert restored==moments(last['optimizer'])
    for group,lr,decay in zip(opt.param_groups,(cfg['phase_lr'],cfg['electronic_lr']),(0,cfg['electronic_weight_decay'])):group.update(lr=lr,initial_lr=lr,weight_decay=decay)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['tune_epochs'],eta_min=cfg['minimum_tune_lr']);weights=class_weights(cfg,'cuda');started=time.monotonic();history=[];collective=dict(t=0,m=np.zeros(depth),v=np.zeros(depth),cumulative=np.zeros(depth));final_before={};steps=last['steps'];best=float(oldbest['metrics']['val']['nll']);selected_epoch=oldbest['epoch'];selected_tune=0
    meta=dict(identity(),pid=os.getpid(),case=a.key,depth=depth,rho=.3,gpu_uuid=a.gpu,policy=policy,config=cfg,parent_last_sha256=s.sha(parent/'last.pt'),parent_best_sha256=s.sha(parent/'best.pt'),parent_history_sha256=s.sha(parent/'history.json'),parent_raw_digest=s.digest(last['model']),parent_ema_digest=s.digest(last['ema']),restored_optimizer_moment_digest=restored,initial_digest=last['initial_digest'],incumbent_best_sha256=s.sha(INCUMBENT/a.key/'best.pt'),incumbent_root=str(INCUMBENT),frozen_controls=controls,class_weights=weights.cpu().tolist(),TEST_images_read=0)
    s.save(a.out/'metadata.json',meta,exclusive=True);incumbent=dict(model=oldbest['model'],epoch=selected_epoch,tune_epoch=0,config=cfg,sources=meta['sources'],metrics=oldbest['metrics'],contract=oldbest['contract']);engine.save_pt(a.out/'best.pt',incumbent)
    for role in ('train','val'):shutil.copyfile(INCUMBENT/a.key/('selected_'+role+'.npz'),a.out/('selected_'+role+'.npz'))
    for tune in range(1,cfg['tune_epochs']+1):
        assert time.monotonic()-started<cfg['maximum_seconds'];epoch=80+tune;begin=time.monotonic();model.train();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);hashes={k:[] for k in ('mix','flip','mask')};losses=[];coordinate_trace=[];coordinate_rate=cfg['minimum_tune_lr']+(cfg['collective_lr']-cfg['minimum_tune_lr'])*(1+math.cos(math.pi*(tune-1)/10))/2;lr=[g['lr'] for g in opt.param_groups]
        for batch,start in enumerate(range(0,len(order),16)):
            idx=order[start:start+16];x,y,mask,h=mixed(data,idx,theta,epoch,batch,cfg,'cuda')
            for kind,value in h.items():hashes[kind].append(value)
            opt.zero_grad(set_to_none=True);value_sum=0.
            for off in range(0,len(idx),4):
                model.linear.mask=mask[off:off+4];p,c=model(x[off:off+4]);loss=base.loss(model,p,c,y[off:off+4],weights,cfg)*len(p)/len(idx);loss.backward();value_sum+=float(loss.detach())
            model.linear.mask=None;torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);cg=collective_gradient(model);previous={n:p.detach().cpu().clone() for n,p in ema.named_parameters()} if tune==cfg['tune_epochs'] and batch==cfg['steps_per_epoch']-1 else None;opt.step();steps+=1;trace,before=collective_update(model,opt,collective,cg,coordinate_rate,cfg,capture=previous is not None);coordinate_trace.append(trace)
            if before:final_before=before
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(cfg['ema_decay']).add_(mp,alpha=1-cfg['ema_decay'])
            losses.append(value_sum)
        tm,tr=base.evaluate(ema,data);vm,vr=base.evaluate(ema,val);entry=dict(epoch=epoch,tune_epoch=tune,train=tm,val=vm,lr=lr,loss=float(np.mean(losses)),order_sha256=s.digest(order),affine_sha256=s.digest(theta),adam_steps=steps,raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin,seconds=time.monotonic()-started)
        entry.update(collective_trace=coordinate_trace,collective_steps=collective['t']);entry.update({kind+'_sha256':hashlib.sha256(''.join(values).encode()).hexdigest() for kind,values in hashes.items()});history.append(entry)
        if vm['nll']<best:
            best=vm['nll'];selected_epoch=epoch;selected_tune=tune;engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,tune_epoch=tune,config=cfg,sources=meta['sources'],metrics=dict(train=tm,val=vm),contract=ema.contract()));base.save_np(a.out/'selected_train.npz',tr);base.save_np(a.out/'selected_val.npz',vr)
        schedule.step();engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous,optimizer=opt.state_dict(),scheduler=schedule.state_dict(),epoch=epoch,tune_epoch=tune,steps=steps,config=cfg,sources=meta['sources'],initial_digest=last['initial_digest'],collective_optimizer=collective,before_final_collective=final_before));s.save(a.out/'history.json',history);s.save(a.out/'status.json',dict(state='training',epoch=epoch,tune_epoch=tune,train=tm,val=vm,selected_epoch=selected_epoch,selected_tune=selected_tune));print(json.dumps(dict(case=a.key,epoch=epoch,tune_epoch=tune,train=tm['accuracy'],val=vm['accuracy'],NLL=vm['nll'],balancedNLL=vm['balanced_nll'])),flush=True)
    assert steps==17730 and collective['t']==1970;identity();admit();best_state=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);s.save(a.out/'fit_result.json',dict(passed=True,actual_joint_epochs=90,extra_epochs=10,selected_epoch=selected_epoch,selected_tune=selected_tune,metrics=best_state['metrics'],steps=steps,best_sha256=s.sha(a.out/'best.pt'),last_sha256=s.sha(a.out/'last.pt'),history_sha256=s.sha(a.out/'history.json'),fit_seconds=time.monotonic()-started,stop_reason='strategy_review_after10',test_evaluations=0),exclusive=True);s.save(a.out/'status.json',dict(state='collective10_complete_awaiting_own_science',epoch=90,extra_epochs=10))
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
    assert not failures and len(done)==3,failures;claim(OUT/'fit_audit_claim.json','own_independent_objective_audit')
    with (OUT/'fit_audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_equal60_collective.py'),'--base',str(OUT),'--out',str(OUT/'science')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(OUT/'science/audit.json')['passed'];s.save(OUT/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(OUT/'science/audit.json')),exclusive=True);s.save(OUT/'status.json',dict(state='collective10_science_complete_awaiting_review_no_TEST',active=[],pending=[],complete=done,failures=[]))
def test(a):
    admit();assert a.key in CASES and a.out==OUT/'once_tests'/a.key and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in GPU.values();proof=s.read(OUT/'audit_completed.json');assert proof['actual_audit_returncode']==0 and proof['audit_sha256']==s.sha(OUT/'science/audit.json') and s.read(OUT/'science/audit.json')['passed'];base.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'own_once_selected_TEST_after_review');a.out.mkdir();torch.set_num_threads(4)
    best=torch.load(OUT/a.key/'best.pt',map_location='cpu',weights_only=False);digest=s.digest(best['model']);known=[root for root in (base.OUT,INCUMBENT,BASE) if s.read(root/'once_tests'/a.key/'started.json')['model_digest']==digest];reused=bool(known);reused_root=known[0] if reused else None
    if reused:
        result=s.read(reused_root/'once_tests'/a.key/'result.json');metrics=result['metrics'];shutil.copyfile(reused_root/'once_tests'/a.key/'predictions.npz',a.out/'predictions.npz');policy=dict(reused_no_model_forward=True)
    else:
        s.gpu_budget.ALLOWED.update(GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);depth,_=CASES[a.key];model=base.RegularizedMoE(depth,.3).cuda();model.load_state_dict(best['model'],strict=True);metrics,saved=base.evaluate(model,base.load('test'));base.save_np(a.out/'predictions.npz',saved)
    s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),selected_sha256=s.sha(OUT/a.key/'best.pt'),model_digest=digest,selected_epoch=best['epoch'],test_evaluations=0 if reused else 1,test_reused=reused,policy=policy,reused_root=str(reused_root) if reused else None,reused_result_sha256=s.sha(reused_root/'once_tests'/a.key/'result.json') if reused else None),exclusive=True)
    s.save(a.out/'result.json',dict(passed=True,metrics=metrics,selected_epoch=best['epoch'],test_evaluations=0 if reused else 1,test_reused=reused,test_used_for_selection=False,test_informed_development=True),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','test'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise
