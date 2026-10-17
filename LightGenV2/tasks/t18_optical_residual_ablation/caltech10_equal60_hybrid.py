"""Human-approved original Caltech10 paired60; early depth2 review before4/6."""
import argparse,copy,hashlib,json,math,os,subprocess,sys,time,types
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import caltech20_equal60_logrepair as frozen
from caltech10_equal60_hybrid_model import RegularizedMoE
from resisc20_input_encoder import encode
from resisc10 import metrics
HERE,ROOT,s,engine=frozen.HERE,frozen.ROOT,frozen.s,frozen.engine
FILES=('caltech10_equal60_hybrid.py', 'caltech10_equal60_hybrid_model.py', 'caltech10_equal60_hybrid_profile.json', 'check_caltech10_equal60_hybrid.py', 'audit_caltech10_equal60_hybrid.py', 'caltech10_resnet_teacher.py', 'CALTECH10_EQUAL60_HYBRID_PROTOCOL_20261017.md')
GROUP=os.environ.get('T18_TEN_GROUP','pilot');assert GROUP in ('pilot','deep')
DEPTHS=(2,) if GROUP=='pilot' else (4,6)
CASES={f'L{d}_rho{tag}':(d,rho) for d in DEPTHS for tag,rho in [('03',.3),('0',0.)]}
GPU=dict(frozen.GPU)
RUN='caltech10_original_fixed_current246_s17_equal60_'+GROUP+'_20261017'
OUT=HERE/'runs/simulation'/RUN
CPU=HERE/'runs/smoke/caltech10_current246_equal60_cpu_20261017'
PROFILE=HERE/'caltech10_equal60_hybrid_profile.json'
@lru_cache(None)
def manifest():
    p=s.read(PROFILE);assert s.sha(p['subset_manifest'])==p['subset_manifest_sha256'] and s.sha(Path(p['data']).parent/'data_manifest.json')==p['parent_manifest_sha256']
    m=s.read(p['subset_manifest']);assert m['classes']==p['classes'] and m['original_indices']==p['original_indices'] and m['cache_sha256']==p['cache_sha256']
    assert [sum(m['supports'][role]) for role in ('train','val','test')]==[761,157,1833]
    ids=[set(m['split_ids'][role]) for role in ('train','val','test')];assert all(not ids[i]&ids[j] for i in range(3) for j in range(i));return m
def config():
    cfg=s.read(PROFILE);m=manifest();cfg.update(group=GROUP,training_samples=761,steps_per_epoch=48,tail_samples=9,class_count=10,manifest_sha256=cfg['subset_manifest_sha256']);assert cfg['maximum_epochs']==60;return cfg
def sources():
    src=frozen.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src
def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def cpu_ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();engine.verify_sources(p['sources'],p['git_commit'])
slots=frozen.slots
@lru_cache(None)
def cache_identity():
    cfg=config();assert s.sha(cfg['data'])==cfg['cache_sha256'];return True
def load(role):
    assert role in ('train','val','test');cfg=config();m=manifest();cache_identity()
    with np.load(cfg['data'],allow_pickle=False) as z:x,y,ids=[z[role+'_'+k].copy() for k in ('images','labels','ids')]
    idx=np.asarray(cfg['original_indices']);keep=np.isin(y,idx);mapping=np.full(20,-1,np.int64);mapping[idx]=np.arange(10);x,y,ids=x[keep],mapping[y[keep]],ids[keep]
    assert ids.tolist()==m['split_ids'][role] and np.bincount(y,minlength=10).tolist()==m['supports'][role]
    assert x.dtype==np.uint8 and x.shape==(len(y),150,150,3)
    return torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids
def mixed(data,idx,theta,epoch,batch,cfg,device):
    flip,mask=frozen.b.augment.augmentation_masks(len(idx),epoch,batch,cfg);images=data[0][idx].to(device);flag=torch.as_tensor(flip,device=device);images=torch.where(flag[:,None,None,None],images.flip(-1),images)
    x=encode(images,torch.as_tensor(theta[idx],device=device),100);y=torch.nn.functional.one_hot(data[1][idx].to(device),10).float()*(1-cfg['label_smoothing'])+cfg['label_smoothing']/10
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261013,4]));lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rng.permutation(len(idx));q=torch.as_tensor(perm,device=device)
    return lam*x+(1-lam)*x[q],lam*y+(1-lam)*y[q],torch.as_tensor(mask,device=device),dict(mix=s.digest(np.r_[lam,perm.astype(float)].astype('<f8')),flip=s.digest(flip),mask=s.digest(mask))
loss=frozen.loss;save_np=frozen.save_np
def gradients(model,x,y,mask,weights,cfg,micro):
    model.train();model.zero_grad(set_to_none=True);ps=[];cs=[]
    for off in range(0,len(x),micro):
        model.linear.mask=None if mask is None else mask[off:off+micro];p,c=model(x[off:off+micro]);(loss(model,p,c,y[off:off+micro],weights,cfg)*len(p)/len(x)).backward();ps.append(p.detach());cs.append(c.detach())
    model.linear.mask=None
    return torch.cat(ps),torch.cat(cs),{n:p.grad.detach().cpu().numpy().astype(np.float64) for n,p in model.named_parameters()}
def gate(depth,rho,data,device):
    cfg=config();engine.seed(cfg['seed']);model=RegularizedMoE(depth,rho).to(device);initial=s.digest(model.state_dict());weights=(len(data[1])/(10*torch.bincount(data[1],minlength=10).float())).to(device)
    order,theta=engine.epoch_plan(1,len(data[1]),cfg);checks=[]
    groups=[('all10',np.array([int((data[1]==k).nonzero()[0]) for k in range(10)]),0),('macro16',order[:16],0),('actual_tail9',order[-9:],47)]
    for name,idx,number in groups:
        x,y,mask,_=mixed(data,idx,theta,1,number,cfg,device)
        for method in ('ordinary','dropout'):
            used=None if method=='ordinary' else mask;p,c,g=gradients(model,x,y,used,weights,cfg,1);q,d,h=gradients(model,x,y,used,weights,cfg,4)
            norms={n:float(np.linalg.norm(v.ravel())) for n,v in h.items()};assert all(np.isfinite(v).all() and norms[n]>0 for n,v in h.items())
            rel=max(float(np.linalg.norm((g[n]-v).ravel())/np.linalg.norm(v.ravel())) for n,v in h.items());cosine=min(float(np.dot(g[n].ravel(),v.ravel())/(np.linalg.norm(g[n].ravel())*np.linalg.norm(v.ravel()))) for n,v in h.items())
            err=float((p-q).abs().max());cerr=float((c-d).abs().max());assert err<=2e-6 and cerr<=2e-6 and rel<=.02 and cosine>=.999,(name,method,err,cerr,rel,cosine)
            checks.append(dict(samples=name,n=len(idx),method=method,probability_max_error=err,capture_max_error=cerr,gradient_max_relative_l2=rel,gradient_min_cosine=cosine,gradient_norms=norms))
    assert s.digest(model.state_dict())==initial;result=dict(passed=True,depth=depth,rho=rho,initial_parameter_sha256=initial,contract=model.contract(),checks=checks,optimizer_steps=0,test_arrays_read=0);del model;torch.cuda.empty_cache();return result
@torch.no_grad()
def evaluate(model,data,micro=4):
    model.eval();ps=[];fs=[];cs=[];ws=[];probe=None;device=model.raw_router.device
    for idx in torch.arange(len(data[1])).split(micro):
        p,c=model(encode(data[0][idx].to(device),None,100));ps.append(p.cpu().numpy());fs.append(model.readout['features'].cpu().numpy());cs.append(c.cpu().numpy());ws.append(model.routing_weights.cpu().numpy())
        if probe is None:probe=model.readout['amplitude'][:2].cpu().numpy()
    p=np.concatenate(ps).astype(np.float64);w=np.concatenate(ws).astype(np.float64);v=metrics(data[1].numpy(),p);v.update(detector_capture=float(np.concatenate(cs).mean()),alphas=model.alphas(),routing_weight_mean=w.mean(0).tolist(),routing_weight_std=w.std(0).tolist())
    return v,dict(probabilities=p,features=np.concatenate(fs),probe_amplitudes=probe,labels=data[1].numpy(),ids=np.asarray(data[2]).astype(str))
dataset=types.SimpleNamespace(manifest=manifest);b=types.SimpleNamespace(encode=encode)


def fit(a):
    cfg=config();assert a.case in CASES and a.gpu in GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    assert not a.out.exists();cpu_ready();slots();claim(a.out.with_name(a.out.name+'.claim.json'),'own_fit')
    s.gpu_budget.ALLOWED.update(GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    depth,rho=CASES[a.case];data,val=load('train'),load('val');a.out.mkdir()
    meta=dict(identity(),pid=os.getpid(),gpu_uuid=a.gpu,policy=policy,case=a.case,config=cfg,depth=depth,rho=rho,test_arrays_read=0)
    s.save(a.out/'metadata.json',meta,exclusive=True);g=gate(depth,rho,data,'cuda');s.save(a.out/'numeric_gate.json',g,exclusive=True)
    engine.seed(cfg['seed']);model=RegularizedMoE(depth,rho).cuda();initial=s.digest(model.state_dict());assert initial==g['initial_parameter_sha256']
    ema=copy.deepcopy(model).eval();phase=[];electronic=[]
    for name,p in model.named_parameters():(phase if name=='raw_router' or name.startswith('phase_sets.') else electronic).append(p)
    optimizer=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['maximum_epochs'],eta_min=cfg['minimum_lr'])
    weights=(len(data[1])/(10*torch.bincount(data[1],minlength=10).float())).cuda();history=[];best=float('inf');started=time.monotonic();steps=0
    for epoch in range(1,cfg['maximum_epochs']+1):
        if GROUP=='pilot' and (OUT/'early_review_stop.json').exists():break
        assert time.monotonic()-started<cfg['maximum_seconds'],'Full60 budget guard reached: no retry or incomplete acceptance'
        begin=time.monotonic();model.train();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);record={k:[] for k in ('mix','flip','mask')};lr=[q['lr'] for q in optimizer.param_groups];losses=[]
        for batch,start in enumerate(range(0,len(order),16)):
            idx=order[start:start+16];x,y,mask,hashes=mixed(data,idx,theta,epoch,batch,cfg,'cuda')
            for kind,h in hashes.items():record[kind].append(h)
            optimizer.zero_grad(set_to_none=True);lv=0.
            for off in range(0,len(idx),4):
                model.linear.mask=mask[off:off+4];p,c=model(x[off:off+4]);value=loss(model,p,c,y[off:off+4],weights,cfg)*len(p)/len(idx);value.backward();lv+=float(value.detach())
            model.linear.mask=None
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            previous_final_ema={n:p.detach().cpu().clone() for n,p in ema.named_parameters()} if epoch==cfg['maximum_epochs'] and batch==cfg['steps_per_epoch']-1 else None
            optimizer.step();steps+=1
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(cfg['ema_decay']).add_(mp,alpha=1-cfg['ema_decay'])
            losses.append(lv)
        tm,tr=evaluate(ema,data);vm,vr=evaluate(ema,val)
        entry=dict(epoch=epoch,train=tm,val=vm,lr=lr,loss=float(np.mean(losses)),order_sha256=s.digest(order),affine_sha256=s.digest(theta),mix_sha256=hashlib.sha256(''.join(record['mix']).encode()).hexdigest(),adam_steps=steps,raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin,seconds=time.monotonic()-started)
        entry.update({kind+'_sha256':hashlib.sha256(''.join(record[kind]).encode()).hexdigest() for kind in ('flip','mask')});history.append(entry)
        if vm['balanced_nll']<best:
            best=vm['balanced_nll'];engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,config=cfg,sources=meta['sources'],initial_digest=initial,contract=ema.contract(),metrics=dict(train=tm,val=vm)))
            save_np(a.out/'selected_train.npz',tr);save_np(a.out/'selected_val.npz',vr)
        scheduler.step()
        engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous_final_ema,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,steps=steps,config=cfg,sources=meta['sources'],initial_digest=initial))
        s.save(a.out/'history.json',history);s.save(a.out/'status.json',dict(state='training',epoch=epoch,train=tm,val=vm,selected_epoch=min(history,key=lambda h:h['val']['balanced_nll'])['epoch'],steps=steps))
        print(json.dumps(dict(case=a.case,epoch=epoch,train=tm['accuracy'],val=vm['accuracy'],bNLL=vm['balanced_nll'],alphas=vm['alphas'],seconds=entry['epoch_seconds'])),flush=True)
    if GROUP=='pilot' and (OUT/'early_review_stop.json').exists() and len(history)<60:
        s.save(a.out/'pilot_review_result.json',dict(baseline_complete=False,epochs_completed=len(history),steps=steps,reason='predeclared_obvious_control_ceiling',TEST_evaluations=0),exclusive=True)
        s.save(a.out/'status.json',dict(state='early_pilot_review_not_complete60',epoch=len(history),steps=steps));identity();return
    assert steps==cfg['maximum_epochs']*cfg['steps_per_epoch'];identity();selected=min(history,key=lambda h:h['val']['balanced_nll'])
    s.save(a.out/'fit_result.json',dict(passed=True,case=a.case,epochs_completed=cfg['maximum_epochs'],selected_epoch=selected['epoch'],metrics=dict(train=selected['train'],val=selected['val']),initial_digest=initial,steps=steps,stop_reason='maximum_epochs',fit_seconds=time.monotonic()-started,best_sha256=s.sha(a.out/'best.pt'),last_sha256=s.sha(a.out/'last.pt'),history_sha256=s.sha(a.out/'history.json'),test_evaluations=0),exclusive=True)
    s.save(a.out/'status.json',dict(state='full60_complete',epoch=cfg['maximum_epochs'],steps=steps))

def campaign(a):
    assert a.out==OUT and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists();cpu_ready();assert slots()['total']==0
    claim(OUT.with_name(OUT.name+'.campaign_claim.json'),'fresh_group_arms_equal60_no_automatic_TEST');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),cases=CASES),exclusive=True)
    pending=list(CASES);active=[];done=[];failures=[]
    while pending or active:
        if GROUP=='pilot' and not (OUT/'early_review_stop.json').exists():
            histories=[s.read(OUT/key/'history.json') if (OUT/key/'history.json').exists() else [] for key in ('L2_rho03','L2_rho0')]
            common=min(map(len,histories))
            if 10<=common<60:
                r,c=[hist[common-1] for hist in histories]
                if c['val']['accuracy']>=.88 and r['val']['accuracy']-c['val']['accuracy']<.03:
                    s.save(OUT/'early_review_stop.json',dict(common_epoch=common,residual_train=r['train']['accuracy'],control_train=c['train']['accuracy'],residual_val=r['val']['accuracy'],control_val=c['val']['accuracy'],reason='predeclared_ceiling_not_error',TEST_evaluations=0),exclusive=True)
        for job in list(active):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();rec={k:v for k,v in job.items() if k not in ('process','log')};rec.update(returncode=rc,actual_exit=True);s.save(Path(job['out']).with_name(job['case']+'.fit_exit.json'),rec,exclusive=True);(failures if rc else done).append(rec);active.remove(job)
        if failures:pending=[]
        for card,gpu in GPU.items():
            if not pending or failures:break
            live=slots()
            if live['total']>=4 or live['counts'][gpu] or any(j['gpu']==gpu for j in active):continue
            s.gpu_budget.ALLOWED.update(GPU.values())
            try:s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            key=pending.pop(0);out=OUT/key;log=(OUT/('fit_'+key+'.log')).open('x');cmd=[sys.executable,str(Path(__file__).resolve()),'--phase','fit','--case',key,'--gpu',gpu,'--out',str(out)];p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);active.append(dict(case=key,gpu=gpu,pid=p.pid,out=str(out),process=p,log=log))
        s.save(OUT/'status.json',dict(state='fit_failed' if failures else 'fit_running' if pending or active else 'fit_complete',active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending,complete=done,failures=failures))
        if active or pending:time.sleep(3)
    assert not failures and len(done)==len(CASES),failures
    if (OUT/'early_review_stop.json').exists():
        s.save(OUT/'status.json',dict(state='pilot_ceiling_strategy_review_no_TEST',active=[],pending=[],failures=[],complete=done,baseline_complete=False));return
    claim(OUT/'fit_audit_claim.json','own_saved_math_audit60')
    with (OUT/'fit_audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech10_equal60_hybrid.py'),'--base',str(OUT),'--out',str(OUT/'science'),'--phase','fit'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(OUT/'science/audit.json')['passed'];s.save(OUT/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(OUT/'science/audit.json')),exclusive=True)
    s.save(OUT/'status.json',dict(state='equal60_science_complete_awaiting_validation_review_no_TEST',active=[],pending=[],failures=[],complete=done,controls_frozen_at60=True))

def supervise(a):
    assert a.out==OUT and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists();cpu_ready();assert slots()['total']==0;claim(OUT.with_name(OUT.name+'.supervisor_claim.json'),'single_parent_no_retry')
    with OUT.with_name(OUT.name+'.campaign.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(OUT)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);s.save(OUT.with_name(OUT.name+'.started.json'),dict(supervisor_pid=os.getpid(),campaign_pid=p.pid,command=p.args),exclusive=True);rc=p.wait()
    s.save(OUT/'campaign_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0

def test(a):
    assert a.case in CASES and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in GPU.values();root=a.out.parent.parent;receipt=s.read(root/'audit_completed.json')
    assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(root/'science/audit.json') and s.read(root/'science/audit.json')['passed']
    assert not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'once_selected_TEST');slots();s.gpu_budget.ALLOWED.update(GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);a.out.mkdir()
    checkpoint=torch.load(root/a.case/'best.pt',map_location='cpu',weights_only=False);depth,rho=CASES[a.case];model=RegularizedMoE(depth,rho).cuda();model.load_state_dict(checkpoint['model'],strict=True)
    s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),selected_sha256=s.sha(root/a.case/'best.pt'),model_digest=s.digest(model.state_dict()),selected_epoch=checkpoint['epoch'],policy=policy,test_evaluations=1),exclusive=True)
    metrics,saved=evaluate(model,load('test'));save_np(a.out/'predictions.npz',saved);s.save(a.out/'result.json',dict(passed=True,metrics=metrics,selected_epoch=checkpoint['epoch'],test_evaluations=1,test_used_for_selection=False,test_informed_development=True),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise
