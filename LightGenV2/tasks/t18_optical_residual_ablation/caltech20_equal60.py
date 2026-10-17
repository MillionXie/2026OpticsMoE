"""Explicit paired60 Caltech experiment; frozen historical kernels are reused."""
import argparse,copy,hashlib,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit as b
import caltech20_alt246_richsplit_data as dataset
HERE,ROOT,s,engine=dataset.HERE,dataset.ROOT,dataset.s,dataset.engine
RegularizedMoE=b.RegularizedMoE
RUN='caltech20_rich20_current246_s17_equal60_20261017'
OUT=HERE/'runs/simulation'/RUN
CPU=HERE/'runs/smoke/caltech20_equal60_cpu_20261017'
FILES=('caltech20_equal60.py', 'caltech20_equal60_profile.json', 'check_caltech20_equal60.py', 'audit_caltech20_equal60.py', 'CALTECH20_EQUAL60_PROTOCOL_20261017.md')
GPU={0:'GPU-afc19890-6209-ee4d-622d-e619da5bd5b2',1:'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',3:'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd',4:'GPU-1b963983-7909-af6e-0528-f0f0661ab549'}
CASES={f'L{d}_rho{tag}':(d,rho) for d in (6,4,2) for tag,rho in [('03',.3),('0',0.)]}
def config():
    cfg=s.read(HERE/FILES[1]);m=dataset.manifest();cfg.update(data=str(Path(cfg['data_directory'])/'caltech20_rich_split.npz'),cache_sha256=m['cache_sha256'],manifest_sha256=s.sha(Path(cfg['data_directory'])/'data_manifest.json'),training_samples=len(m['split_ids']['train']),steps_per_epoch=math.ceil(len(m['split_ids']['train'])/16),tail_samples=len(m['split_ids']['train'])%16 or 16)
    assert cfg['maximum_epochs']==60 and cfg['training_samples']==3141 and cfg['classes']==m['classes'] and cfg['alpha_bounds']==[.7001,.95];return cfg
def sources():
    src=dataset.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src
def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def cpu_ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();engine.verify_sources(p['sources'],p['git_commit'])
b.dataset=dataset;b.config=config;b.sources=sources;b.identity=identity;b.claim=claim;b.cpu_ready=cpu_ready;b.GPU=GPU;b.CASES=CASES
slots,load,gate,evaluate,save_np,loss,mixed=b.slots,b.load,b.gate,b.evaluate,b.save_np,b.loss,b.mixed

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
    weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda();history=[];best=float('inf');started=time.monotonic();steps=0
    for epoch in range(1,cfg['maximum_epochs']+1):
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
    assert steps==cfg['maximum_epochs']*cfg['steps_per_epoch'];identity();selected=min(history,key=lambda h:h['val']['balanced_nll'])
    s.save(a.out/'fit_result.json',dict(passed=True,case=a.case,epochs_completed=cfg['maximum_epochs'],selected_epoch=selected['epoch'],metrics=dict(train=selected['train'],val=selected['val']),initial_digest=initial,steps=steps,stop_reason='maximum_epochs',fit_seconds=time.monotonic()-started,best_sha256=s.sha(a.out/'best.pt'),last_sha256=s.sha(a.out/'last.pt'),history_sha256=s.sha(a.out/'history.json'),test_evaluations=0),exclusive=True)
    s.save(a.out/'status.json',dict(state='full60_complete',epoch=cfg['maximum_epochs'],steps=steps))

def campaign(a):
    assert a.out==OUT and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists();cpu_ready();assert slots()['total']==0
    claim(OUT.with_name(OUT.name+'.campaign_claim.json'),'fresh_six_arms_equal60_no_automatic_TEST');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),cases=CASES),exclusive=True)
    pending=list(CASES);active=[];done=[];failures=[]
    while pending or active:
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
    assert not failures and len(done)==6,failures
    claim(OUT/'fit_audit_claim.json','own_saved_math_audit60')
    with (OUT/'fit_audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_equal60.py'),'--base',str(OUT),'--out',str(OUT/'science'),'--phase','fit'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(OUT/'science/audit.json')['passed'];s.save(OUT/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(OUT/'science/audit.json')),exclusive=True)
    s.save(OUT/'status.json',dict(state='equal60_science_complete_awaiting_validation_review_no_TEST',active=[],pending=[],failures=[],complete=done,controls_frozen_at60=True))
def supervise(a):
    assert a.out==OUT and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists();cpu_ready();assert slots()['total']==0;claim(OUT.with_name(OUT.name+'.supervisor_claim.json'),'single_parent_no_retry')
    with OUT.with_name(OUT.name+'.campaign.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(OUT)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);s.save(OUT.with_name(OUT.name+'.started.json'),dict(supervisor_pid=os.getpid(),campaign_pid=p.pid,command=p.args),exclusive=True);rc=p.wait()
    s.save(OUT/'campaign_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise
