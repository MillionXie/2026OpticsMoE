"""Fresh alternating 2/4/6 hybrid experiment; old experiments remain immutable."""
import argparse,copy,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_current4 as old
from caltech20_alt246_model import HybridMoE,CONTRACT,alpha_value
from resisc20_input_encoder import encode

HERE,ROOT=old.HERE,old.ROOT
s=old.s;engine=s.engine
PROFILE=HERE/'caltech20_alt246_profile.json'
FILES=('caltech20_alt246_model.py','caltech20_alt246.py','caltech20_alt246_profile.json','check_caltech20_alt246.py','audit_caltech20_alt246.py','CALTECH20_ALT246_PROTOCOL_20261016.md')
ABO='LightGenV2/tasks/t07_abo_image_retrieval/standalone/model.py'
RUN='caltech20_alt246_rmslearned_linear_s17_equal100_20261016'
CPU=HERE/'runs/smoke/caltech20_alt246_cpu_20261016'
GPU={0:'GPU-afc19890-6209-ee4d-622d-e619da5bd5b2',2:'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b',4:'GPU-1b963983-7909-af6e-0528-f0f0661ab549',5:'GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb'}
FORBIDDEN='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
CASES={f'L{d}_rho{r}':(d,.3 if r=='03' else 0.) for d in (4,6,2) for r in ('03','0')}

def config():
    p=s.read(PROFILE);assert p['maximum_epochs']==100 and p['batch_size']==16 and p['microbatch']==4
    assert p['alpha_bounds']==[.7001,.95] and p['automatic_retry'] is False
    return p

def sources():
    src=old.sources()
    for path in [HERE/n for n in FILES]+[ROOT/ABO]:src[path.relative_to(ROOT).as_posix()]=s.sha(path)
    return src

def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    engine.verify_sources(src,commit)
    return dict(sources=src,git_commit=commit)

def slots():
    jobs=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            cmd=path.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(HERE) not in cmd:continue
            gpu=next((v.decode().split('=',1)[1] for v in path.joinpath('environ').read_bytes().split(b'\0') if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if gpu in list(GPU.values())+[FORBIDDEN]:jobs.append(dict(pid=int(path.name),gpu_uuid=gpu))
        except (OSError,UnicodeError):continue
    counts={g:sum(j['gpu_uuid']==g for j in jobs) for g in list(GPU.values())+[FORBIDDEN]}
    assert counts[FORBIDDEN]==0 and len(jobs)<=4 and max(counts.values(),default=0)<=1,jobs
    return dict(total=len(jobs),counts=counts,jobs=jobs)

def claim(path,phase):
    s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def load(role):
    cfg=config();m=old.manifest(Path(cfg['data']))
    if role in ('train','val'):return engine.load_fit(Path(cfg['data']),role,m)
    assert role=='test'
    with np.load(cfg['data'],allow_pickle=False) as z:x,y,ids=[z[role+'_'+k].copy() for k in ('images','labels','ids')]
    assert ids.tolist()==m['split_ids'][role] and np.bincount(y,minlength=20).tolist()==m['supports'][role]
    return torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids

def mixed(data,idx,theta,epoch,batch,cfg,device):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261013,4]))
    lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rng.permutation(len(idx))
    x=encode(data[0][idx].to(device),torch.as_tensor(theta[idx],device=device),100)
    y=engine.F_onehot(data[1][idx].to(device),cfg['label_smoothing']);p=torch.as_tensor(perm,device=device)
    return lam*x+(1-lam)*x[p],lam*y+(1-lam)*y[p],s.digest(np.r_[lam,perm.astype(float)].astype('<f8'))

def loss(model,p,c,y,weights,cfg):
    return -(y*weights[None]*p.clamp_min(1e-12).log()).sum(1).mean()-cfg['capture_weight']*c.clamp_min(1e-12).log().mean()+cfg['phase_smooth_weight']*model.phase_smoothness()

def gradients(model,x,y,weights,cfg,micro):
    model.zero_grad(set_to_none=True);ps=[];cs=[]
    for start in range(0,len(x),micro):
        p,c=model(x[start:start+micro]);(loss(model,p,c,y[start:start+micro],weights,cfg)*len(p)/len(x)).backward();ps.append(p.detach());cs.append(c.detach())
    return torch.cat(ps),torch.cat(cs),{n:v.grad.detach().clone() for n,v in model.named_parameters()}

def gate(depth,rho,data,device):
    cfg=config();engine.seed(cfg['seed']);model=HybridMoE(depth,rho).to(device);initial=s.digest(model.state_dict())
    weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device)
    order,theta=engine.epoch_plan(1,len(data[1]),cfg);checks=[]
    groups=[('all20',np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]),0),('macro16',order[:16],0),('tail12',order[-12:],85)]
    for name,idx,b in groups:
        x,y,_=mixed(data,idx,theta,1,b,cfg,device);p,c,g=gradients(model,x,y,weights,cfg,1);q,d,h=gradients(model,x,y,weights,cfg,4)
        rel=max(float((h[n]-v).norm()/v.norm().clamp_min(1e-12)) for n,v in g.items())
        cosine=min(float(torch.nn.functional.cosine_similarity(h[n].flatten(),v.flatten(),dim=0)) for n,v in g.items())
        norms={n:float(v.norm()) for n,v in h.items()};finite=all(torch.isfinite(v).all() and v.norm()>0 for v in h.values())
        err=float((p-q).abs().max());cerr=float((c-d).abs().max())
        assert err<=2e-6 and cerr<=2e-6 and rel<=.02 and cosine>=.999 and finite,(name,err,cerr,rel,cosine,norms)
        checks.append(dict(samples=name,probability_max_error=err,capture_max_error=cerr,gradient_max_relative_l2=rel,gradient_min_cosine=cosine,gradient_norms=norms))
    assert s.digest(model.state_dict())==initial
    result=dict(passed=True,depth=depth,rho=rho,initial_parameter_sha256=initial,contract=model.contract(),checks=checks,optimizer_steps=0,test_arrays_read=0)
    del model;torch.cuda.empty_cache();return result

@torch.no_grad()
def evaluate(model,data,micro=4):
    model.eval();ps=[];fs=[];cs=[];ws=[];probe=None;device=model.raw_router.device
    for idx in torch.arange(len(data[1])).split(micro):
        p,c=model(encode(data[0][idx].to(device),None,100));ps.append(p.cpu().numpy());fs.append(model.readout['features'].cpu().numpy());cs.append(c.cpu().numpy());ws.append(model.routing_weights.cpu().numpy())
        if probe is None:probe=model.readout['amplitude'][:2].cpu().numpy()
    p=np.concatenate(ps).astype(np.float64);w=np.concatenate(ws).astype(np.float64);v=engine.metrics(data[1].numpy(),p)
    v.update(detector_capture=float(np.concatenate(cs).mean()),alphas=model.alphas(),routing_weight_mean=w.mean(0).tolist(),routing_weight_std=w.std(0).tolist())
    saved=dict(probabilities=p,features=np.concatenate(fs),probe_amplitudes=probe,labels=data[1].numpy(),ids=np.asarray(data[2]).astype(str))
    return v,saved

def save_np(path,arrays):
    temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path)

def cpu_ready():
    proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();engine.verify_sources(proof['sources'],proof['git_commit'])

def fit(a):
    cfg=config();assert a.case in CASES and a.gpu in GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    assert not a.out.exists();cpu_ready();slots();claim(a.out.with_name(a.out.name+'.claim.json'),'own_fit')
    s.gpu_budget.ALLOWED.update(GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    depth,rho=CASES[a.case];data,val=load('train'),load('val');a.out.mkdir()
    meta=dict(identity(),pid=os.getpid(),gpu_uuid=a.gpu,policy=policy,case=a.case,config=cfg,depth=depth,rho=rho,test_arrays_read=0)
    s.save(a.out/'metadata.json',meta,exclusive=True);g=gate(depth,rho,data,'cuda');s.save(a.out/'numeric_gate.json',g,exclusive=True)
    engine.seed(cfg['seed']);model=HybridMoE(depth,rho).cuda();initial=s.digest(model.state_dict());assert initial==g['initial_parameter_sha256']
    ema=copy.deepcopy(model).eval();phase=[];electronic=[]
    for name,p in model.named_parameters():(phase if name=='raw_router' or name.startswith('phase_sets.') else electronic).append(p)
    optimizer=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=cfg['minimum_lr'])
    weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda();history=[];best=float('inf');started=time.monotonic();steps=0
    for epoch in range(1,101):
        assert time.monotonic()-started<cfg['maximum_seconds'],'Full100 budget guard reached: no retry or incomplete acceptance'
        begin=time.monotonic();model.train();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);mixhash=[];lr=[q['lr'] for q in optimizer.param_groups];losses=[]
        for batch,start in enumerate(range(0,len(order),16)):
            idx=order[start:start+16];x,y,h=mixed(data,idx,theta,epoch,batch,cfg,'cuda');mixhash.append(h);optimizer.zero_grad(set_to_none=True);lv=0.
            for off in range(0,len(idx),4):
                p,c=model(x[off:off+4]);value=loss(model,p,c,y[off:off+4],weights,cfg)*len(p)/len(idx);value.backward();lv+=float(value.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            previous_final_ema={n:p.detach().cpu().clone() for n,p in ema.named_parameters()} if epoch==100 and batch==85 else None
            optimizer.step();steps+=1
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(cfg['ema_decay']).add_(mp,alpha=1-cfg['ema_decay'])
            losses.append(lv)
        tm,tr=evaluate(ema,data);vm,vr=evaluate(ema,val)
        entry=dict(epoch=epoch,train=tm,val=vm,lr=lr,loss=float(np.mean(losses)),order_sha256=s.digest(order),affine_sha256=s.digest(theta),mix_sha256=hashlib.sha256(''.join(mixhash).encode()).hexdigest(),adam_steps=steps,raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin,seconds=time.monotonic()-started)
        history.append(entry)
        if vm['balanced_nll']<best:
            best=vm['balanced_nll'];engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,config=cfg,sources=meta['sources'],initial_digest=initial,contract=ema.contract(),metrics=dict(train=tm,val=vm)))
            save_np(a.out/'selected_train.npz',tr);save_np(a.out/'selected_val.npz',vr)
        scheduler.step()
        engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous_final_ema,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,steps=steps,config=cfg,sources=meta['sources'],initial_digest=initial))
        s.save(a.out/'history.json',history);s.save(a.out/'status.json',dict(state='training',epoch=epoch,train=tm,val=vm,selected_epoch=min(history,key=lambda h:h['val']['balanced_nll'])['epoch'],steps=steps))
        print(json.dumps(dict(case=a.case,epoch=epoch,train=tm['accuracy'],val=vm['accuracy'],bNLL=vm['balanced_nll'],alphas=vm['alphas'],seconds=entry['epoch_seconds'])),flush=True)
    assert steps==8600;identity();selected=min(history,key=lambda h:h['val']['balanced_nll'])
    s.save(a.out/'fit_result.json',dict(passed=True,case=a.case,epochs_completed=100,selected_epoch=selected['epoch'],metrics=dict(train=selected['train'],val=selected['val']),initial_digest=initial,steps=steps,stop_reason='maximum_epochs',fit_seconds=time.monotonic()-started,best_sha256=s.sha(a.out/'best.pt'),last_sha256=s.sha(a.out/'last.pt'),history_sha256=s.sha(a.out/'history.json'),test_evaluations=0),exclusive=True)
    s.save(a.out/'status.json',dict(state='full100_complete',epoch=100,steps=steps))

def test(a):
    assert a.case in CASES and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in GPU.values()
    root=a.out.parent.parent;science=s.read(root/'science/audit.json');receipt=s.read(root/'audit_completed.json')
    assert science['passed'] and receipt['audit_sha256']==s.sha(root/'science/audit.json') and receipt['actual_audit_returncode']==0
    assert not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'once_selected_test');slots()
    s.gpu_budget.ALLOWED.update(GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    a.out.mkdir();checkpoint=torch.load(root/a.case/'best.pt',map_location='cpu',weights_only=False);depth,rho=CASES[a.case];model=HybridMoE(depth,rho).cuda();model.load_state_dict(checkpoint['model'],strict=True)
    s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),selected_sha256=s.sha(root/a.case/'best.pt'),model_digest=s.digest(model.state_dict()),selected_epoch=checkpoint['epoch'],policy=policy,test_evaluations=1),exclusive=True)
    metrics,saved=evaluate(model,load('test'));save_np(a.out/'predictions.npz',saved)
    s.save(a.out/'result.json',dict(passed=True,metrics=metrics,selected_epoch=checkpoint['epoch'],test_evaluations=1,test_used_for_selection=False,test_informed_development=True),exclusive=True)

def launch_child(root,phase,key,gpu,out):
    log=(root/(phase+'_'+key+'.log')).open('x');cmd=[sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--case',key,'--gpu',gpu,'--out',str(out)]
    proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
    return dict(case=key,gpu=gpu,pid=proc.pid,process=proc,log=log,out=str(out))

def run_queue(root,phase):
    pending=list(CASES);active=[];complete=[];failures=[]
    while pending or active:
        for j in list(active):
            rc=j['process'].poll()
            if rc is None:continue
            j['log'].close();rec={k:v for k,v in j.items() if k not in ('process','log')};rec.update(returncode=rc,actual_exit=True)
            s.save(Path(j['out']).with_name(Path(j['out']).name+'.'+phase+'_exit.json'),rec,exclusive=True)
            (complete if rc==0 else failures).append(rec);active.remove(j)
        if failures:pending=[]
        for card in (2,5,4,0):
            if not pending or failures:break
            gpu=GPU[card];live=slots()
            if live['total']>=4 or live['counts'][gpu] or any(j['gpu']==gpu for j in active):continue
            s.gpu_budget.ALLOWED.update(GPU.values())
            try:s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            key=pending.pop(0);out=root/key if phase=='fit' else root/'once_tests'/key
            active.append(launch_child(root,phase,key,gpu,out))
        state=dict(state=phase+'_failed' if failures else phase+'_running' if active or pending else phase+'_complete',active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending,complete=complete,failures=failures)
        s.save(root/'status.json',state)
        if active or pending:time.sleep(5)
    assert not failures,failures
    return complete

def audit_child(root,phase):
    claim(root/(phase+'_audit_claim.json'),phase+'_independent_saved_math')
    out=root/('science' if phase=='fit' else 'saved_test_audit')
    with (root/(phase+'_audit.log')).open('x') as log:
        rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246.py'),'--base',str(root),'--phase',phase,'--out',str(out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Own independent saved audit failed; no retry'
    s.save(root/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/'audit.json')),exclusive=True)

def campaign(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==HERE/'runs/simulation'/RUN and not a.out.exists()
    cpu_ready();slots();claim(a.out.with_name(a.out.name+'.campaign_claim.json'),'new_equal100_hybrid')
    a.out.mkdir();s.save(a.out/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),cases=CASES),exclusive=True)
    fitted=run_queue(a.out,'fit');assert len(fitted)==6;audit_child(a.out,'fit')
    (a.out/'once_tests').mkdir();tested=run_queue(a.out,'test');assert len(tested)==6;audit_child(a.out,'test')
    comparisons=[]
    for d in (2,4,6):
        pair={}
        for key in (f'L{d}_rho03',f'L{d}_rho0'):
            f=s.read(a.out/key/'fit_result.json');t=s.read(a.out/'once_tests'/key/'result.json')
            pair[key]=dict(actual_epochs=100,selected_epoch=f['selected_epoch'],**f['metrics'],test=t['metrics'])
        gap=100*(pair[f'L{d}_rho03']['test']['accuracy']-pair[f'L{d}_rho0']['test']['accuracy']);comparisons.append(dict(depth=d,models=pair,test_gap_pp=gap,preferred_gap_above10=gap>10))
    s.save(a.out/'comparison.json',dict(comparisons=comparisons,test_informed_development=True,controls_frozen_at100=True),exclusive=True)
    s.save(a.out/'status.json',dict(state='equal100_science_and_frozen_tests_complete',active=[],pending=[],failures=[],complete=fitted,test_complete=tested))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        dest=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not dest.exists():s.save(dest,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise

if __name__=='__main__':main()
