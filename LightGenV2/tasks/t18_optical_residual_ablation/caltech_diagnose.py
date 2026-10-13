"""Bounded diagnostic queue/fit; never tests and never launches formal training."""
import argparse,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import caltech_deploy234 as common
import gpu_budget
from caltech_diagnose_model import build,encode

PLAN=HERE/'caltech_diagnose_plan.json'
ALLOWED={2:'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b',
         5:'GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb',
         6:'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'}
read,save,sha,digest=common.read,common.save,common.sha,common.digest


def sources():
    import fast_optical as old
    result=old.optical.source_identity(HERE/'caltech_selected_profile.json')
    for p in sorted(HERE.glob('*.py')):result[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in [PLAN,HERE/'caltech_deploy234_profile.json',ROOT/'LightGenV2/tasks/t07_abo_image_retrieval/standalone/optics.py',ROOT/'LightGenV2/tasks/t07_abo_image_retrieval/standalone/__init__.py']:
        result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result


def verify(src,commit):common.verify_sources(src,commit)


EXECUTION_KEYS={'allowed_physical_gpu_indices','prefer_physical_gpu_indices'}
HANDOFF_FILES={'caltech_diagnose.py','audit_caltech_diagnose.py','caltech_diagnose_plan.json'}


def scientific_config(cfg):return {k:v for k,v in cfg.items() if k not in EXECUTION_KEYS}


def verify_parent(meta):
    """Original Git blobs stay authoritative; only the declared handoff files change."""
    for name,h in meta['sources'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',meta['git_commit']+':'+name],cwd=ROOT)).hexdigest()==h,name
        if Path(name).name not in HANDOFF_FILES:assert sha(ROOT/name)==h,name


def tensor_tree_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(tensor_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(tensor_tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def resume_state(parent,cfg):
    """Read a coherent completed prefix; never open test arrays or rerun its epochs."""
    meta=read(parent.parent/'metadata.json');exit_receipt=read(parent.parent/'execution_handoff_exit.json')
    assert exit_receipt['all_original_processes_exited']
    for pid in [exit_receipt['parent_pid']]+[j['pid'] for j in exit_receipt['jobs']]:
        proc=Path('/proc')/str(pid)
        assert not proc.exists() or proc.joinpath('stat').read_text().split()[2]=='Z',pid
    verify_parent(meta)
    job=next(j for j in exit_receipt['jobs'] if j['variant']==cfg['variant'])
    for file,key in [('history.json','history_sha256'),('gradients.json','gradients_sha256'),('last_checkpoint.pt','last_sha256'),('best_checkpoint.pt','best_sha256')]:assert sha(parent/file)==job[key],file
    last=torch.load(parent/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    best=torch.load(parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    hist=read(parent/'history.json');gr=read(parent/'gradients.json');smoke=read(parent/'smoke.json')
    assert scientific_config(last['config'])==scientific_config(best['config'])==scientific_config(cfg)
    assert last['sources']==best['sources']==meta['sources'] and last['data_sha256']==best['data_sha256']==cfg['cache_sha256']
    assert last['epoch']==job['epoch']==len(hist)==len(gr)==len(last['orders'])==len(last['transforms'])
    assert [h['epoch'] for h in hist]==list(range(1,last['epoch']+1))
    assert {int(s['step']) for s in last['optimizer']['state'].values()}=={86*last['epoch']}
    assert min(hist,key=lambda h:h['val']['balanced_nll'])['epoch']==best['epoch']
    return meta,last,best,hist,gr,smoke,job


def restore(model,ema,opt,last):
    model.load_state_dict(last['model']);ema.load_state_dict(last['ema']);opt.load_state_dict(last['optimizer'])
    assert tensor_tree_equal(model.state_dict(),last['model']) and tensor_tree_equal(ema.state_dict(),last['ema'])
    assert tensor_tree_equal(opt.state_dict(),last['optimizer'])


def settings(variant):
    cfg=read(PLAN)
    assert variant in cfg['variants']
    cfg=dict(cfg,variant=variant)
    cfg['mixup_alpha']={'mixup02':.2,'mixup06':.6}.get(variant,cfg['mixup_alpha'])
    return cfg


def dataset(cfg):
    data=Path(cfg['data']);m=read(data.parent/'data_manifest.json')
    assert sha(data)==cfg['cache_sha256'] and sha(data.parent/'data_manifest.json')==cfg['manifest_sha256']
    assert m['cache_sha256']==cfg['cache_sha256'] and len(m['classes'])==20
    return m,[common.load_fit(data,s,m) for s in ('train','val')]


def mixed(data,idx,theta,epoch,batch,cfg,device):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261013,4]))
    lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam)
    perm=rng.permutation(len(idx));x=encode(data[0][idx].to(device),torch.as_tensor(theta[idx],device=device),cfg['variant'])
    y=common.F_onehot(data[1][idx].to(device),cfg['label_smoothing']);order=torch.as_tensor(perm,device=device)
    return lam*x+(1-lam)*x[order],lam*y+(1-lam)*y[order],digest(np.r_[lam,perm.astype(float)].astype('<f8'))


def objective(model,p,c,y,w,cfg):return common.objective(model,p,c,y,w,cfg)


@torch.no_grad()
def evaluate(model,data,cfg):
    model.eval();device=next(model.parameters()).device;ps=[];captures=[]
    for idx in torch.arange(len(data[1])).split(1):
        p,c=model(encode(data[0][idx].to(device),None,cfg['variant']));ps.append(p.cpu().numpy());captures.append(c.cpu().numpy())
    p=np.concatenate(ps).astype(np.float64);metric=common.metrics(data[1].numpy(),p);metric['detector_capture']=float(np.concatenate(captures).mean())
    rows=[dict(sample_id=str(sid),label_true=int(y),label_pred=int(v.argmax()),**{f'score{k}':float(v[k]) for k in range(20)}) for sid,y,v in zip(data[2],data[1],p)]
    return metric,rows


def class_health(metric,classes):
    return dict(zero_recall_classes=[classes[k] for k,v in enumerate(metric['recall']) if v==0],
        below20_classes=[classes[k] for k,v in enumerate(metric['recall']) if v<.2],minimum_recall=min(metric['recall']),
        per_class={c:dict(support=metric['support'][k],recall=metric['recall'][k]) for k,c in enumerate(classes)})


def gate(cfg,data,device,state=None):
    common.seed(cfg['seed']);model=build(cfg['variant'],cfg).to(device)
    if state is not None:model.load_state_dict(state)
    initial=digest(model.state_dict());w=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device)
    idx=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)])
    _,theta=common.epoch_plan(1,len(data[1]),cfg);x,y,_=mixed(data,idx,theta,1,0,cfg,device)
    model.train();probs=[]
    for k in range(20):
        p,c=model(x[k:k+1]);(objective(model,p,c,y[k:k+1],w,cfg)/20).backward();probs.append(p.detach())
    norms={n:float(v.grad.norm()) for n,v in model.named_parameters()}
    assert all(torch.isfinite(v.grad).all() and norms[n]>0 for n,v in model.named_parameters()),norms
    before=torch.cat(probs);model.zero_grad(set_to_none=True);after=[]
    with torch.no_grad():
        for k in range(20):after.append(model(x[k:k+1])[0])
    error=float((before-torch.cat(after)).abs().max());assert error<=1e-6,error
    out=dict(passed=True,variant=cfg['variant'],initial_parameter_sha256=initial,probability_repeat_error=error,
        all_phase_gradients_finite_nonzero=True,phase_gradients=norms,parameters=sum(v.numel() for v in model.parameters()),
        microbatch=1,classes_checked=20,test_arrays_materialized=0,test_evaluations=0,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else 0)
    del model;torch.cuda.empty_cache() if device=='cuda' else None
    return out


def prepare_gpu(gpu):
    assert gpu in ALLOWED.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu
    gpu_budget.ALLOWED.update(ALLOWED.values())
    policy=gpu_budget.configure(torch,gpu,shared=True,memory_gib=2)
    torch.set_num_threads(4)
    return policy


def preflight(a):
    policy=prepare_gpu(a.gpu);cfg=settings('base');m,(data,val)=dataset(cfg)
    a.out.mkdir(parents=True,exist_ok=False);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();src=sources();verify(src,commit)
    save(a.out/'metadata.json',dict(pid=os.getpid(),gpu_uuid=a.gpu,policy=policy,sources=src,git_commit=commit))
    results=[]
    for v in cfg['variants']:
        variant_cfg=settings(v);variant_cfg['classes']=m['classes']
        result=gate(variant_cfg,data,'cuda');results.append(result);save(a.out/'checks.json',dict(passed=False,results=results,test_evaluations=0))
    # The independent base subclass must preserve the sealed graph and state exactly.
    from caltech_deploy234_model import DeployMoE,input_amplitude
    common.seed(cfg['seed']);reference=DeployMoE(4,.3).cuda()
    common.seed(cfg['seed']);base=build('base',cfg).cuda();assert digest(reference.state_dict())==digest(base.state_dict())
    with torch.no_grad():
        x=input_amplitude(data[0][:1].cuda());p,c=reference(x);q,d=base(x)
    assert torch.equal(p,q) and torch.equal(c,d)
    save(a.out/'checks.json',dict(passed=True,results=results,base_frozen_exact_identity=True,test_arrays_materialized=0,test_evaluations=0))


def fit(a):
    policy=prepare_gpu(a.gpu);cfg=settings(a.variant);m,(data,val)=dataset(cfg);cfg['classes']=m['classes']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();src=sources();verify(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'config.json',cfg);save(a.out/'metadata.json',dict(pid=os.getpid(),gpu_uuid=a.gpu,policy=policy,sources=src,git_commit=commit,command=sys.argv,test_evaluations=0))
    smoke=gate(cfg,data,'cuda');save(a.out/'smoke.json',smoke)
    common.seed(cfg['seed']);model=build(cfg['variant'],cfg).cuda();assert digest(model.state_dict())==smoke['initial_parameter_sha256']
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=0)
    w=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda()
    history=[];gradients=[];orders=[];transforms=[];best=float('inf');prefix_seconds=0.;start_epoch=1
    if a.resume:
        parent_meta,last,parent_best,history,gradients,parent_smoke,job=resume_state(a.resume,cfg)
        assert parent_smoke['initial_parameter_sha256']==smoke['initial_parameter_sha256']
        restore(model,ema,opt,last);orders=list(last['orders']);transforms=list(last['transforms'])
        best=parent_best['validation']['balanced_nll'];prefix_seconds=history[-1]['seconds'];start_epoch=last['epoch']+1
        restored_gate=gate(cfg,data,'cuda',state=last['model']);save(a.out/'restored_gate.json',restored_gate)
        assert restored_gate['initial_parameter_sha256']==digest(last['model'])
        receipt=dict(parent=str(a.resume),parent_git_commit=parent_meta['git_commit'],parent_sources=parent_meta['sources'],
            prefix_epochs=last['epoch'],first_continued_epoch=start_epoch,parent_history_sha256=job['history_sha256'],
            parent_last_sha256=job['last_sha256'],parent_best_sha256=job['best_sha256'],parent_gradients_sha256=job['gradients_sha256'],
            raw_sha256=digest(last['model']),ema_sha256=digest(last['ema']),optimizer_exact=True,raw_exact=True,ema_exact=True,
            scheduler='none_fixed_lr_preserved',completed_steps=last['epoch']*86,initial_parameter_sha256=smoke['initial_parameter_sha256'],
            restored_gate_passed=restored_gate['passed'],completed_epochs_replayed=0,test_evaluations=0)
        save(a.out/'resume_receipt.json',receipt,exclusive=True)
        # Preserve the old selected EMA as a candidate, with explicit source provenance.
        carried=dict(parent_best,config=cfg,sources=src,parent_checkpoint_sha256=job['best_sha256'],parent_sources=parent_meta['sources'])
        common.save_pt(a.out/'best_checkpoint.pt',carried)
        common.csvwrite(a.out/'val_predictions.csv',list(__import__('csv').DictReader((a.resume/'val_predictions.csv').open())))
        save(a.out/'history.json',history);save(a.out/'gradients.json',gradients)
        common.save_pt(a.out/'last_checkpoint.pt',dict(last,config=cfg,sources=src,parent_checkpoint_sha256=job['last_sha256']))
    started=time.monotonic()
    # Fixed LR, fixed common budget isolate factors; no early stop/scheduler confounding.
    for epoch in range(start_epoch,cfg['epochs']+1):
        if prefix_seconds+time.monotonic()-started>=cfg['maximum_seconds_per_arm']:raise RuntimeError('Diagnostic time budget exhausted; retain failure, no retry')
        begin=time.monotonic();model.train();order,theta=common.epoch_plan(epoch,len(data[1]),cfg);orders.append(digest(order));mh=hashlib.sha256();total=0
        for b,start in enumerate(range(0,len(order),cfg['batch_size'])):
            idx=order[start:start+cfg['batch_size']];x,y,mix=mixed(data,idx,theta,epoch,b,cfg,'cuda');mh.update(mix.encode());opt.zero_grad(set_to_none=True)
            for k in range(len(idx)):
                p,c=model(x[k:k+1]);loss=objective(model,p,c,y[k:k+1],w,cfg);assert torch.isfinite(loss);(loss/len(idx)).backward();total+=float(loss.detach())
            if b==0:
                norms={n:float(v.grad.norm()) for n,v in model.named_parameters()};assert all(np.isfinite(v) and v>0 for v in norms.values());gradients.append(dict(epoch=epoch,norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,1-cfg['ema_decay'])
        vm,rows=evaluate(ema,val,cfg);entry=dict(epoch=epoch,train_loss=total/len(data[1]),val=vm,mixup_sha256=mh.hexdigest(),lr=cfg['lr'],validation_class_health=class_health(vm,m['classes']))
        transforms.append(hashlib.sha256((digest(theta)+mh.hexdigest()).encode()).hexdigest())
        if epoch==1 or epoch%5==0:
            entry['train']=evaluate(ema,data,cfg)[0];entry['training_class_health']=class_health(entry['train'],m['classes'])
        if vm['balanced_nll']<best:
            best=vm['balanced_nll'];common.save_pt(a.out/'best_checkpoint.pt',dict(model=ema.state_dict(),epoch=epoch,config=cfg,sources=src,validation=vm,data_sha256=m['cache_sha256'],selection_split='val',selected_state_kind='ema'));common.csvwrite(a.out/'val_predictions.csv',rows)
        entry.update(seconds=prefix_seconds+time.monotonic()-started,epoch_seconds=time.monotonic()-begin);history.append(entry)
        save(a.out/'history.json',history);save(a.out/'gradients.json',gradients)
        common.save_pt(a.out/'last_checkpoint.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),epoch=epoch,config=cfg,sources=src,orders=orders,transforms=transforms,initial_parameter_sha256=smoke['initial_parameter_sha256'],data_sha256=m['cache_sha256']))
        state=dict(state='diagnostic_training',variant=a.variant,epoch=epoch,val_accuracy=vm['accuracy'],val_macro=vm['balanced_accuracy'],val_balanced_nll=vm['balanced_nll'],class_health=entry['validation_class_health'],epoch_seconds=entry['epoch_seconds'],test_evaluations=0)
        save(a.out/'status.json',state);print(json.dumps(state),flush=True)
    ck=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);ema.load_state_dict(ck['model']);tm,rows=evaluate(ema,data,cfg);common.csvwrite(a.out/'train_predictions.csv',rows)
    verify(src,commit)
    result=dict(variant=a.variant,depth=4,rho=.3,epochs_completed=len(history),selected_epoch=ck['epoch'],metrics=dict(train=tm,val=ck['validation']),
        class_health={s:class_health(v,m['classes']) for s,v in [('train',tm),('val',ck['validation'])]},
        config=cfg,sources=src,history_sha256=sha(a.out/'history.json'),checkpoint_sha256=sha(a.out/'best_checkpoint.pt'),initial_parameter_sha256=smoke['initial_parameter_sha256'],
        seconds=prefix_seconds+time.monotonic()-started,test_evaluations=0,test_arrays_materialized=0,stop_reason='maximum_epochs',
        deployment_eligible=a.variant not in ('unquantized','legacy_reference'),scope='Diagnostic only, not formal training or an independently generalized result')
    save(a.out/'fit_result.json',result);save(a.out/'status.json',dict(state='diagnostic_fit_complete',epochs=len(history),test_evaluations=0))


def available():
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,name','--format=csv,noheader,nounits'],text=True)
    cards={int(x.split(',')[0]):(x.split(',')[1].strip(),int(x.split(',')[2]),','.join(x.split(',')[3:]).strip()) for x in info.splitlines()}
    return cards


def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=read(PLAN);proof=read(a.preflight/'checks.json');assert proof['passed'] and set(x['variant'] for x in proof['results'])==set(cfg['variants'])
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();verify(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'metadata.json',dict(pid=os.getpid(),git_commit=commit,sources=src,config=cfg,preflight=str(a.preflight),test_evaluations=0))
    if a.resume:
        handoff=read(a.resume/'execution_handoff_exit.json');assert handoff['all_original_processes_exited']
        assert set(j['variant'] for j in handoff['jobs'])|set(handoff['pending'])==set(cfg['variants'])
        verify_parent(read(a.resume/'metadata.json'));save(a.out/'parent_handoff.json',dict(parent=str(a.resume),receipt=handoff),exclusive=True)
    pending=list(cfg['variants']);active={};completed=[];failures=[];start=time.time()
    while active or (pending and not failures):
        for gpu,(proc,v) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            try:
                assert rc==0,rc;result=read(a.out/v/'fit_result.json');assert result['epochs_completed']==cfg['epochs'] and result['test_evaluations']==0
                completed.append(dict(variant=v,pid=proc.pid,gpu_uuid=gpu,exit_code=rc))
            except Exception as e:failures.append(dict(variant=v,pid=proc.pid,exit_code=rc,error=repr(e)))
            del active[gpu]
        if pending and not failures:
            cards=available()
            for index in cfg['prefer_physical_gpu_indices']:
                if len(active)>=3 or not pending:break
                gpu,free,name=cards[index];assert ALLOWED[index]==gpu
                if gpu in active or free<cfg['memory_gib']*1024+800:continue
                verify(src,commit);v=pending.pop(0);cmd=[sys.executable,str(HERE/'caltech_diagnose.py'),'--phase','fit','--variant',v,'--out',str(a.out/v),'--gpu',gpu]
                if a.resume and (a.resume/v/'last_checkpoint.pt').exists():cmd.extend(['--resume',str(a.resume/v)])
                with (a.out/(v+'.log')).open('x') as log:proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(proc,v)
        save(a.out/'current_processes.json',dict(active=[dict(variant=v,pid=p.pid,gpu_uuid=g) for g,(p,v) in active.items()],pending=pending))
        save(a.out/'status.json',dict(state='failure_draining' if failures else 'diagnostic_running',completed=completed,failures=failures,active=len(active),pending=len(pending),seconds=time.time()-start,test_evaluations=0))
        if active or (pending and not failures):time.sleep(5)
    save(a.out/'status.json',dict(state='failed_no_retry' if failures else 'diagnostics_complete_awaiting_CPU_review',completed=completed,failures=failures,active=0,pending=pending,seconds=time.time()-start,test_evaluations=0))
    if failures:raise RuntimeError(failures)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','fit','queue'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu');p.add_argument('--variant');p.add_argument('--preflight',type=Path);p.add_argument('--resume',type=Path)
    a=p.parse_args()
    try:{'preflight':preflight,'fit':fit,'queue':queue}[a.phase](a)
    except BaseException as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),pid=os.getpid(),test_evaluations=0))
        raise
