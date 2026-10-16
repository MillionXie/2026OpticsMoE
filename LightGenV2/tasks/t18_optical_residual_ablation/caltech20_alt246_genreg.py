"""Evidence-based residual-only continuation; baseline controls remain frozen."""
import argparse,copy,hashlib,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_biasfree as b
from caltech20_alt246_genreg_model import RegularizedMoE

HERE,ROOT,s,engine=b.HERE,b.ROOT,b.s,b.engine
PROFILE=HERE/'caltech20_alt246_genreg_profile.json'
FILES=('caltech20_alt246_genreg_model.py','caltech20_alt246_genreg.py','caltech20_alt246_genreg_profile.json','check_caltech20_alt246_genreg.py','audit_caltech20_alt246_genreg.py','CALTECH20_ALT246_GENREG_PROTOCOL_20261016.md')
RUN='caltech20_alt246_residual_genreg_s17_20261016'
CPU=HERE/'runs/smoke/caltech20_alt246_genreg_cpu_20261016'
BASE=HERE/'runs/simulation'/b.RUN
CASES={f'L{d}_rho03':(d,.3) for d in (2,4,6)}

def config(depth):
    p=s.read(PROFILE);cfg=dict(b.config());cfg.pop('maximum_epochs');cfg.update(p['common']);cfg.update(p['depths'][str(depth)]);cfg['main_depth']=depth
    assert cfg['block_epochs']==10 and cfg['horizontal_flip_probability']==.5 and cfg['automatic_retry'] is False
    return cfg

def sources():
    src=b.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src

def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def optimizer_moment_digest(state):
    return s.digest({str(i)+':'+n:v for i,values in state['state'].items() for n,v in values.items() if isinstance(v,torch.Tensor)})

def baseline():
    state=s.read(BASE/'status.json');assert state['state']=='equal100_science_and_frozen_tests_complete'
    pid=s.read(BASE/'metadata.json')['pid'];p=Path('/proc',str(pid),'cmdline');assert not p.exists() or b.RUN.encode() not in p.read_bytes()
    for rec,folder in (('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')):
        v=s.read(BASE/rec);assert v['actual_audit_returncode']==0 and v['audit_sha256']==s.sha(BASE/folder/'audit.json') and s.read(BASE/folder/'audit.json')['passed']
    profile=s.read(PROFILE);assert s.sha(BASE/'science/audit.json')==profile['baseline_science_sha256'] and s.sha(BASE/'saved_test_audit/audit.json')==profile['baseline_test_audit_sha256']
    assert not any((BASE/n).exists() for n in ('failure.json','fit_audit_failure.json','test_audit_failure.json'))
    meta=s.read(BASE/'metadata.json');assert meta['sources']==b.sources()
    science=s.read(BASE/'science/audit.json');tests=s.read(BASE/'saved_test_audit/audit.json')
    for key in b.CASES:
        folder=BASE/key;f=s.read(folder/'fit_result.json');assert f['epochs_completed']==100 and f['steps']==8600
        assert f['best_sha256']==s.sha(folder/'best.pt') and f['last_sha256']==s.sha(folder/'last.pt') and f['history_sha256']==s.sha(folder/'history.json')
        for role in ('train','val'):assert science['checks'][key+'_'+role]['prediction_sha256']==s.sha(folder/('selected_'+role+'.npz'))
        assert tests['checks'][key]['prediction_sha256']==s.sha(BASE/'once_tests'/key/'predictions.npz')
    return meta

def cpu_ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();engine.verify_sources(p['sources'],p['git_commit'])

def augmentation_masks(n,epoch,batch,cfg):
    flip=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261016,1])).random(n)<cfg['horizontal_flip_probability']
    keep=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261016,2])).random((n,256))>=cfg['feature_dropout']
    mask=keep.astype(np.float32)/(1-cfg['feature_dropout'])
    return flip,mask

def mixed(data,idx,theta,epoch,batch,cfg,device):
    flip,mask=augmentation_masks(len(idx),epoch,batch,cfg);images=data[0][idx].to(device);flag=torch.as_tensor(flip,device=device)
    # Flip RGB images before channel tiling, never the completed R/G/B mosaic.
    images=torch.where(flag[:,None,None,None],images.flip(-1),images)
    x=b.encode(images,torch.as_tensor(theta[idx],device=device),100);y=engine.F_onehot(data[1][idx].to(device),cfg['label_smoothing'])
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261013,4]));lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rng.permutation(len(idx));q=torch.as_tensor(perm,device=device)
    return lam*x+(1-lam)*x[q],lam*y+(1-lam)*y[q],torch.as_tensor(mask,device=device),dict(mix=s.digest(np.r_[lam,perm.astype(float)].astype('<f8')),flip=s.digest(flip),mask=s.digest(mask))

def gradients(model,x,y,mask,weights,cfg,micro):
    model.train();model.zero_grad(set_to_none=True);ps=[];cs=[]
    for off in range(0,len(x),micro):
        model.linear.mask=None if mask is None else mask[off:off+micro];p,c=model(x[off:off+micro]);(b.loss(model,p,c,y[off:off+micro],weights,cfg)*len(p)/len(x)).backward();ps.append(p.detach());cs.append(c.detach())
    model.linear.mask=None
    return torch.cat(ps),torch.cat(cs),{n:p.grad.detach().cpu().numpy().astype(np.float64) for n,p in model.named_parameters()}

def gate(depth,state,data,cfg,device,epoch):
    model=RegularizedMoE(depth).to(device);model.load_state_dict(state,strict=True);initial=s.digest(model.state_dict());weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device)
    order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);checks=[]
    for name,idx,number in [('all20',np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]),0),('macro16',order[:16],0),('tail12',order[-12:],85)]:
        x,y,mask,_=mixed(data,idx,theta,epoch,number,cfg,device)
        for method in ('ordinary','dropout'):
            used=None if method=='ordinary' else mask;p,c,g=gradients(model,x,y,used,weights,cfg,1);q,d,h=gradients(model,x,y,used,weights,cfg,4)
            norms={n:float(np.linalg.norm(v.ravel())) for n,v in h.items()};assert all(np.isfinite(v).all() and norms[n]>0 for n,v in h.items())
            relative=max(float(np.linalg.norm((g[n]-v).ravel())/np.linalg.norm(v.ravel())) for n,v in h.items())
            cosine=min(float(np.dot(g[n].ravel(),v.ravel())/(np.linalg.norm(g[n].ravel())*np.linalg.norm(v.ravel()))) for n,v in h.items())
            error=float((p-q).abs().max());cerr=float((c-d).abs().max());assert error<=2e-6 and cerr<=2e-6 and relative<=.02 and cosine>=.999,(name,method,error,cerr,relative,cosine)
            checks.append(dict(samples=name,method=method,probability_maxabs=error,capture_maxabs=cerr,gradient_relative_l2=relative,gradient_cosine=cosine,gradient_norms=norms))
    assert s.digest(model.state_dict())==initial;del model;torch.cuda.empty_cache();return dict(passed=True,initial_digest=initial,checks=checks,optimizer_updates=0,test_arrays_read=0)

def parent_info(parent,key):
    if parent==BASE:
        f=s.read(parent/key/'fit_result.json');return dict(last=str(parent/key/'last.pt'),best=str(parent/key/'best.pt'),train=str(parent/key/'selected_train.npz'),val=str(parent/key/'selected_val.npz'),selected_epoch=f['selected_epoch'],best_metrics=f['metrics'],extra_epochs=0,history=[])
    f=s.read(parent/key/'fit_result.json');last=torch.load(parent/key/'last.pt',map_location='cpu',weights_only=False)
    return dict(last=str(parent/key/'last.pt'),best=f['best_path'],train=f['selected_train_path'],val=f['selected_val_path'],selected_epoch=f['selected_epoch'],best_metrics=f['metrics'],extra_epochs=f['extra_epochs'],history=s.read(parent/key/'history_extra.json'))

def fit(a):
    baseline();cpu_ready();b.slots();assert a.key in CASES and a.gpu in b.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and not a.out.exists()
    depth,_=CASES[a.key];cfg=config(depth);parent=parent_info(a.parent,a.key);claim(a.out.with_name(a.out.name+'.claim.json'),'own_genreg_fit');a.out.mkdir()
    b.s.gpu_budget.ALLOWED.update(b.GPU.values());policy=b.s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    last=torch.load(parent['last'],map_location='cpu',weights_only=False);data,val=b.load('train'),b.load('val');start=100+parent['extra_epochs']+1
    proof=gate(depth,last['model'],data,cfg,'cuda',start);s.save(a.out/'numeric_gate.json',proof,exclusive=True)
    model=RegularizedMoE(depth).cuda();model.load_state_dict(last['model'],strict=True);ema=copy.deepcopy(model).eval();ema.load_state_dict(last['ema'],strict=True)
    assert proof['initial_digest']==s.digest(model.state_dict());parent_raw=s.digest(last['model']);parent_ema=s.digest(last['ema'])
    phase=[];electronic=[]
    for n,p in model.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(p)
    optimizer=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr'])
    optimizer.load_state_dict(last['optimizer']);steps=last['steps'];assert steps==8600+86*parent['extra_epochs']
    initial_optimizer_moments=optimizer_moment_digest(optimizer.state_dict());assert initial_optimizer_moments==optimizer_moment_digest(last['optimizer'])
    for group in optimizer.param_groups:group['lr']=cfg['lr'];group['initial_lr']=cfg['lr']
    optimizer.param_groups[0]['weight_decay']=0;optimizer.param_groups[1]['weight_decay']=cfg['electronic_weight_decay']
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3,threshold=.002,threshold_mode='abs',min_lr=2e-5)
    if a.parent!=BASE:scheduler.load_state_dict(last['scheduler']);[group.update(lr=lr) for group,lr in zip(optimizer.param_groups,last['scheduler']['_last_lr'])]
    else:scheduler.best=parent['best_metrics']['val']['balanced_nll']
    meta=dict(identity(),case=a.key,depth=depth,rho=.3,pid=os.getpid(),gpu_uuid=a.gpu,policy=policy,config=cfg,parent=parent,parent_last_sha256=s.sha(parent['last']),parent_best_sha256=s.sha(parent['best']),parent_raw_digest=parent_raw,parent_ema_digest=parent_ema,initial_optimizer_moment_digest=initial_optimizer_moments,baseline_source=baseline()['sources'])
    s.save(a.out/'metadata.json',meta,exclusive=True);history=parent['history'];weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda();best=parent['best_metrics']['val']['balanced_nll'];selected=copy.deepcopy(parent);started=time.monotonic()
    for epoch in range(start,start+10):
        assert time.monotonic()-started<cfg['maximum_seconds'],'Incomplete block: preserve failure, no automatic retry'
        begin=time.monotonic();model.train();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);record={k:[] for k in ('mix','flip','mask')};lr=[q['lr'] for q in optimizer.param_groups]
        for batch,off in enumerate(range(0,len(order),16)):
            idx=order[off:off+16];x,y,mask,hashes=mixed(data,idx,theta,epoch,batch,cfg,'cuda')
            for kind,h in hashes.items():record[kind].append(h)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0,len(idx),4):
                model.linear.mask=mask[micro:micro+4];p,c=model(x[micro:micro+4]);(b.loss(model,p,c,y[micro:micro+4],weights,cfg)*len(p)/len(idx)).backward()
            model.linear.mask=None;torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            previous_final_ema={n:p.detach().cpu().clone() for n,p in ema.named_parameters()} if epoch==start+9 and batch==85 else None
            optimizer.step();steps+=1
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(cfg['ema_decay']).add_(mp,alpha=1-cfg['ema_decay'])
        tm,tr=b.evaluate(ema,data);vm,vr=b.evaluate(ema,val);entry=dict(epoch=epoch,train=tm,val=vm,lr=lr,adam_steps=steps,order_sha256=s.digest(order),affine_sha256=s.digest(theta),**{k+'_sha256':hashlib.sha256(''.join(v).encode()).hexdigest() for k,v in record.items()},raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin)
        history.append(entry)
        if vm['balanced_nll']<best:
            best=vm['balanced_nll'];engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,config=cfg,sources=meta['sources'],contract=ema.contract(),metrics=dict(train=tm,val=vm)))
            b.save_np(a.out/'selected_train.npz',tr);b.save_np(a.out/'selected_val.npz',vr);selected.update(best=str(a.out/'best.pt'),train=str(a.out/'selected_train.npz'),val=str(a.out/'selected_val.npz'),selected_epoch=epoch,best_metrics=dict(train=tm,val=vm))
        scheduler.step(vm['balanced_nll']);engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous_final_ema,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),steps=steps,epoch=epoch,config=cfg,sources=meta['sources']))
        s.save(a.out/'history_extra.json',history);s.save(a.out/'status.json',dict(state='training',epoch=epoch,train=tm,val=vm,selected_epoch=selected['selected_epoch']));print(dict(case=a.key,epoch=epoch,train=tm['accuracy'],val=vm['accuracy'],bNLL=vm['balanced_nll']),flush=True)
    identity();f=dict(passed=True,case=a.key,actual_epochs=epoch,extra_epochs=epoch-100,selected_epoch=selected['selected_epoch'],best_path=selected['best'],selected_train_path=selected['train'],selected_val_path=selected['val'],metrics=selected['best_metrics'],steps=steps,fit_seconds=time.monotonic()-started,best_sha256=s.sha(selected['best']),last_sha256=s.sha(a.out/'last.pt'),history_sha256=s.sha(a.out/'history_extra.json'))
    s.save(a.out/'fit_result.json',f,exclusive=True);s.save(a.out/'status.json',dict(state='complete_block',epoch=epoch,selected_epoch=f['selected_epoch']))

def test(a):
    baseline();cpu_ready();b.slots();depth,_=CASES[a.key];block=a.out.parent.parent;receipt=s.read(block/'audit_completed.json');assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(block/'science/audit.json')
    f=s.read(block/a.key/'fit_result.json');checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);digest=s.digest(checkpoint['model']);root=block.parent.parent
    known=root/'test_digests'/a.key;known.mkdir(parents=True,exist_ok=True);cache=known/(digest+'.json');assert not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'selected_once_test_or_true_reuse');a.out.mkdir()
    if cache.exists():result=s.read(cache);assert result['model_digest']==digest;result=dict(result,reused=True,new_model_forwards=0)
    else:
        s.save(known/(digest+'.claim.json'),dict(model_digest=digest,case=a.key,pid=os.getpid()),exclusive=True)
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in b.GPU.values();b.s.gpu_budget.ALLOWED.update(b.GPU.values());b.s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
        model=RegularizedMoE(depth).cuda();model.load_state_dict(checkpoint['model'],strict=True);metrics,saved=b.evaluate(model,b.load('test'));b.save_np(a.out/'predictions.npz',saved)
        result=dict(metrics=metrics,model_digest=digest,predictions_path=str(a.out/'predictions.npz'),predictions_sha256=s.sha(a.out/'predictions.npz'),test_evaluations=1,reused=False,selected_epoch=f['selected_epoch'],test_informed_development=True,test_used_for_epoch_selection=False)
        s.save(cache,result,exclusive=True)
    s.save(a.out/'result.json',dict(result,selected_epoch=f['selected_epoch']),exclusive=True)

def jobs(block,phase,keys,parent):
    pending=list(keys);active=[];completed=[];failed=[]
    while pending or active:
        for j in list(active):
            rc=j['proc'].poll()
            if rc is None:continue
            j['log'].close();rec={k:v for k,v in j.items() if k not in ('proc','log')};rec.update(returncode=rc,actual_exit=True);s.save(Path(j['out']).with_name(j['key']+'.'+phase+'_exit.json'),rec,exclusive=True);(completed if rc==0 else failed).append(rec);active.remove(j)
        if failed:pending=[]
        for card in (2,4,5):
            if not pending or failed:break
            gpu=b.GPU[card];live=b.slots()
            if live['total']>=4 or live['counts'][gpu] or any(j['gpu']==gpu for j in active):continue
            b.s.gpu_budget.ALLOWED.update(b.GPU.values())
            try:b.s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            key=pending.pop(0);out=block/key if phase=='fit' else block/'once_tests'/key;log=(block/(phase+'_'+key+'.log')).open('x');cmd=[sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--key',key,'--gpu',gpu,'--out',str(out),'--parent',str(parent[key])]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);active.append(dict(key=key,gpu=gpu,pid=proc.pid,proc=proc,log=log,out=str(out)))
        s.save(block/'status.json',dict(state=phase+'_failed' if failed else phase+'_running' if pending or active else phase+'_complete',active=[{k:v for k,v in j.items() if k not in ('proc','log')} for j in active],pending=pending,completed=completed,failures=failed))
        if pending or active:time.sleep(5)
    assert not failed,failed

def audit_block(block,phase):
    claim(block/(phase+'_audit_claim.json'),'independent_genreg_'+phase);out=block/('science' if phase=='fit' else 'saved_test_audit')
    with (block/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_genreg.py'),'--base',str(block),'--out',str(out),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Own science/test saved audit failed; no retry'
    s.save(block/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/'audit.json')),exclusive=True)

def campaign(a):
    assert a.out==HERE/'runs/simulation'/RUN and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';baseline();cpu_ready();assert b.slots()['total']==0
    claim(a.out.with_name(a.out.name+'.campaign_claim.json'),'residual_only_genreg');a.out.mkdir();(a.out/'blocks').mkdir();(a.out/'test_digests').mkdir();s.save(a.out/'metadata.json',dict(identity(),pid=os.getpid(),profile=s.read(PROFILE),baseline=str(BASE)),exclusive=True)
    parents={key:BASE for key in CASES};states={key:'ready' for key in CASES};latest={}
    for key in CASES:
        (a.out/'test_digests'/key).mkdir();start=s.read(BASE/'once_tests'/key/'started.json');result=s.read(BASE/'once_tests'/key/'result.json');p=BASE/'once_tests'/key/'predictions.npz'
        value=dict(metrics=result['metrics'],model_digest=start['model_digest'],predictions_path=str(p),predictions_sha256=s.sha(p),test_evaluations=1,selected_epoch=start['selected_epoch'],reused=True,test_informed_development=True,test_used_for_epoch_selection=False)
        s.save(a.out/'test_digests'/key/(start['model_digest']+'.json'),value,exclusive=True);latest[key]=dict(test=value['metrics'],model_digest=value['model_digest'],selected_epoch=value['selected_epoch'],actual_epochs=100,extra_epochs=0)
    number=0
    while True:
        # A six-layer target is final only after the four-layer reference has stopped.
        four_final=states['L4_rho03'] in ('target_achieved','strategy_review_required')
        four_reference=max(latest['L4_rho03']['test']['accuracy'],s.read(BASE/'once_tests/L4_rho03/result.json')['metrics']['accuracy'])
        for key in CASES:
            d,_=CASES[key];control=s.read(BASE/'once_tests'/f'L{d}_rho0'/'result.json')['metrics']['accuracy'];accuracy=latest[key]['test']['accuracy'];gap=100*(accuracy-control)
            achieved=gap>10 and (d!=4 or accuracy>.8) and (d!=6 or accuracy>four_reference)
            if achieved:states[key]='target_achieved' if d!=6 or four_final else 'held_pending_four_layer_final'
            elif states[key]=='strategy_review_required':continue
            elif states[key] in ('target_achieved','held_pending_four_layer_final'):states[key]='ready'
        keys=[k for k,v in states.items() if v=='ready']
        if not keys:
            if any(v=='held_pending_four_layer_final' for v in states.values()) and not four_final:continue
            break
        number+=1;block=a.out/'blocks'/f'block{number:04d}';block.mkdir();s.save(block/'metadata.json',dict(identity(),cases=keys,parents={k:str(parents[k]) for k in keys}),exclusive=True)
        s.save(a.out/'status.json',dict(state='genreg_running',states=states,block=number,latest=latest))
        jobs(block,'fit',keys,parents);audit_block(block,'fit');(block/'once_tests').mkdir();jobs(block,'test',keys,parents);audit_block(block,'test')
        for key in keys:
            f=s.read(block/key/'fit_result.json');t=s.read(block/'once_tests'/key/'result.json');latest[key]=dict(test=t['metrics'],model_digest=t['model_digest'],selected_epoch=f['selected_epoch'],actual_epochs=f['actual_epochs'],extra_epochs=f['extra_epochs'],train=f['metrics']['train'],val=f['metrics']['val'],fit_result_path=str(block/key/'fit_result.json'),test_result_path=str(block/'once_tests'/key/'result.json'))
            h=s.read(block/key/'history_extra.json');initial=s.read(BASE/key/'fit_result.json')['metrics'];allh=[dict(epoch=100,train=initial['train'],val=initial['val'])]+h
            significant=initial['val']['balanced_nll'];last_improved=100;best_accuracy=initial['val']['accuracy']
            for row in h:
                if row['val']['balanced_nll']<significant-.002 or row['val']['accuracy']>best_accuracy:significant=min(significant,row['val']['balanced_nll']);best_accuracy=max(best_accuracy,row['val']['accuracy']);last_improved=row['epoch']
            stale=f['actual_epochs']-last_improved;latest[key]['review_stale_epochs']=stale;parents[key]=block
            if stale>=20 or f['extra_epochs']>=config(CASES[key][0])['method_review_epochs']:
                states[key]='strategy_review_required';latest[key]['review_reason']='validation_plateau' if stale>=20 else 'registered_method_review_checkpoint'
        s.save(a.out/'comparison.json',dict(latest=latest,states=states,controls_frozen100=True,test_informed_development=True));s.save(block/'status.json',dict(state='science_and_tests_complete',active=[],pending=[],failures=[]))
    s.save(a.out/'comparison.json',dict(latest=latest,states=states,controls_frozen100=True,test_informed_development=True));s.save(a.out/'status.json',dict(state='all_genreg_stopped_targets_or_review',states=states,latest=latest,block=number))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--key');p.add_argument('--gpu');p.add_argument('--parent',type=Path);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        path=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not path.exists():s.save(path,dict(error=repr(e),automatic_retry=False,pid=os.getpid()),exclusive=True)
        raise
