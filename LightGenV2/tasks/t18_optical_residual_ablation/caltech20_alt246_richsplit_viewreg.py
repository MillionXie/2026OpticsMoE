"""Residual-only view coverage; inference graph, split and controls stay frozen."""
import argparse,copy,hashlib,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import PIL
from PIL import Image
import caltech20_alt246_richsplit as b
import caltech20_alt246_genreg as g
from caltech20_alt246_genreg_model import RegularizedMoE
HERE,ROOT,s,engine=b.HERE,b.ROOT,b.s,b.engine
PROFILE=HERE/'caltech20_alt246_richsplit_viewreg_profile.json'
FILES=('caltech20_alt246_richsplit_viewreg.py','caltech20_alt246_richsplit_viewreg_profile.json','check_caltech20_alt246_richsplit_viewreg.py','audit_caltech20_alt246_richsplit_viewreg.py','CALTECH20_ALT246_RICHSPLIT_VIEWREG_PROTOCOL_20261016.md')
BASE=HERE/'runs/simulation'/b.RUN
CPU=HERE/'runs/smoke/caltech20_richsplit_viewreg_cpu_20261016'

def versions():return dict(torch=str(torch.__version__),numpy=np.__version__,Pillow=PIL.__version__)

def config():
    p=s.read(PROFILE);cfg=b.config();cfg.pop('maximum_epochs');cfg.update(p['training']);cfg.update(scope=p['scope'],selection=p['selection'],rho=[.3],baseline_maximum_epochs=100,teacher=None);assert cfg['block_epochs']==20 and p['automatic_retry'] is False and cfg['cache_sha256']==p['cache_sha256'] and cfg['manifest_sha256']==p['manifest_sha256'];return cfg

def sources():
    src=b.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src

def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def receipt(root,record,folder):
    r=s.read(root/record);q=s.read(root/folder/'audit.json');assert r['actual_audit_returncode']==0 and q['passed'] and r['audit_sha256']==s.sha(root/folder/'audit.json');return r['audit_sha256']

def base_ready():
    assert s.read(BASE/'status.json')['state']=='richsplit_equal100_science_and_frozen_tests_complete';meta=s.read(BASE/'metadata.json');assert meta['sources']==b.sources();engine.verify_sources(meta['sources'],meta['git_commit'])
    proc=Path('/proc',str(meta['pid']),'cmdline');assert not proc.exists() or str(BASE).encode() not in proc.read_bytes()
    receipt(BASE,'audit_completed.json','science');receipt(BASE,'test_audit_completed.json','saved_test_audit')
    assert not any((BASE/n).exists() for n in ('failure.json','fit_audit_failure.json','test_audit_failure.json'))
    for key in b.CASES:
        f=s.read(BASE/key/'fit_result.json');assert f['epochs_completed']==100 and f['steps']==19700 and f['best_sha256']==s.sha(BASE/key/'best.pt') and f['last_sha256']==s.sha(BASE/key/'last.pt')
        for folder,phase in ((BASE/key,'fit'),(BASE/'once_tests'/key,'test')):
            ex=s.read(folder.with_name(key+'.'+phase+'_exit.json'));assert ex['returncode']==0 and ex['actual_exit']
    return meta

def parent_info(parent,depth):
    base_ready();assert depth in (2,4,6);key=f'L{depth}_rho03'
    if parent==BASE:
        f=s.read(BASE/key/'fit_result.json');t=BASE/'once_tests'/key;started=s.read(t/'started.json');test=s.read(t/'result.json')
        return dict(actual_epochs=100,steps=19700,last=str(BASE/key/'last.pt'),best=str(BASE/key/'best.pt'),train=str(BASE/key/'selected_train.npz'),val=str(BASE/key/'selected_val.npz'),selected_epoch=f['selected_epoch'],metrics=f['metrics'],history_path=str(BASE/key/'history.json'),test_registry={started['model_digest']:dict(result_path=str(t/'result.json'),predictions_path=str(t/'predictions.npz'),predictions_sha256=s.sha(t/'predictions.npz'),metrics=test['metrics'])})
    assert parent.parent==BASE.parent and parent.name.startswith(f'caltech20_rich20_L{depth}_viewreg_s17_to') and s.read(parent/'status.json')['state']=='viewreg_science_test_complete'
    meta=s.read(parent/'metadata.json');assert meta['sources']==sources() and meta['depth']==depth and meta['config']==config();engine.verify_sources(meta['sources'],meta['git_commit']);proc=Path('/proc',str(meta['pid']),'cmdline');assert not proc.exists() or str(parent).encode() not in proc.read_bytes()
    receipt(parent,'audit_completed.json','science');receipt(parent,'test_audit_completed.json','saved_test_audit');ex=s.read(parent/'campaign_exit.json');assert ex['returncode']==0 and ex['actual_exit'];assert s.read(parent/'status.json')['comparison_sha256']==s.sha(parent/'comparison.json');f=s.read(parent/'fit/fit_result.json');assert f['last_sha256']==s.sha(parent/'fit/last.pt') and f['history_sha256']==s.sha(parent/'fit/history.json') and f['best_sha256']==s.sha(f['best_path'])
    assert not any((parent/n).exists() for n in ('failure.json','fit_audit_failure.json','test_audit_failure.json'))
    return dict(actual_epochs=f['actual_epochs'],steps=f['steps'],last=str(parent/'fit/last.pt'),best=f['best_path'],train=f['selected_train_path'],val=f['selected_val_path'],selected_epoch=f['selected_epoch'],metrics=f['metrics'],history_path=str(parent/'fit/history.json'),test_registry=s.read(parent/'comparison.json')['test_registry'])

def expected_root(depth,parent):return HERE/'runs/simulation'/f'caltech20_rich20_L{depth}_viewreg_s17_to{parent_info(parent,depth)["actual_epochs"]+20}_20261016'

def cpu_ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['versions']==versions();engine.verify_sources(q['sources'],q['git_commit'])

def view_parameters(n,epoch,batch,cfg):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,cfg['view_seed_tag'],3]));params=[]
    for _ in range(n):
        area=float(rng.uniform(*cfg['crop_area']))*22500;ratio=math.exp(float(rng.uniform(math.log(cfg['crop_aspect'][0]),math.log(cfg['crop_aspect'][1]))));w=min(150,max(1,int(np.rint(math.sqrt(area*ratio)))));h=min(150,max(1,int(np.rint(math.sqrt(area/ratio)))))
        left=int(rng.integers(0,151-w));top=int(rng.integers(0,151-h));params.append([left,top,w,h,float(rng.uniform(*cfg['brightness'])),float(rng.uniform(*cfg['contrast'])),float(rng.uniform(*cfg['gamma']))])
    return np.asarray(params,dtype='<f8')

def transform(images,params):
    out=[]
    for image,p in zip(images,params):
        left,top,w,h=map(int,p[:4]);rgb=np.asarray(Image.fromarray(image).crop((left,top,left+w,top+h)).resize((150,150),Image.Resampling.BICUBIC)).astype(np.float64);mean=rgb.mean((0,1),keepdims=True)
        rgb=((rgb-mean)*p[5]+mean)*p[4];rgb=np.clip(rgb/255,0,1)**p[6];out.append(np.rint(rgb*255).astype(np.uint8))
    return np.stack(out)

def mixed(data,idx,theta,epoch,batch,cfg,device):
    params=view_parameters(len(idx),epoch,batch,cfg);pixels=transform(data[0][idx].permute(0,2,3,1).numpy(),params);local=(torch.from_numpy(pixels.transpose(0,3,1,2).copy()),data[1][idx],np.asarray(data[2])[idx])
    x,y,mask,record=g.mixed(local,np.arange(len(idx)),theta[idx],epoch,batch,cfg,device);record.update(view=s.digest(params),pixels=s.digest(pixels));return x,y,mask,record

def gate(depth,state,data,cfg,device,epoch):
    model=RegularizedMoE(depth).to(device);model.load_state_dict(state,strict=True);initial=s.digest(model.state_dict());weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device);order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);checks=[]
    for name,idx,batch in [('all20',np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]),0),('macro16',order[:16],0),('actual_tail',order[-cfg['tail_samples']:],cfg['steps_per_epoch']-1)]:
        x,y,mask,_=mixed(data,idx,theta,epoch,batch,cfg,device)
        for method in ('ordinary','view_and_dropout'):
            used=None if method=='ordinary' else mask;p,c,a=g.gradients(model,x,y,used,weights,cfg,1);q,d,z=g.gradients(model,x,y,used,weights,cfg,4);norms={n:float(np.linalg.norm(v.ravel())) for n,v in z.items()};assert all(np.isfinite(v).all() and norms[n]>0 for n,v in z.items())
            rel=max(float(np.linalg.norm((a[n]-v).ravel())/norms[n]) for n,v in z.items());cos=min(float(np.dot(a[n].ravel(),v.ravel())/(np.linalg.norm(a[n].ravel())*norms[n])) for n,v in z.items());err=max(float((p-q).abs().max()),float((c-d).abs().max()));assert err<=2e-6 and rel<=.02 and cos>=.999,(name,method,err,rel,cos)
            checks.append(dict(samples=name,n=len(idx),method=method,maxabs=err,gradient_relative_l2=rel,gradient_cosine=cos,gradient_norms=norms))
    assert s.digest(model.state_dict())==initial;del model;torch.cuda.empty_cache();return dict(passed=True,initial_digest=initial,checks=checks,optimizer_steps=0,test_arrays_read=0)

def fit(a):
    cpu_ready();cfg=config();parent=parent_info(a.parent,a.depth);assert not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in b.GPU.values();b.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'viewreg_restored_fit');a.out.mkdir()
    s.gpu_budget.ALLOWED.update(b.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    last=torch.load(parent['last'],map_location='cpu',weights_only=False);assert last['epoch']==parent['actual_epochs'] and last['steps']==parent['steps']==cfg['steps_per_epoch']*last['epoch'];data,val=b.load('train'),b.load('val');first=last['epoch']+1;end=last['epoch']+20
    proof=gate(a.depth,last['model'],data,cfg,'cuda',first);s.save(a.out/'numeric_gate.json',proof,exclusive=True)
    model=RegularizedMoE(a.depth).cuda();ema=copy.deepcopy(model).eval();model.load_state_dict(last['model'],strict=True);ema.load_state_dict(last['ema'],strict=True);assert s.digest(model.state_dict())==proof['initial_digest'];phase=[];electronic=[]
    for n,p in model.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(p)
    optimizer=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr']);optimizer.load_state_dict(last['optimizer']);moment=g.optimizer_moment_digest(optimizer.state_dict());assert moment==g.optimizer_moment_digest(last['optimizer'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3,threshold=.002,threshold_mode='abs',min_lr=cfg['minimum_lr'])
    if parent['actual_epochs']==100:
        assert np.allclose([q['lr'] for q in optimizer.param_groups],[cfg['lr'],cfg['lr']],atol=1e-12);scheduler.best=parent['metrics']['val']['balanced_nll']
    else:scheduler.load_state_dict(last['scheduler'])
    meta=dict(identity(),pid=os.getpid(),depth=a.depth,config=cfg,versions=versions(),gpu_uuid=a.gpu,policy=policy,parent=parent,parent_last_sha256=s.sha(parent['last']),parent_history_sha256=s.sha(parent['history_path']),parent_raw_digest=s.digest(last['model']),parent_ema_digest=s.digest(last['ema']),initial_optimizer_moment_digest=moment)
    s.save(a.out/'metadata.json',meta,exclusive=True);history=s.read(parent['history_path']);best=parent['metrics']['val']['balanced_nll'];selected=dict(epoch=parent['selected_epoch'],best_path=parent['best'],train_path=parent['train'],val_path=parent['val'],metrics=parent['metrics']);weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda();steps=parent['steps'];started=time.monotonic()
    for epoch in range(first,end+1):
        assert time.monotonic()-started<cfg['maximum_seconds'],'Incomplete review: retain evidence, no retry';begin=time.monotonic();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);record={k:[] for k in ('mix','flip','mask','view','pixels')};lr=[q['lr'] for q in optimizer.param_groups];model.train()
        for batch,off in enumerate(range(0,len(order),16)):
            idx=order[off:off+16];x,y,mask,hashes=mixed(data,idx,theta,epoch,batch,cfg,'cuda')
            for kind,h in hashes.items():record[kind].append(h)
            g.gradients(model,x,y,mask,weights,cfg,4);torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);previous={n:p.detach().cpu().clone() for n,p in ema.named_parameters()} if epoch==end and batch==cfg['steps_per_epoch']-1 else None;optimizer.step();steps+=1
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(cfg['ema_decay']).add_(mp,alpha=1-cfg['ema_decay'])
        tm,tr=b.evaluate(ema,data);vm,vr=b.evaluate(ema,val);entry=dict(epoch=epoch,train=tm,val=vm,lr=lr,adam_steps=steps,order_sha256=s.digest(order),affine_sha256=s.digest(theta),**{kind+'_sha256':hashlib.sha256(''.join(h).encode()).hexdigest() for kind,h in record.items()},raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin);history.append(entry)
        if vm['balanced_nll']<best:
            best=vm['balanced_nll'];engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,sources=meta['sources'],config=cfg,contract=ema.contract(),metrics=dict(train=tm,val=vm)));b.save_np(a.out/'selected_train.npz',tr);b.save_np(a.out/'selected_val.npz',vr);selected=dict(epoch=epoch,best_path=str(a.out/'best.pt'),train_path=str(a.out/'selected_train.npz'),val_path=str(a.out/'selected_val.npz'),metrics=dict(train=tm,val=vm))
        scheduler.step(vm['balanced_nll']);engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,steps=steps,sources=meta['sources'],config=cfg));s.save(a.out/'history.json',history);s.save(a.out/'status.json',dict(state='training',epoch=epoch,train=tm,val=vm));print(dict(depth=a.depth,epoch=epoch,train=tm['accuracy'],val=vm['accuracy'],NLL=vm['balanced_nll']),flush=True)
    assert steps==cfg['steps_per_epoch']*end;identity();s.save(a.out/'fit_result.json',dict(passed=True,actual_epochs=end,additional_epochs=end-100,block_epochs=20,selected_epoch=selected['epoch'],best_path=selected['best_path'],best_sha256=s.sha(selected['best_path']),selected_train_path=selected['train_path'],selected_val_path=selected['val_path'],metrics=selected['metrics'],steps=steps,fit_seconds=time.monotonic()-started,history_sha256=s.sha(a.out/'history.json'),last_sha256=s.sha(a.out/'last.pt'),stop_reason='registered20_epoch_method_review'),exclusive=True)

def test(a):
    root=a.out.parent;cfg=config();parent=parent_info(a.parent,a.depth);receipt(root,'audit_completed.json','science');assert not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'selected_digest_once_test');a.out.mkdir();f=s.read(root/'fit/fit_result.json');checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);digest=s.digest(checkpoint['model'])
    s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),selected_epoch=f['selected_epoch'],best_sha256=f['best_sha256'],model_digest=digest),exclusive=True)
    if digest in parent['test_registry']:
        known=parent['test_registry'][digest];assert known['predictions_sha256']==s.sha(known['predictions_path']);result=dict(metrics=known['metrics'],predictions_path=known['predictions_path'],predictions_sha256=known['predictions_sha256'],reused=True,new_test_evaluations=0)
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in b.GPU.values();b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);claim(root/('digest_'+digest+'.claim.json'),'new_parameter_digest_once_test');model=RegularizedMoE(a.depth).cuda();model.load_state_dict(checkpoint['model'],strict=True);metrics,arrays=b.evaluate(model,b.load('test'));b.save_np(a.out/'predictions.npz',arrays);result=dict(metrics=metrics,predictions_path=str(a.out/'predictions.npz'),predictions_sha256=s.sha(a.out/'predictions.npz'),reused=False,new_test_evaluations=1)
    s.save(a.out/'result.json',dict(result,selected_epoch=f['selected_epoch'],best_sha256=f['best_sha256'],model_digest=digest,test_used_for_epoch_selection=False,test_informed_development=True),exclusive=True)

def run_child(a,phase,gpu):
    out=a.out/phase
    with (a.out/(phase+'.log')).open('x') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--out',str(out),'--depth',str(a.depth),'--parent',str(a.parent),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);rc=proc.wait()
    s.save(a.out/(phase+'_exit.json'),dict(pid=proc.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Producer failed: retain evidence, no retry'

def audit_child(a,phase):
    claim(a.out/(phase+'_audit_claim.json'),'own_saved_'+phase+'_math');out=a.out/('science' if phase=='fit' else 'saved_test_audit')
    with (a.out/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_richsplit_viewreg.py'),'--base',str(a.out),'--out',str(out),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Saved audit failed: preserve/no retry';s.save(a.out/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/'audit.json')),exclusive=True)

def campaign(a):
    cpu_ready();parent=parent_info(a.parent,a.depth);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==expected_root(a.depth,a.parent) and not a.out.exists();b.slots();claim(a.out.with_name(a.out.name+'.campaign_claim.json'),'residual_view_coverage_review');a.out.mkdir();s.save(a.out/'metadata.json',dict(identity(),pid=os.getpid(),depth=a.depth,config=config(),parent_root=str(a.parent),parent=parent),exclusive=True)
    s.gpu_budget.ALLOWED.update(b.GPU.values());gpu=None
    for card in (4,2,5,0):
        if b.slots()['counts'][b.GPU[card]]:continue
        try:s.gpu_budget.preflight(b.GPU[card],shared=True,memory_gib=2);gpu=b.GPU[card];break
        except AssertionError:continue
    assert gpu is not None,'No safe slot';s.save(a.out/'status.json',dict(state='viewreg_fit_running'));run_child(a,'fit',gpu);audit_child(a,'fit');s.save(a.out/'status.json',dict(state='viewreg_test_running'));run_child(a,'test',gpu);audit_child(a,'test')
    f=s.read(a.out/'fit/fit_result.json');t=s.read(a.out/'test/result.json');registry=parent['test_registry'];registry[t['model_digest']]=dict(result_path=str(a.out/'test/result.json'),predictions_path=t['predictions_path'],predictions_sha256=t['predictions_sha256'],metrics=t['metrics']);control=s.read(BASE/'once_tests'/f'L{a.depth}_rho0/result.json')['metrics'];four=s.read(BASE/'once_tests/L4_rho03/result.json')['metrics']
    s.save(a.out/'comparison.json',dict(depth=a.depth,actual_epochs=f['actual_epochs'],selected_epoch=f['selected_epoch'],train=f['metrics']['train'],val=f['metrics']['val'],test=t['metrics'],control100=control,test_gap_pp=100*(t['metrics']['accuracy']-control['accuracy']),six_above_baseline_four=t['metrics']['accuracy']>four['accuracy'] if a.depth==6 else None,test_registry=registry,test_informed_development=True,controls_frozen=True),exclusive=True);s.save(a.out/'status.json',dict(state='viewreg_science_test_complete',comparison_sha256=s.sha(a.out/'comparison.json')))

def supervise(a):
    cpu_ready();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==expected_root(a.depth,a.parent) and not a.out.exists();claim(a.out.with_name(a.out.name+'.launch_claim.json'),'finite_foreground_experiment_supervision')
    with a.out.with_name(a.out.name+'.campaign.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(a.out),'--depth',str(a.depth),'--parent',str(a.parent)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        s.save(a.out.with_name(a.out.name+'.started.json'),dict(identity(),supervisor_pid=os.getpid(),campaign_pid=proc.pid,depth=a.depth,parent=str(a.parent)),exclusive=True);rc=proc.wait()
    record=a.out/'campaign_exit.json' if a.out.exists() else a.out.with_name(a.out.name+'.campaign_exit.json');s.save(record,dict(pid=proc.pid,supervisor_pid=os.getpid(),returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Campaign failure retained; no retry'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--depth',type=int,choices=[2,4,6],required=True);p.add_argument('--parent',type=Path,default=BASE);p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
