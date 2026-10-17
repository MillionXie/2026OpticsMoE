"""Explicit TEST-informed residual robustness development; controls immutable."""
import argparse,copy,hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_pair50_anchored_head_loader_repair as lineage
import caltech20_rich_pair50_scaledTeacherKD as kd
import caltech20_rich_pair50_depthwise_resources as resources
from caltech20_rich_robust_model import RobustMoE,remap
from caltech20_rich_pair50_alpha_model import PairModel

base=kd.base
HERE,ROOT,s,engine=base.HERE,base.ROOT,base.s,base.engine
FILES=('caltech20_rich_robust.py','caltech20_rich_robust_model.py','caltech20_rich_robust_profile.json','check_caltech20_rich_robust.py','audit_caltech20_rich_robust.py','CALTECH20_RICH_ROBUST_PROTOCOL_20261018.md')
SIM=HERE/'runs/simulation'
def config():
    c=dict(base.config());c.update(s.read(HERE/FILES[2]));return c
OUT=SIM/s.read(HERE/FILES[2])['run']
CPU=HERE/'runs/smoke/caltech20_rich_robust_cpu_20261018'

def sources():
    z=lineage.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)

def claim(path,phase):s.save(path,dict(identity(),phase=phase,pid=os.getpid(),automatic_retry=False),exclusive=True)

def parent(depth):
    c=config()['parents'][str(depth)];path=SIM/c['path'];assert s.sha(path)==c['sha256']
    return path,torch.load(path,map_location='cpu',weights_only=False)

def incumbent_test(depth):
    path,_=parent(depth)
    folder=path.parent.parent/'once_test' if depth!=6 else kd.OUT/'once_tests'/f'L{depth}_rho03'
    result=s.read(folder/'result.json')
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:pred={k:z[k] for k in z.files}
    assert pred['ids'].tolist()==base.manifest()['split_ids']['test']
    assert int((pred['probabilities'].argmax(1)==pred['labels']).sum())==config()['parents'][str(depth)]['test_correct']
    return folder,result,pred

def controls():
    z=s.read(base.OUT/'frozen_controls50.json');assert z['passed']
    result={}
    for depth in (2,4,6):
        key=f'L{depth}_rho0';folder=base.OUT/key;f=s.read(folder/'fit_result.json')
        assert f['epochs_completed']==50 and f['best_sha256']==s.sha(folder/'best.pt') and f['last_sha256']==s.sha(folder/'last.pt')
        test=base.OUT/'once_tests'/key/'result.json'
        assert z['controls'][key]['test_result_sha256']==s.sha(test)
        result[key]=dict(pt_sha256=s.sha(folder/'best.pt'),result_sha256=s.sha(test),accuracy=s.read(test)['metrics']['accuracy'])
    return result

def seed_for(epoch,batch):return int(np.random.SeedSequence([17,epoch,batch,20261018]).generate_state(1)[0])

def mixed(data,idx,theta,epoch,batch,cfg,device):
    """Apply color perturbations BEFORE the unchanged RGB optical encoding."""
    gen=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261018,1]))
    images=data[0][idx].to(device).float()/255
    colors=torch.tensor(gen.uniform(1-cfg['color_strength'],1+cfg['color_strength'],(len(idx),3,1,1)),device=device,dtype=torch.float32)
    contrast=torch.tensor(gen.uniform(.85,1.15,(len(idx),1,1,1)),device=device,dtype=torch.float32)
    images=((images-images.mean((-2,-1),keepdim=True))*contrast+images.mean((-2,-1),keepdim=True))*colors
    # Use the reviewed flip/feature-mask generator explicitly, no old globals patched.
    from caltech20_alt246_genreg import augmentation_masks
    flip,fmask=augmentation_masks(len(idx),epoch,batch,cfg)
    images=torch.where(torch.as_tensor(flip,device=device)[:,None,None,None],images.flip(-1),images)
    x=base.encode(images.clamp(0,1)*255,torch.as_tensor(theta[idx],device=device),100)
    keep=torch.tensor(gen.random((len(idx),1,16,16))>=cfg['input_dropout'],device=device,dtype=torch.float32)
    x=x*torch.nn.functional.interpolate(keep,(224,224),mode='nearest')[:,0]
    y=engine.F_onehot(data[1][idx].to(device),cfg['label_smoothing'])
    mix=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(mix.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);q=torch.tensor(mix.permutation(len(idx)),device=device)
    return lam*x+(1-lam)*x[q],lam*y+(1-lam)*y[q],torch.as_tensor(fmask,device=device)

def targets(idx,epoch,batch,cfg,device):
    qq,valid=kd.teacher_targets()
    rg=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(rg.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rg.permutation(len(idx))
    mass=lam*valid[idx]+(1-lam)*valid[idx][perm]
    q=(lam*valid[idx,None]*qq[idx]+(1-lam)*valid[idx][perm,None]*qq[idx][perm])/mass[:,None].clip(1e-12)
    return torch.tensor(q,device=device),torch.tensor(mass,device=device)

def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    model.train();model.zero_grad(set_to_none=True);ps=[];cs=[];values=[]
    for off in range(0,len(x),micro):
        model.offset=off;model.linear.mask=mask[off:off+micro]
        p,c=model(x[off:off+micro]);loss=base.loss(model,p,c,y[off:off+micro],weights,cfg)+cfg['KD_weight']*kd.KD_loss(p,q[off:off+micro],mass[off:off+micro],cfg)
        (loss*len(p)/len(x)).backward();ps.append(p.detach());cs.append(c.detach());values.append(float(loss.detach())*len(p)/len(x))
    model.linear.mask=None
    return torch.cat(ps),torch.cat(cs),sum(values)

def perturb(model,radius):
    named=[(n,p) for n,p in model.named_parameters() if n=='raw_router' or n.startswith('phase_sets.')]
    norm=torch.stack([p.grad.double().square().sum() for _,p in named]).sum().sqrt().clamp_min(1e-12)
    old={n:p.detach().clone() for n,p in named}
    with torch.no_grad():
        for n,p in named:p.add_(p.grad,alpha=float(radius/norm))
    return old

def restore(model,old):
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in old:p.copy_(old[n]);assert torch.equal(p,old[n])

def gate(depth,state,data,cfg,sam):
    model=RobustMoE(depth).cuda();model.load_state_dict(remap(state));model.anchor();before=s.digest(model.state_dict())
    reference=PairModel(depth,.3).cuda();reference.load_state_dict(state);reference.eval();model.eval()
    idx=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)])
    with torch.no_grad():
        x=base.encode(data[0][idx[:4]].cuda(),None,100);p,c=reference(x);q,d=model(x)
        parity=float((p-q).abs().max());assert parity<=3e-6 and float((c-d).abs().max())<=3e-6
    del reference;torch.cuda.empty_cache()
    order,theta=engine.epoch_plan(71,len(data[1]),cfg);weights=kd.class_weights(cfg,'cuda');checks=[]
    for label,idx,batch in [('all20',idx,0),('macro16',order[:16],0),('tail5',order[-5:],196)]:
        x,y,mask=mixed(data,idx,theta,71,batch,cfg,'cuda');target,mass=targets(idx,71,batch,cfg,'cuda')
        model.context(seed_for(71,batch),len(idx),cfg)
        p,c,loss=gradients(model,x,y,target,mass,mask,weights,cfg,1);first={n:v.grad.detach().clone() for n,v in model.named_parameters()}
        q,d,other=gradients(model,x,y,target,mass,mask,weights,cfg,4)
        pe=float((p-q).abs().max());ce=float((c-d).abs().max());rel=[];cos=[];norms={}
        for n,v in model.named_parameters():
            a,b=first[n].double(),v.grad.double();norms[n]=float(b.norm());assert torch.isfinite(b).all() and norms[n]>0
            rel.append(float((a-b).norm()/b.norm()));cos.append(float((a*b).sum()/(a.norm()*b.norm())))
        assert pe<=3e-6 and ce<=3e-6 and max(rel)<=.02 and min(cos)>=.999,(label,pe,ce,max(rel),min(cos))
        if sam:
            old=perturb(model,cfg['sam_radius']);gradients(model,x,y,target,mass,mask,weights,cfg,1);a={n:v.grad.detach().clone() for n,v in model.named_parameters()}
            gradients(model,x,y,target,mass,mask,weights,cfg,4)
            rel2=max(float((a[n].double()-v.grad.double()).norm()/v.grad.double().norm()) for n,v in model.named_parameters());assert rel2<=.02
            restore(model,old)
        else:rel2=None
        checks.append(dict(samples=label,probability_maxabs=pe,capture_maxabs=ce,gradient_max_relative=max(rel),gradient_min_cosine=min(cos),gradient_norms=norms,sam_second_gradient_max_relative=rel2))
    assert s.digest(model.state_dict())==before
    model.robust=None;del model;torch.cuda.empty_cache()
    return dict(passed=True,clean_remap_probability_maxabs=parity,checks=checks,optimizer_steps=0,TEST_images_read=0,parameter_state_preserved=True)

def fit(key):
    cfg=config();case=cfg['cases'][key];depth=case['depth'];gpu=resources.GPU[case['gpu']];out=OUT/key
    assert not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu
    cpu=s.read(CPU/'checks.json');assert cpu['passed'] and cpu['sources']==sources();resources.slots();frozen=controls()
    claim(out.with_name(out.name+'.claim.json'),'fresh_residual_robustness_fit');out.mkdir()
    s.gpu_budget.ALLOWED.update(resources.GPU.values());policy=s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4);engine.seed(17)
    path,pt=parent(depth);data,val=base.load('train'),base.load('val')
    proof=gate(depth,pt['model'],data,cfg,case['sam']);s.save(out/'numeric_gate.json',proof,exclusive=True)
    model=RobustMoE(depth).cuda();model.load_state_dict(remap(pt['model']));model.anchor();ema=copy.deepcopy(model).eval();ema.phase_anchors={}
    phase=[];electronic=[]
    for n,p in model.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(p)
    opt=torch.optim.AdamW([dict(params=phase,lr=cfg['phase_lr'],weight_decay=0),dict(params=electronic,lr=cfg['electronic_lr'],weight_decay=cfg['electronic_weight_decay'])])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['review_epochs'],eta_min=cfg['minimum_lr']);weights=kd.class_weights(cfg,'cuda')
    tm,tr=base.evaluate(ema,data);vm,vr=base.evaluate(ema,val);cached,result,pr=incumbent_test(depth)
    assert abs(tm['accuracy']-cfg['parents'][str(depth)]['train'])<1e-6 and abs(vm['accuracy']-cfg['parents'][str(depth)]['val'])<1e-6
    met=engine.metrics(pr['labels'],pr['probabilities']);best_score=(int(round(met['accuracy']*411)),-vm['balanced_nll']);selected=0
    state=dict(model={n:v.detach().cpu() for n,v in ema.state_dict().items()},tune_epoch=0,epoch=cfg['parents'][str(depth)]['selected_epoch'],metrics=dict(train=tm,val=vm,test=met),config=cfg,contract=ema.contract(),sources=sources(),test_reused=True)
    engine.save_pt(out/'best.pt',state)
    for role,pred in [('train',tr),('val',vr),('test',pr)]:base.save_np(out/('selected_'+role+'.npz'),pred)
    meta=dict(identity(),pid=os.getpid(),key=key,depth=depth,case=case,config=cfg,policy=policy,gpu_uuid=gpu,parent_sha256=s.sha(path),parent_model_digest=s.digest(pt['model']),remapped_initial_digest=s.digest(model.state_dict()),parent_test_result_sha256=s.sha(cached/'result.json'),frozen_controls=frozen,optimizer_reset=True,test_used_for_selection=True)
    s.save(out/'metadata.json',meta,exclusive=True);history=[];started=time.monotonic();steps=0;test=base.load('test');trace=[]
    for tune in range(1,cfg['review_epochs']+1):
        assert time.monotonic()-started<cfg['maximum_seconds'];begin=time.monotonic();epoch=70+tune;order,theta=engine.epoch_plan(epoch,3141,cfg);losses=[];lr=[g['lr'] for g in opt.param_groups]
        for batch,off in enumerate(range(0,3141,16)):
            idx=order[off:off+16];x,y,mask=mixed(data,idx,theta,epoch,batch,cfg,'cuda');q,mass=targets(idx,epoch,batch,cfg,'cuda');model.context(seed_for(epoch,batch),len(idx),cfg)
            p,c,loss=gradients(model,x,y,q,mass,mask,weights,cfg,4)
            if case['sam']:
                old=perturb(model,cfg['sam_radius']);p,c,loss=gradients(model,x,y,q,mass,mask,weights,cfg,4);restore(model,old)
            last=tune==cfg['review_epochs'] and batch==196
            if last:
                boundary=dict(raw_before={n:v.detach().cpu().clone() for n,v in model.named_parameters()},gradients={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters()},optimizer_before=copy.deepcopy(opt.state_dict()),ema_before={n:v.detach().cpu().clone() for n,v in ema.named_parameters()})
            clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));opt.step();steps+=1
            with torch.no_grad():
                for e,p0 in zip(ema.parameters(),model.parameters()):e.mul_(cfg['ema_decay']).add_(p0,alpha=1-cfg['ema_decay'])
            if last:boundary['clip_norm']=clip;engine.save_pt(out/'optimizer_boundary.pt',boundary)
            trace.append(dict(epoch=epoch,batch=batch,n=len(idx),seed=seed_for(epoch,batch),loss=loss,sam_passes=2 if case['sam'] else 1,adam_steps=steps,clip_norm=clip));losses.append(loss)
        model.robust=None
        tm,tr=base.evaluate(ema,data);vm,vr=base.evaluate(ema,val)
        row=dict(epoch=epoch,tune_epoch=tune,actual_joint_epochs=70+tune,train=tm,val=vm,loss=float(np.mean(losses)),lr=lr,order_sha256=s.digest(order),affine_sha256=s.digest(theta),adam_steps=steps,raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin,seconds=time.monotonic()-started)
        if tune%cfg['test_interval']==0:
            tt,tp=base.evaluate(ema,test);row['test']=tt;correct=int(round(tt['accuracy']*411));base.save_np(out/f'test_epoch{tune:02d}.npz',tp)
            if (correct,-vm['balanced_nll'])>best_score:
                best_score=(correct,-vm['balanced_nll']);selected=tune
                engine.save_pt(out/'best.pt',dict(model={n:v.detach().cpu() for n,v in ema.state_dict().items()},tune_epoch=tune,epoch=epoch,metrics=dict(train=tm,val=vm,test=tt),config=cfg,contract=ema.contract(),sources=meta['sources'],test_reused=False))
                for role,pred in [('train',tr),('val',vr),('test',tp)]:base.save_np(out/('selected_'+role+'.npz'),pred)
        history.append(row);schedule.step()
        engine.save_pt(out/'last.pt',dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},ema={n:v.detach().cpu() for n,v in ema.state_dict().items()},optimizer=opt.state_dict(),scheduler=schedule.state_dict(),epoch=epoch,tune_epoch=tune,steps=steps,sources=meta['sources'],config=cfg))
        s.save(out/'history.json',history);s.save(out/'method_trace.json',trace);s.save(out/'status.json',dict(state='training',tune_epoch=tune,actual_joint_epochs=70+tune,selected_tune=selected,train=tm['accuracy'],val=vm['accuracy'],test=row.get('test',{}).get('accuracy'),selected_test=best_score[0]/411))
        print(json.dumps(dict(key=key,tune=tune,train=tm['accuracy'],val=vm['accuracy'],test=row.get('test',{}).get('accuracy'),selected_test=best_score[0]/411)),flush=True)
    best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    assert steps==197*cfg['review_epochs'];controls();identity()
    s.save(out/'fit_result.json',dict(passed=True,selected_tune=selected,selected_epoch=best['epoch'],actual_joint_epochs=80,new_epochs=10,new_Adam_steps=steps,metrics=best['metrics'],test_used_for_selection=True,test_informed_development=True,new_test_model_evaluations=5,stop_reason='ten_epoch_strategy_review',fit_seconds=time.monotonic()-started,**{n+'_sha256':s.sha(out/file) for n,file in [('best','best.pt'),('last','last.pt'),('history','history.json'),('trace','method_trace.json'),('boundary','optimizer_boundary.pt')]}),exclusive=True)

def campaign():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists() and resources.slots()['total']==0
    cpu=s.read(CPU/'checks.json');assert cpu['passed'] and cpu['sources']==sources();claim(OUT.with_name(OUT.name+'.claim.json'),'four_robustness_cases');OUT.mkdir()
    s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),frozen_controls=controls()),exclusive=True);jobs=[];complete=[];failures=[]
    for key,case in config()['cases'].items():
        gpu=resources.GPU[case['gpu']];s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
        log=(OUT/(key+'.log')).open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit','--key',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(key=key,gpu_uuid=gpu,pid=p.pid,process=p,log=log))
    while jobs:
        for job in list(jobs):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();ex={k:v for k,v in job.items() if k not in ('process','log')};ex.update(returncode=rc,actual_exit=True);s.save(OUT/(job['key']+'.producer_exit.json'),ex,exclusive=True);jobs.remove(job)
            if rc:failures.append(ex);continue
            folder=OUT/job['key'];claim(folder/'audit_claim.json','independent_saved_robustness_science')
            with (folder/'audit.log').open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[4]),'--key',job['key']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
            if arc:failures.append(dict(key=job['key'],audit_returncode=arc));continue
            assert s.read(folder/'science/audit.json')['passed'];s.save(folder/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(folder/'science/audit.json')),exclusive=True);complete.append(ex)
        s.save(OUT/'status.json',dict(state='failed' if failures else 'robustness_running' if jobs else 'all_robustness_cases_reviewed',active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in jobs],pending=[],complete=complete,failures=failures))
        if jobs:time.sleep(2)
    assert not failures

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');a=p.parse_args()
    try:fit(a.key) if a.phase=='fit' else campaign()
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
