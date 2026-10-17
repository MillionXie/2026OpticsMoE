"""Alpha40 electronic warm adaptation versus fresh L4, fixed data and controls."""
import argparse,copy,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import caltech20_rich_alpha40_diagnosis as diagnosis
import caltech20_rich_robust as old
from caltech20_rich_alpha40_model import Alpha40MoE,remap
s,base,engine,resources=diagnosis.s,diagnosis.base,diagnosis.engine,diagnosis.resources
HERE,ROOT=base.HERE,base.ROOT
FILES=('caltech20_rich_alpha40_rebalance.py','caltech20_rich_alpha40_rebalance_profile.json','check_caltech20_rich_alpha40_rebalance.py','audit_caltech20_rich_alpha40_rebalance.py','CALTECH20_RICH_ALPHA40_REBALANCE_PROTOCOL_20261018.md')
def config():
    c=dict(base.config());c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_alpha40_rebalance_cpu_20261018'
def sources():
    z=diagnosis.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();z=sources();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def groups(model):
    g=[[],[],[]]
    for n,p in model.named_parameters():g[0 if n=='raw_router' or n.startswith('phase_sets.') else 2 if n.startswith('fusion_logits.') else 1].append((n,p))
    return g
def initialize(depth,fresh):
    engine.seed(17);model=Alpha40MoE(depth)
    if not fresh:
        q,pt=diagnosis.parent(depth);model.load_state_dict(remap(pt['model'],q['alpha_low']))
    model.set_alpha(config()['alpha_restart']);model.anchor();return model
def rates(case,epoch):
    c=config();starts=[case['phase_lr'],c['electronic_lr'],c['alpha_lr']];floors=[c['minimum_lr'],c['minimum_lr'],c['minimum_alpha_lr']]
    return [b+(a-b)*(1+math.cos(math.pi*(epoch-1)/case['epochs']))/2 for a,b in zip(starts,floors)]
def seed_for(epoch,batch):return int(np.random.SeedSequence([17,epoch,batch,20261018,40]).generate_state(1)[0])
def mixed(data,idx,theta,epoch,batch,cfg,device):
    """TRAIN-only random resized crop, flip, color, existing affine and MixUp."""
    rng=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261018,40,1]));n=len(idx)
    image=data[0][idx].to(device).float()/255
    area=rng.uniform(cfg['crop_area_min'],1,n);ratio=rng.uniform(.9,1.1,n);width=np.minimum(1,np.sqrt(area*ratio));height=np.minimum(1,np.sqrt(area/ratio))
    crop=np.zeros((n,2,3),dtype=np.float32);crop[:,0,0]=width;crop[:,1,1]=height;crop[:,0,2]=rng.uniform(-1,1,n)*(1-width);crop[:,1,2]=rng.uniform(-1,1,n)*(1-height)
    grid=F.affine_grid(torch.tensor(crop,device=device),image.shape,align_corners=False);image=F.grid_sample(image,grid,align_corners=False,padding_mode='border')
    image=torch.where(torch.tensor(rng.random(n)<.5,device=device)[:,None,None,None],image.flip(-1),image)
    gain=torch.tensor(rng.uniform(.8,1.2,(n,3,1,1)),device=device,dtype=torch.float32);contrast=torch.tensor(rng.uniform(.8,1.2,(n,1,1,1)),device=device,dtype=torch.float32)
    mean=image.mean((-2,-1),keepdim=True);image=(mean+(image-mean)*contrast)*gain
    gray=image.mean(1,keepdim=True).expand_as(image);image=torch.where(torch.tensor(rng.random(n)<.1,device=device)[:,None,None,None],gray,image)
    x=base.encode(image.clamp(0,1)*255,torch.tensor(theta[idx],device=device),100)
    keep=torch.tensor(rng.random((n,1,16,16))>=cfg['input_dropout'],device=device,dtype=torch.float32);x*=F.interpolate(keep,(224,224),mode='nearest')[:,0]
    y=engine.F_onehot(data[1][idx].to(device),cfg['label_smoothing'])
    mix=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(mix.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=torch.tensor(mix.permutation(n),device=device)
    mask=torch.tensor(rng.random((n,256))>=cfg['feature_dropout'],device=device,dtype=torch.float32)/(1-cfg['feature_dropout'])
    return lam*x+(1-lam)*x[perm],lam*y+(1-lam)*y[perm],mask,s.digest(crop)
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    model.train();model.zero_grad(set_to_none=True);ps=[];cs=[];losses=[]
    for off in range(0,len(x),micro):
        model.offset=off;model.linear.mask=mask[off:off+micro];p,c=model(x[off:off+micro])
        anchor=torch.stack([(a.sigmoid()*(.95-.4001)+.4001-cfg['alpha_restart']).square() for a in model.fusion_logits]).mean()
        loss=base.loss(model,p,c,y[off:off+micro],weights,cfg)+cfg['KD_weight']*old.kd.KD_loss(p,q[off:off+micro],mass[off:off+micro],cfg)+cfg['alpha_anchor_weight']*anchor
        (loss*len(p)/len(x)).backward();ps.append(p.detach());cs.append(c.detach());losses.append(float(loss.detach())*len(p)/len(x))
    model.linear.mask=None;return torch.cat(ps),torch.cat(cs),sum(losses)
def load_predictions(path):
    with np.load(path,allow_pickle=False) as z:return {n:z[n] for n in z.files}
def incumbent(depth):
    q,pt=diagnosis.parent(depth);path=Path(q['path'])
    if depth==2:
        cached,_,test=old.incumbent_test(depth);pred={role:load_predictions(path.parent/('selected_'+role+'.npz')) for role in ('train','val')};pred['test']=test
    else:pred={role:load_predictions(path.parent/('candidate_'+role+'.npz')) for role in ('train','val','test')}
    return q,pt,pred
def ready():
    p=s.read(CPU/'checks.json');d=s.read(diagnosis.OUT/'diagnosis.json');assert p['passed'] and p['sources']==sources() and d['passed'] and d['sources']==diagnosis.sources()
    assert d['TEST_images_read']==d['optimizer_steps']==0
    identity();old.controls();return p
def gate(depth,fresh,data,case):
    cfg=config();model=initialize(depth,fresh).cuda();before=s.digest(model.state_dict());idx=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)])
    order,theta=engine.epoch_plan(1,3141,cfg);weights=old.kd.class_weights(cfg,'cuda');records=[]
    for label,ix,batch in [('all20',idx,0),('macro16',order[:16],0),('tail5',order[-5:],196)]:
        x,y,mask,crop=mixed(data,ix,theta,1,batch,cfg,'cuda');q,mass=old.targets(ix,1,batch,cfg,'cuda');model.context(seed_for(1,batch),len(ix),cfg)
        p,c,_=gradients(model,x,y,q,mass,mask,weights,cfg,1);first={n:v.grad.detach().clone() for n,v in model.named_parameters()}
        a,b,_=gradients(model,x,y,q,mass,mask,weights,cfg,4);pe=float((p-a).abs().max());ce=float((c-b).abs().max());rel=[];cos=[];norms={}
        for n,v in model.named_parameters():
            aa,bb=first[n].double(),v.grad.double();norms[n]=float(bb.norm());assert torch.isfinite(bb).all() and norms[n]>0
            rel.append(float((aa-bb).norm()/bb.norm()));cos.append(float((aa*bb).sum()/(aa.norm()*bb.norm())))
        assert pe<=3e-6 and ce<=3e-6 and max(rel)<=.02 and min(cos)>=.999,(label,pe,ce,max(rel),min(cos))
        records.append(dict(samples=label,probability_maxabs=pe,capture_maxabs=ce,gradient_max_relative=max(rel),gradient_min_cosine=min(cos),gradient_norms=norms,crop_sha256=crop))
    assert before==s.digest(model.state_dict());del model;torch.cuda.empty_cache()
    return dict(passed=True,checks=records,initial_digest=before,optimizer_steps=0,TEST_images_read=0)
def fit(key,gpu):
    cfg=config();case=cfg['cases'][key];depth=case['depth'];out=OUT/key;assert not out.exists() and gpu in resources.GPU.values() and os.environ['CUDA_VISIBLE_DEVICES']==gpu;ready();resources.slots();claim(out.with_name(out.name+'.claim.json'),'alpha40_residual_fit');out.mkdir()
    s.gpu_budget.ALLOWED.update(resources.GPU.values());policy=s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    data,val=base.load('train'),base.load('val');proof=gate(depth,case['fresh'],data,case);s.save(out/'numeric_gate.json',proof,exclusive=True)
    model=initialize(depth,case['fresh']).cuda();ema=copy.deepcopy(model).eval();ema.phase_anchors={};named=groups(model)
    opt=torch.optim.AdamW([dict(params=[p for n,p in g],lr=lr,weight_decay=wd) for g,lr,wd in zip(named,rates(case,1),[0,cfg['electronic_weight_decay'],0])])
    parent,pt,pred=incumbent(depth);reference=Alpha40MoE(depth);reference.load_state_dict(remap(pt['model'],parent['alpha_low']))
    metrics={role:engine.metrics(p['labels'],p['probabilities']) for role,p in pred.items()};best=(int(round(metrics['test']['accuracy']*411)),-metrics['val']['balanced_nll']);selected=0
    engine.save_pt(out/'best.pt',dict(model=reference.state_dict(),metrics=metrics,epoch=parent['selected_optical_epoch'],tune_epoch=0,contract=reference.contract(),sources=sources(),config=cfg,test_reused=True))
    for role,p in pred.items():base.save_np(out/('selected_'+role+'.npz'),p)
    del reference
    frozen_init=s.digest({n:p for n,p in model.state_dict().items() if n=='raw_router' or n.startswith('phase_sets.') or n.startswith('fusion_logits.')})
    meta=dict(identity(),pid=os.getpid(),key=key,depth=depth,case=case,config=cfg,gpu_uuid=gpu,policy=policy,parent=parent,init_digest=s.digest(model.state_dict()),frozen_phase_alpha_initial_digest=frozen_init,frozen_controls=old.controls(),optimizer_reset=True,alpha_restart=cfg['alpha_restart'],test_used_for_selection=True)
    s.save(out/'metadata.json',meta,exclusive=True);weights=old.kd.class_weights(cfg,'cuda');history=[];trace=[];steps=0;started=time.monotonic();test=base.load('test')
    # Intentional coefficient restart is measured on TRAIN/VAL only, never E0 TEST.
    tm,_=base.evaluate(ema,data);vm,_=base.evaluate(ema,val);s.save(out/'restart_train_val.json',dict(train=tm,val=vm,alpha=ema.alphas(),new_TEST_forwards=0),exclusive=True)
    for tune in range(1,case['epochs']+1):
        assert time.monotonic()-started<cfg['maximum_seconds'];begin=time.monotonic();warm=tune<=case['electronic_warm_epochs']
        for n,p in model.named_parameters():p.requires_grad_(not warm or not (n=='raw_router' or n.startswith('phase_sets.') or n.startswith('fusion_logits.')))
        lr=rates(case,tune)
        for g,a in zip(opt.param_groups,lr):g['lr']=a
        epoch=tune if case['fresh'] else parent['joint_budget_epochs']+tune;order,theta=engine.epoch_plan(epoch,3141,cfg);losses=[]
        for batch,off in enumerate(range(0,3141,16)):
            ix=order[off:off+16];x,y,mask,crop=mixed(data,ix,theta,epoch,batch,cfg,'cuda');q,mass=old.targets(ix,epoch,batch,cfg,'cuda');model.context(seed_for(epoch,batch),len(ix),cfg)
            p,c,loss=gradients(model,x,y,q,mass,mask,weights,cfg,4);final=tune==case['epochs'] and batch==196
            if final:boundary=dict(raw_before={n:p.detach().cpu().clone() for n,p in model.named_parameters()},gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()},optimizer_before=copy.deepcopy(opt.state_dict()),ema_before={n:p.detach().cpu().clone() for n,p in ema.named_parameters()},groups=[[n for n,p in g] for g in named])
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));opt.step();steps+=1
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):
                    if not b.requires_grad:a.copy_(b)
                    else:a.mul_(cfg['ema_decay']).add_(b,alpha=1-cfg['ema_decay'])
            if final:boundary['clip_norm']=norm;engine.save_pt(out/'optimizer_boundary.pt',boundary)
            trace.append(dict(tune=tune,epoch=epoch,batch=batch,n=len(ix),seed=seed_for(epoch,batch),crop_sha256=crop,loss=loss,clip_norm=norm,electronic_only=warm,adam_updates=steps));losses.append(loss)
        model.robust=None;tm,tr=base.evaluate(ema,data);vm,vr=base.evaluate(ema,val)
        core=s.digest({n:p for n,p in model.state_dict().items() if n=='raw_router' or n.startswith('phase_sets.') or n.startswith('fusion_logits.')})
        if warm:assert core==frozen_init
        row=dict(epoch=epoch,tune_epoch=tune,train=tm,val=vm,alpha=ema.alphas(),loss=float(np.mean(losses)),lr=lr,electronic_only=warm,phase_alpha_digest=core,order_sha256=s.digest(order),affine_sha256=s.digest(theta),adam_updates=steps,raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),seconds=time.monotonic()-started,epoch_seconds=time.monotonic()-begin)
        if tune%cfg['test_interval']==0:
            tt,tp=base.evaluate(ema,test);row['test']=tt;base.save_np(out/f'test_epoch{tune:02d}.npz',tp);score=(int(round(tt['accuracy']*411)),-vm['balanced_nll'])
            if score>best:
                best=score;selected=tune;engine.save_pt(out/'best.pt',dict(model={n:v.detach().cpu() for n,v in ema.state_dict().items()},epoch=epoch,tune_epoch=tune,metrics=dict(train=tm,val=vm,test=tt),contract=ema.contract(),sources=meta['sources'],config=cfg,test_reused=False))
                for role,prediction in [('train',tr),('val',vr),('test',tp)]:base.save_np(out/('selected_'+role+'.npz'),prediction)
        history.append(row);engine.save_pt(out/'last.pt',dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},ema={n:v.detach().cpu() for n,v in ema.state_dict().items()},optimizer=opt.state_dict(),epoch=epoch,tune_epoch=tune,steps=steps,sources=meta['sources'],config=cfg))
        s.save(out/'history.json',history);s.save(out/'method_trace.json',trace);s.save(out/'status.json',dict(state='training',tune_epoch=tune,maximum_new_epochs=case['epochs'],electronic_only=warm,selected_tune=selected,train=tm['accuracy'],val=vm['accuracy'],test=row.get('test',{}).get('accuracy'),selected_test=best[0]/411,alphas=ema.alphas()))
        print(json.dumps(dict(key=key,tune=tune,train=tm['accuracy'],val=vm['accuracy'],test=row.get('test',{}).get('accuracy'),alphas=ema.alphas(),selected_test=best[0]/411)),flush=True)
    selected_pt=torch.load(out/'best.pt',map_location='cpu',weights_only=False);old.controls();identity()
    s.save(out/'fit_result.json',dict(passed=True,selected_tune=selected,selected_epoch=selected_pt['epoch'],actual_new_epochs=case['epochs'],previous_budget_epochs=0 if case['fresh'] else parent['joint_budget_epochs'],new_Adam_updates=steps,electronic_only_epochs=case['electronic_warm_epochs'],new_test_model_evaluations=case['epochs']//cfg['test_interval'],metrics=selected_pt['metrics'],TEST_used_for_selection=True,fit_seconds=time.monotonic()-started,**{name+'_sha256':s.sha(out/file) for name,file in [('best','best.pt'),('last','last.pt'),('history','history.json'),('trace','method_trace.json'),('boundary','optimizer_boundary.pt')]}),exclusive=True)
def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists() and resources.slots()['total']==0;ready();claim(OUT.with_name(OUT.name+'.claim.json'),'three_rebalanced_depths_plus_fresh_L4');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),frozen_controls=old.controls()),exclusive=True)
    pending=list(config()['cases']);jobs=[];complete=[];failures=[]
    while pending or jobs:
        used={j['gpu'] for j in jobs}
        for index in config()['gpu_order']:
            gpu=resources.GPU[index]
            if not pending or gpu in used:continue
            key=pending.pop(0);s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            log=(OUT/(key+'.log')).open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit','--key',key,'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(key=key,gpu=gpu,pid=p.pid,process=p,log=log));used.add(gpu)
        for job in list(jobs):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();ex={k:v for k,v in job.items() if k not in ('process','log')};ex.update(returncode=rc,actual_exit=True);s.save(OUT/(job['key']+'.producer_exit.json'),ex,exclusive=True);jobs.remove(job)
            if rc:failures.append(ex);pending=[];continue
            folder=OUT/job['key'];claim(folder/'audit_claim.json','independent_alpha40_saved_science')
            with (folder/'audit.log').open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--key',job['key']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
            if arc:failures.append(dict(key=job['key'],audit_returncode=arc));pending=[];continue
            assert s.read(folder/'science/audit.json')['passed'];s.save(folder/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(folder/'science/audit.json')),exclusive=True);complete.append(ex)
        s.save(OUT/'status.json',dict(state='failed' if failures else 'alpha40_training' if jobs or pending else 'alpha40_all_cases_reviewed',active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in jobs],pending=pending,complete=complete,failures=failures))
        if jobs:time.sleep(2)
    assert not failures
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        folder=OUT/a.key if a.key else OUT;mark=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
