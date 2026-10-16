"""Fresh residual-six regularization from epoch one; immutable controls."""
import argparse,copy,hashlib,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_crossbias as prior
import caltech20_alt246_richsplit_cachedkd as k
import caltech20_alt246_richsplit_viewreg as v
b,g=k.b,k.g
HERE,ROOT,s,engine=k.HERE,k.ROOT,k.s,k.engine
RegularizedMoE=k.RegularizedMoE
PROFILE=HERE/'caltech20_alt246_richsplit_freshreg_profile.json'
FILES=('caltech20_alt246_richsplit_freshreg.py','caltech20_alt246_richsplit_freshreg_profile.json','check_caltech20_alt246_richsplit_freshreg.py','audit_caltech20_alt246_richsplit_freshreg.py','CALTECH20_ALT246_RICHSPLIT_FRESHREG_PROTOCOL_20261017.md')
BASE=k.BASE
CPU=HERE/'runs/smoke/caltech20_richsplit_freshreg_cpu_20261017'
CASES=('regularized','regularized_kd')
versions=k.versions
receipt=k.receipt

def config(case):
    p=s.read(PROFILE);assert case in CASES and not p['automatic_retry'];q=b.config();q.update(p['training']);q.update(kd_alpha=p['kd_alpha'][case],case=case,selection=p['selection']);assert q['maximum_epochs']==100 and q['maximum_seconds']==10800;return q

def sources():
    q=prior.sources()
    for n in FILES:q[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return q

def identity():
    q=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(q,commit);return dict(sources=q,git_commit=commit)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def expected_root(case):
    assert case in CASES;return HERE/'runs/simulation'/('caltech20_rich20_L6_fresh_'+case+'_s17_full100_20261017')

def cpu_ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['versions']==versions();engine.verify_sources(p['sources'],p['git_commit'])

def prerequisites():
    k.base_ready();old=HERE/'runs/simulation'/prior.RUN;assert s.read(old/'status.json')['state']=='crossfit_rejected_no_test';prior.receipt(old,'cv');ex=s.read(old/'campaign_exit.json');assert ex['returncode']==0 and ex['actual_exit'];meta=s.read(old/'metadata.json');assert meta['sources']==prior.sources();return k.teacher_info()

def initial_state():
    engine.seed(17);m=RegularizedMoE(6,.3);f=s.read(BASE/'L6_rho03/fit_result.json');assert s.digest(m.state_dict())==f['initial_digest'];return m

def selected_row(history):return min(history,key=lambda q:(-q['val']['accuracy'],q['val']['nll'],q['epoch']))

def effective_config(cfg,epoch):return dict(cfg,kd_alpha=cfg['kd_alpha']*min(epoch/10,1.))

mixed=v.mixed

def training_backward(model,x,y,mask,weights,cfg,target,eligible):
    model.train();model.zero_grad(set_to_none=True);ps=[];cs=[]
    for off in range(0,len(x),4):
        model.linear.mask=mask[off:off+4];p,c=model(x[off:off+4]);value=k.loss_core(model,p,c,y[off:off+4],weights,cfg,target[off:off+4],eligible[off:off+4]);(value*len(p)/len(x)).backward();ps.append(p.detach());cs.append(c.detach())
    model.linear.mask=None;return torch.cat(ps),torch.cat(cs)

def gate(depth,state,data,cfg,device,epoch,teacher):
    model=RegularizedMoE(depth).to(device);model.load_state_dict(state,strict=True);initial=s.digest(model.state_dict());weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device);order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);checks=[]
    for name,idx,batch in [('all20',np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]),0),('macro16',order[:16],0),('actual_tail',order[-cfg['tail_samples']:],cfg['steps_per_epoch']-1)]:
        x,y,mask,_=mixed(data,idx,theta,epoch,batch,cfg,device)
        target,eligible=k.kd_targets(teacher,idx,epoch,batch,cfg);assert eligible.any() or name!='all20';qt=torch.as_tensor(target,device=device);gt=torch.as_tensor(eligible,device=device)
        for method in ('ordinary','fresh_view_dropout_cached_teacher'):
            used=None if method=='ordinary' else mask;target=None if method=='ordinary' else qt;eligible_used=None if method=='ordinary' else gt;p,c,a=k.gradients(model,x,y,used,weights,cfg,1,target,eligible_used)
            if method=='ordinary':q,d,z=k.gradients(model,x,y,used,weights,cfg,4,target,eligible_used)
            else:
                q,d=training_backward(model,x,y,mask,weights,cfg,qt,gt);z={n:p.grad.detach().cpu().numpy().astype(np.float64) for n,p in model.named_parameters()}
            norms={n:float(np.linalg.norm(v.ravel())) for n,v in z.items()};assert all(np.isfinite(v).all() and norms[n]>0 for n,v in z.items())
            rel=max(float(np.linalg.norm((a[n]-v).ravel())/norms[n]) for n,v in z.items());cos=min(float(np.dot(a[n].ravel(),v.ravel())/(np.linalg.norm(a[n].ravel())*norms[n])) for n,v in z.items());err=max(float((p-q).abs().max()),float((c-d).abs().max()));assert err<=2e-6 and rel<=.02 and cos>=.999,(name,method,err,rel,cos)
            checks.append(dict(samples=name,n=len(idx),method=method,maxabs=err,gradient_relative_l2=rel,gradient_cosine=cos,gradient_norms=norms,teacher_eligible=int(eligible.sum())))
    assert s.digest(model.state_dict())==initial;del model;torch.cuda.empty_cache();return dict(passed=True,initial_digest=initial,checks=checks,optimizer_steps=0,test_arrays_read=0)


def fit(a):
    cpu_ready();teacher_info=prerequisites();cfg=config(a.case);assert a.out==expected_root(a.case)/'fit' and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in b.GPU.values();b.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'fresh_same_initialization_full100');a.out.mkdir()
    s.gpu_budget.ALLOWED.update(b.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);data,val=b.load('train'),b.load('val');teacher=k.teacher_data();fresh=initial_state();initial=s.digest(fresh.state_dict());state={n:p.clone() for n,p in fresh.state_dict().items()};del fresh
    proof=gate(6,state,data,cfg,'cuda',1,teacher);s.save(a.out/'numeric_gate.json',proof,exclusive=True)
    model=initial_state().cuda();assert s.digest(model.state_dict())==initial==proof['initial_digest'];ema=copy.deepcopy(model).eval();phase=[];electronic=[]
    for n,p in model.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(p)
    optimizer=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=cfg['minimum_lr']);weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda()
    meta=dict(identity(),pid=os.getpid(),depth=6,case=a.case,config=cfg,versions=versions(),gpu_uuid=a.gpu,policy=policy,initial_digest=initial,teacher=teacher_info,optimizer_state_restored=False);s.save(a.out/'metadata.json',meta,exclusive=True);history=[];started=time.monotonic();steps=0;trace={};best_key=None
    for epoch in range(1,101):
        assert time.monotonic()-started<cfg['maximum_seconds'],'Full100 budget guard: retain incomplete evidence; no retry';begin=time.monotonic();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);record={n:[] for n in ('mix','flip','mask','view','pixels','teacher_target','teacher_gate')};count=0;support=np.zeros(20,np.int64);kl=0.;predictions=[];lr=[q['lr'] for q in optimizer.param_groups];current=effective_config(cfg,epoch);model.train()
        for batch,off in enumerate(range(0,len(order),16)):
            idx=order[off:off+16];x,y,mask,hashes=mixed(data,idx,theta,epoch,batch,cfg,'cuda');target,eligible=k.kd_targets(teacher,idx,epoch,batch,cfg);hashes.update(teacher_target=s.digest(target),teacher_gate=s.digest(eligible));count+=int(eligible.sum());support+=np.bincount(data[1][idx].numpy(),weights=eligible,minlength=20).astype(np.int64)
            for kind,h in hashes.items():record[kind].append(h)
            # KD coefficient is exactly zero in the plain regularization arm.
            p,_=training_backward(model,x,y,mask,weights,current,torch.as_tensor(target,device='cuda'),torch.as_tensor(eligible,device='cuda'));raw=p.cpu().numpy().astype(np.float64);predictions.append(torch.from_numpy(raw.astype(np.float32)));kl+=k.kl_sum(raw,target,eligible,cfg);torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);previous={n:p.detach().cpu().clone() for n,p in ema.named_parameters()} if epoch==100 and batch==196 else None;optimizer.step();steps+=1
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(cfg['ema_decay']).add_(mp,alpha=1-cfg['ema_decay'])
        trace[epoch]=torch.cat(predictions);tm,tr=b.evaluate(ema,data);vm,vr=b.evaluate(ema,val);row=dict(epoch=epoch,train=tm,val=vm,lr=lr,effective_KD_alpha=current['kd_alpha'],adam_steps=steps,order_sha256=s.digest(order),affine_sha256=s.digest(theta),**{n+'_sha256':hashlib.sha256(''.join(h).encode()).hexdigest() for n,h in record.items()},raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin,kd_eligible_count=count,kd_eligible_support=support.tolist(),kd_KL_T2_sum=kl);history.append(row);key=(-vm['accuracy'],vm['nll'],epoch)
        if best_key is None or key<best_key:
            best_key=key;engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,sources=meta['sources'],config=cfg,initial_digest=initial,contract=ema.contract(),metrics=dict(train=tm,val=vm)));b.save_np(a.out/'selected_train.npz',tr);b.save_np(a.out/'selected_val.npz',vr)
        scheduler.step();engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,steps=steps,sources=meta['sources'],config=cfg,initial_digest=initial,kd_student_probabilities=trace,teacher=teacher_info));s.save(a.out/'history.json',history);s.save(a.out/'status.json',dict(state='training',epoch=epoch,selected_epoch=selected_row(history)['epoch'],train=tm,val=vm));print(dict(case=a.case,epoch=epoch,train=tm['accuracy'],val=vm['accuracy'],seconds=row['epoch_seconds']),flush=True)
    assert steps==19700;selected=selected_row(history);identity();s.save(a.out/'fit_result.json',dict(passed=True,actual_epochs=100,additional_epochs=100,fresh_restart_not_resume=True,selected_epoch=selected['epoch'],best_path=str(a.out/'best.pt'),best_sha256=s.sha(a.out/'best.pt'),selected_train_path=str(a.out/'selected_train.npz'),selected_val_path=str(a.out/'selected_val.npz'),metrics=dict(train=selected['train'],val=selected['val']),steps=steps,fit_seconds=time.monotonic()-started,history_sha256=s.sha(a.out/'history.json'),last_sha256=s.sha(a.out/'last.pt'),stop_reason='maximum_epochs'),exclusive=True)

def test(a):
    root=a.out.parent;receipt(root,'audit_completed.json','science');assert a.out==expected_root(a.case)/'test' and not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'fresh_selected_parameter_digest_once_test');a.out.mkdir();f=s.read(root/'fit/fit_result.json');checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);digest=s.digest(checkpoint['model']);s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),selected_epoch=f['selected_epoch'],best_sha256=f['best_sha256'],model_digest=digest),exclusive=True)
    registry=prior.parent()['test_registry']
    if digest in registry:
        known=registry[digest];assert known['predictions_sha256']==s.sha(known['predictions_path']);result=dict(metrics=known['metrics'],predictions_path=known['predictions_path'],predictions_sha256=known['predictions_sha256'],reused=True,new_test_evaluations=0)
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in b.GPU.values();b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);claim(root/('digest_'+digest+'.claim.json'),'new_parameter_digest_once_test');model=RegularizedMoE(6,.3).cuda();model.load_state_dict(checkpoint['model'],strict=True);metrics,arrays=b.evaluate(model,b.load('test'));b.save_np(a.out/'predictions.npz',arrays);result=dict(metrics=metrics,predictions_path=str(a.out/'predictions.npz'),predictions_sha256=s.sha(a.out/'predictions.npz'),reused=False,new_test_evaluations=1)
    s.save(a.out/'result.json',dict(result,selected_epoch=f['selected_epoch'],best_sha256=f['best_sha256'],model_digest=digest,test_used_for_epoch_selection=False,test_informed_development=True),exclusive=True)

def run_child(a,phase):
    with (a.out/(phase+'.log')).open('x') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--case',a.case,'--out',str(a.out/phase),'--gpu',a.gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu),stdout=log,stderr=subprocess.STDOUT);rc=proc.wait()
    s.save(a.out/(phase+'_exit.json'),dict(pid=proc.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Producer failed: preserve evidence, no retry'

def audit_child(a,phase):
    claim(a.out/(phase+'_audit_claim.json'),'fresh_saved_'+phase+'_math');out=a.out/('science' if phase=='fit' else 'saved_test_audit')
    with (a.out/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_richsplit_freshreg.py'),'--base',str(a.out),'--out',str(out),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Saved audit failed: preserve/no retry';s.save(a.out/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/'audit.json')),exclusive=True)

def campaign(a):
    cpu_ready();info=prerequisites();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==expected_root(a.case) and not a.out.exists() and a.gpu==b.GPU[4 if a.case=='regularized_kd' else 5];b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.preflight(a.gpu,shared=True,memory_gib=2);claim(a.out.with_name(a.out.name+'.campaign_claim.json'),'fresh_registered_residual_six');a.out.mkdir();s.save(a.out/'metadata.json',dict(identity(),pid=os.getpid(),depth=6,case=a.case,config=config(a.case),teacher=info,gpu_uuid=a.gpu),exclusive=True);s.save(a.out/'status.json',dict(state='fresh_full100_training'));run_child(a,'fit');audit_child(a,'fit');s.save(a.out/'status.json',dict(state='fresh_selected_once_test'));run_child(a,'test');audit_child(a,'test');f=s.read(a.out/'fit/fit_result.json');t=s.read(a.out/'test/result.json');control=s.read(BASE/'once_tests/L6_rho0/result.json')['metrics'];four=s.read(BASE/'once_tests/L4_rho03/result.json')['metrics'];s.save(a.out/'comparison.json',dict(depth=6,case=a.case,actual_epochs=100,additional_fresh_epochs=100,selected_epoch=f['selected_epoch'],train=f['metrics']['train'],val=f['metrics']['val'],test=t['metrics'],control100=control,test_gap_pp=100*(t['metrics']['accuracy']-control['accuracy']),six_above_four=t['metrics']['accuracy']>four['accuracy'],test_informed_development=True,controls_frozen=True),exclusive=True);s.save(a.out/'status.json',dict(state='freshreg_science_test_complete',comparison_sha256=s.sha(a.out/'comparison.json')))

def supervise(a):
    cpu_ready();assert a.out==expected_root(a.case) and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(a.out.with_name(a.out.name+'.launch_claim.json'),'finite_experiment_not_scheduled_task')
    with a.out.with_name(a.out.name+'.campaign.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--case',a.case,'--gpu',a.gpu,'--out',str(a.out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);s.save(a.out.with_name(a.out.name+'.started.json'),dict(identity(),supervisor_pid=os.getpid(),campaign_pid=proc.pid,case=a.case),exclusive=True);rc=proc.wait()
    path=a.out/'campaign_exit.json' if a.out.exists() else a.out.with_name(a.out.name+'.campaign_exit.json');s.save(path,dict(pid=proc.pid,supervisor_pid=os.getpid(),returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Campaign failure preserved; no retry'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',choices=CASES,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
