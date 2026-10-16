"""Longer EMA window after fresh100; same inference, losses and LR floor."""
import argparse,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_richsplit_objective_diagnosis as diagnosis
fresh=diagnosis.fresh
b,k,g,s,engine,HERE,ROOT=fresh.b,fresh.k,fresh.g,fresh.s,fresh.engine,fresh.HERE,fresh.ROOT
RegularizedMoE=fresh.RegularizedMoE
PROFILE=HERE/'caltech20_alt246_richsplit_emastable_profile.json'
FILES=('caltech20_alt246_richsplit_emastable.py','caltech20_alt246_richsplit_emastable_profile.json','check_caltech20_alt246_richsplit_emastable.py','audit_caltech20_alt246_richsplit_emastable.py','CALTECH20_ALT246_RICHSPLIT_EMASTABLE_PROTOCOL_20261017.md')
CASES=('ema_only','ema_phaseTV25')
def expected_root(case):
    assert case in CASES;return HERE/'runs/simulation'/('caltech20_rich20_L6_EMA999_'+case+'_to120_s17_20261017')
def card(case):return 4 if case=='ema_phaseTV25' else 5
CPU=HERE/'runs/smoke/caltech20_richsplit_emastable_cpu_20261017'
BASE=fresh.BASE
versions=fresh.versions
receipt=fresh.receipt

def sources():
    q=diagnosis.sources()
    for n in FILES:q[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return q

def identity():
    q=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(q,commit);return dict(sources=q,git_commit=commit)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def parent():
    candidates=[];registry=dict(fresh.prior.parent()['test_registry']);profile=s.read(PROFILE)
    for case in fresh.CASES:
        root=fresh.expected_root(case);assert s.read(root/'status.json')['state']=='freshreg_science_test_complete';meta=s.read(root/'metadata.json');assert meta['sources']==fresh.sources() and meta['config']==fresh.config(case);engine.verify_sources(meta['sources'],meta['git_commit']);receipt(root,'audit_completed.json','science');receipt(root,'test_audit_completed.json','saved_test_audit');ex=s.read(root/'campaign_exit.json');assert ex['returncode']==0 and ex['actual_exit']
        for pid in (ex['pid'],ex['supervisor_pid']):
            p=Path('/proc',str(pid),'cmdline');assert not p.exists() or str(root).encode() not in p.read_bytes()
        f=s.read(root/'fit/fit_result.json');t=s.read(root/'test/result.json');assert f['actual_epochs']==100 and f['steps']==19700 and f['best_sha256']==s.sha(f['best_path']) and f['last_sha256']==s.sha(root/'fit/last.pt') and f['history_sha256']==s.sha(root/'fit/history.json') and t['predictions_sha256']==s.sha(t['predictions_path']);registry[t['model_digest']]=dict(result_path=str(root/'test/result.json'),predictions_path=t['predictions_path'],predictions_sha256=t['predictions_sha256'],metrics=t['metrics']);candidates.append(dict(case=case,root=str(root),f=f,training_commit=meta['git_commit']))
    chosen=min(candidates,key=lambda q:(-q['f']['metrics']['val']['accuracy'],q['f']['metrics']['val']['nll'],q['f']['selected_epoch'],q['case']));f=chosen['f'];root=Path(chosen['root']);diagnostic=diagnosis.OUT/'diagnosis.json';assert s.sha(diagnostic)==profile['TRAIN_diagnosis_sha256'];proof=s.read(diagnostic);assert proof['passed'] and proof['optimizer_updates']==proof['VAL_images_read']==proof['TEST_images_read']==0;assert s.read(diagnosis.OUT.with_name(diagnosis.OUT.name+'.producer_exit.json'))['returncode']==0
    return dict(case=chosen['case'],root=str(root),training_commit=chosen['training_commit'],actual_epochs=100,steps=19700,last=str(root/'fit/last.pt'),last_sha256=f['last_sha256'],best=f['best_path'],best_sha256=f['best_sha256'],train=f['selected_train_path'],val=f['selected_val_path'],selected_epoch=f['selected_epoch'],metrics=f['metrics'],history_path=str(root/'fit/history.json'),history_sha256=f['history_sha256'],test_registry=registry)

def config(case):
    p=s.read(PROFILE);assert case in CASES;assert p['ema_decay']==.999 and p['additional_epochs']==20 and p['maximum_seconds']==3600 and p['automatic_retry'] is False;parent_info=parent();q=fresh.config(parent_info['case']);q.update(ema_decay=p['ema_decay'],block_epochs=20,maximum_seconds=p['maximum_seconds'],selection=p['selection'],phase_smooth_weight=p['phase_smooth_weights'][case],continuation_case=case);return q

def cpu_ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['versions']==versions();engine.verify_sources(q['sources'],q['git_commit'])

def known_registry(case):
    registry=dict(parent()['test_registry'])
    for other in CASES:
        if other==case:continue
        root=expected_root(other);result=root/'test/result.json';exit_path=root/'test_exit.json'
        if not result.exists() or not exit_path.exists():continue
        ex=s.read(exit_path);assert ex['returncode']==0 and ex['actual_exit'];receipt(root,'audit_completed.json','science');q=s.read(result);assert q['predictions_sha256']==s.sha(q['predictions_path']);registry[q['model_digest']]=dict(result_path=str(result),predictions_path=q['predictions_path'],predictions_sha256=q['predictions_sha256'],metrics=q['metrics'])
    return registry

def shared_digest_claim(case,digest):
    directory=HERE/'runs/simulation/caltech20_rich20_EMA999_shared_digest_claims_20261017';directory.mkdir(exist_ok=True);path=directory/(digest+'.json')
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            registry=known_registry(case)
            if digest in registry:return registry[digest]
            time.sleep(2)
        raise AssertionError('Digest owner did not complete once-test: preserve/no duplicate inference')
    with os.fdopen(fd,'w') as f:json.dump(dict(identity(),owner_root=str(expected_root(case)),case=case,model_digest=digest),f);f.flush();os.fsync(f.fileno())
    return None

def fit(a):
    cpu_ready();p=parent();cfg=config(a.case);assert not a.out.exists() and a.out==expected_root(a.case)/'fit' and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu==b.GPU[card(a.case)];b.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'restore_raw_EMA_Adam100_change_only_EMA_horizon');a.out.mkdir();s.gpu_budget.ALLOWED.update(b.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);old=torch.load(p['last'],map_location='cpu',weights_only=False);assert old['epoch']==100 and old['steps']==19700 and old['scheduler']['last_epoch']==100;data,val=b.load('train'),b.load('val');teacher=k.teacher_data();proof=fresh.gate(6,old['model'],data,cfg,'cuda',101,teacher);s.save(a.out/'numeric_gate.json',proof,exclusive=True);model=RegularizedMoE(6,.3).cuda();model.load_state_dict(old['model'],strict=True);ema=copy.deepcopy(model).eval();ema.load_state_dict(old['ema'],strict=True);phase=[];electronic=[]
    for n,w in model.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(w)
    opt=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr']);opt.load_state_dict(old['optimizer']);assert g.optimizer_moment_digest(opt.state_dict())==g.optimizer_moment_digest(old['optimizer']) and np.allclose([q['lr'] for q in opt.param_groups],cfg['minimum_lr'],atol=1e-12,rtol=0)
    meta=dict(identity(),pid=os.getpid(),depth=6,config=cfg,parent=p,versions=versions(),gpu_uuid=a.gpu,policy=policy,initial_raw_digest=s.digest(old['model']),initial_EMA_digest=s.digest(old['ema']),initial_optimizer_moment_digest=g.optimizer_moment_digest(opt.state_dict()),teacher=k.teacher_info());s.save(a.out/'metadata.json',meta,exclusive=True);h=s.read(p['history_path']);selected=dict(epoch=p['selected_epoch'],best_path=p['best'],train_path=p['train'],val_path=p['val'],metrics=p['metrics']);key=(-p['metrics']['val']['accuracy'],p['metrics']['val']['nll'],p['selected_epoch']);trace=dict(old['kd_student_probabilities']);steps=19700;started=time.monotonic();weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda()
    for epoch in range(101,121):
        assert time.monotonic()-started<cfg['maximum_seconds'],'Incomplete20: retain/no retry';begin=time.monotonic();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);record={n:[] for n in ('mix','flip','mask','view','pixels','teacher_target','teacher_gate')};count=0;support=np.zeros(20,np.int64);kl=0.;predictions=[];lr=[q['lr'] for q in opt.param_groups];model.train()
        for batch,off in enumerate(range(0,len(order),16)):
            idx=order[off:off+16];x,y,mask,hashes=fresh.mixed(data,idx,theta,epoch,batch,cfg,'cuda');q,gate=k.kd_targets(teacher,idx,epoch,batch,cfg);hashes.update(teacher_target=s.digest(q),teacher_gate=s.digest(gate));count+=int(gate.sum());support+=np.bincount(data[1][idx].numpy(),weights=gate,minlength=20).astype(np.int64)
            for kind,z in hashes.items():record[kind].append(z)
            prob,_=fresh.training_backward(model,x,y,mask,weights,cfg,torch.as_tensor(q,device='cuda'),torch.as_tensor(gate,device='cuda'));raw=prob.cpu().numpy().astype(np.float64);predictions.append(torch.from_numpy(raw.astype(np.float32)));kl+=k.kl_sum(raw,q,gate,cfg);torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);previous={n:w.detach().cpu().clone() for n,w in ema.named_parameters()} if epoch==120 and batch==196 else None;opt.step();steps+=1
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(cfg['ema_decay']).add_(mp,alpha=1-cfg['ema_decay'])
        trace[epoch]=torch.cat(predictions);tm,tr=b.evaluate(ema,data);vm,vr=b.evaluate(ema,val);row=dict(epoch=epoch,train=tm,val=vm,lr=lr,effective_KD_alpha=cfg['kd_alpha'],adam_steps=steps,order_sha256=s.digest(order),affine_sha256=s.digest(theta),**{n+'_sha256':hashlib.sha256(''.join(z).encode()).hexdigest() for n,z in record.items()},raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin,raw_phase_smoothness=float(model.phase_smoothness().detach()),EMA_phase_smoothness=float(ema.phase_smoothness().detach()),kd_eligible_count=count,kd_eligible_support=support.tolist(),kd_KL_T2_sum=kl);h.append(row);current=(-vm['accuracy'],vm['nll'],epoch)
        if current<key:
            key=current;engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,sources=meta['sources'],config=cfg,contract=ema.contract(),metrics=dict(train=tm,val=vm)));b.save_np(a.out/'selected_train.npz',tr);b.save_np(a.out/'selected_val.npz',vr);selected=dict(epoch=epoch,best_path=str(a.out/'best.pt'),train_path=str(a.out/'selected_train.npz'),val_path=str(a.out/'selected_val.npz'),metrics=dict(train=tm,val=vm))
        engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous,optimizer=opt.state_dict(),parent_scheduler=old['scheduler'],scheduler=dict(type='constant_after_cosine100',last_epoch=epoch,epochs_after_parent=epoch-100,lr=cfg['minimum_lr']),epoch=epoch,steps=steps,sources=meta['sources'],config=cfg,parent=p,initial_digest=old['initial_digest'],kd_student_probabilities=trace,teacher=meta['teacher']));s.save(a.out/'history.json',h);s.save(a.out/'status.json',dict(state='training',epoch=epoch,selected_epoch=selected['epoch'],train=tm,val=vm));print(dict(epoch=epoch,train=tm['accuracy'],val=vm['accuracy'],selected_epoch=selected['epoch']),flush=True)
    assert steps==23640;identity();s.save(a.out/'fit_result.json',dict(passed=True,actual_epochs=120,additional_epochs=20,selected_epoch=selected['epoch'],best_path=selected['best_path'],best_sha256=s.sha(selected['best_path']),selected_train_path=selected['train_path'],selected_val_path=selected['val_path'],metrics=selected['metrics'],steps=steps,fit_seconds=time.monotonic()-started,history_sha256=s.sha(a.out/'history.json'),last_sha256=s.sha(a.out/'last.pt'),stop_reason='registered20_epoch_EMA_horizon_review'),exclusive=True)

def test(a):
    root=a.out.parent;receipt(root,'audit_completed.json','science');assert a.out==expected_root(a.case)/'test' and not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'fresh_selected_parameter_digest_once_test');a.out.mkdir();f=s.read(root/'fit/fit_result.json');checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);digest=s.digest(checkpoint['model']);s.save(a.out/'started.json',dict(identity(),pid=os.getpid(),selected_epoch=f['selected_epoch'],best_sha256=f['best_sha256'],model_digest=digest),exclusive=True)
    registry=known_registry(a.case)
    if digest not in registry:
        shared=shared_digest_claim(a.case,digest)
        if shared is not None:registry[digest]=shared
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
    with (a.out/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_richsplit_emastable.py'),'--base',str(a.out),'--out',str(out),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Saved audit failed: preserve/no retry';s.save(a.out/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/'audit.json')),exclusive=True)

def supervise(a):
    cpu_ready();assert a.out==expected_root(a.case) and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';claim(a.out.with_name(a.out.name+'.launch_claim.json'),'finite_experiment_not_scheduled_task')
    with a.out.with_name(a.out.name+'.campaign.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--case',a.case,'--gpu',a.gpu,'--out',str(a.out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);s.save(a.out.with_name(a.out.name+'.started.json'),dict(identity(),supervisor_pid=os.getpid(),campaign_pid=proc.pid,case=a.case),exclusive=True);rc=proc.wait()
    path=a.out/'campaign_exit.json' if a.out.exists() else a.out.with_name(a.out.name+'.campaign_exit.json');s.save(path,dict(pid=proc.pid,supervisor_pid=os.getpid(),returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Campaign failure preserved; no retry'

def campaign(a):
    cpu_ready();p=parent();assert a.out==expected_root(a.case) and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.gpu==b.GPU[card(a.case)];b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.preflight(a.gpu,shared=True,memory_gib=2);claim(a.out.with_name(a.out.name+'.campaign_claim.json'),'single_registered_EMA_horizon_method');a.out.mkdir();s.save(a.out/'metadata.json',dict(identity(),pid=os.getpid(),depth=6,case=a.case,parent=p,config=config(a.case),gpu_uuid=a.gpu),exclusive=True);s.save(a.out/'status.json',dict(state='EMAstable_fit_running'));run_child(a,'fit');audit_child(a,'fit');s.save(a.out/'status.json',dict(state='EMAstable_selected_once_test'));run_child(a,'test');audit_child(a,'test');f=s.read(a.out/'fit/fit_result.json');t=s.read(a.out/'test/result.json');control=s.read(BASE/'once_tests/L6_rho0/result.json')['metrics'];four=s.read(BASE/'once_tests/L4_rho03/result.json')['metrics'];s.save(a.out/'comparison.json',dict(depth=6,actual_epochs=120,additional_epochs=20,selected_epoch=f['selected_epoch'],train=f['metrics']['train'],val=f['metrics']['val'],test=t['metrics'],control100=control,test_gap_pp=100*(t['metrics']['accuracy']-control['accuracy']),six_above_four=t['metrics']['accuracy']>four['accuracy'],test_informed_development=True,controls_frozen=True),exclusive=True);s.save(a.out/'status.json',dict(state='EMAstable_science_test_complete',comparison_sha256=s.sha(a.out/'comparison.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',choices=CASES,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
