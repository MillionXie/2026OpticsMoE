"""Evidence-led distillation and quantization-scaled SAM, frozen100 controls."""
import argparse,copy,hashlib,inspect,math,os,subprocess,sys,textwrap,time
from pathlib import Path
import numpy as np
import torch
import caltech10_sam_dropout as prior
import caltech10_targeted_diagnosis as diagnosis
import caltech10_l2_refit as previous2
import caltech10_l4_refine as previous4
old=previous4.old;s=old.s;parent=old.parent;baseline=old.baseline;wait_slot=old.wait_slot;resources=old.resources;HERE=old.HERE;ROOT=old.ROOT
PROFILE=HERE/'caltech10_targeted_refinement_profile.json';P=s.read(PROFILE);RUN=P['run'];CPU=HERE/'runs/smoke/caltech10_targeted_refinement_cpu_20261015';NUMERIC=HERE/'runs/smoke/caltech10_targeted_refinement_numeric_20261015'
FILES=('caltech10_targeted_refinement.py','caltech10_targeted_refinement_profile.json','audit_caltech10_targeted_refinement.py','CALTECH10_TARGETED_REFINEMENT_PROTOCOL_20261015.md')
CASES=P['cases'];TARGET=None;PARENT=None

def sources():
    src=diagnosis.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src

def sam_gradients(parameters,closure,radius):
    """Two full-macro passes, same inputs; restore exact parameters before Adam."""
    parameters=list(parameters);loss=closure();first=[q.grad.detach().clone() for q in parameters]
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in first)
    norm=torch.stack([g.norm() for g in first]).norm();assert torch.isfinite(norm) and norm>0
    saved=[q.detach().clone() for q in parameters]
    with torch.no_grad():
        for q,g in zip(parameters,first):q.add_(g,alpha=float(radius/(norm+1e-12)))
    perturb=float(torch.stack([(q.detach()-w).norm() for q,w in zip(parameters,saved)]).norm())
    try:
        for q in parameters:q.grad=None
        second_loss=closure();assert all(q.grad is not None and torch.isfinite(q.grad).all() and q.grad.norm()>0 for q in parameters)
    finally:
        with torch.no_grad():
            for q,w in zip(parameters,saved):q.copy_(w)
    assert all(torch.equal(q.detach(),w) for q,w in zip(parameters,saved))
    assert .95*radius<=perturb<=1.05*radius,(perturb,radius)
    return loss,dict(sam=True,gradient_passes=2,first_grad_norm=float(norm),perturbation_norm=perturb,second_loss=second_loss,exact_restore=True)

def mask_cpu(shape,probability,seed):
    b,h,w=shape;g=torch.Generator().manual_seed(seed)
    coarse=(torch.rand((b,math.ceil(h/16),math.ceil(w/16)),generator=g)>=probability)
    return coarse.repeat_interleave(16,1).repeat_interleave(16,2)[:,:h,:w]

TEACHER=None
def teacher_model(device):
    global TEACHER
    if TEACHER is None:
        assert s.sha(P['teacher_checkpoint'])==P['teacher_checkpoint_sha256'];ck=torch.load(P['teacher_checkpoint'],map_location='cpu',weights_only=False);assert s.digest(ck['model'])==P['teacher_model_digest'];TEACHER=parent.SubsetMoE(4,.3).to(device);TEACHER.load_state_dict(ck['model'],strict=True);TEACHER.eval()
        for q in TEACHER.parameters():q.requires_grad_(False)
    assert TEACHER.raw_router.device==torch.device(device)
    return TEACHER

def macro_gradients(model,x,y,weights,cfg,micro,epoch,batch):
    record=dict(epoch=epoch,batch=batch,shape=list(x.shape),dropout_probability=0.,seed=cfg['seed']+epoch*1000003+batch*97,teacher_batches=0,teacher_active_samples=0,teacher_kl=0.,teacher_alpha=cfg['teacher_alpha'])
    def closure():
        for q in model.parameters():q.grad=None
        value=0.
        for k in range(0,len(x),micro):
            p,c=model(x[k:k+micro]);loss=s.engine.objective(model,p,c,y[k:k+micro],weights,cfg)
            if cfg['teacher_alpha']:
                teacher=teacher_model(x.device)
                with torch.no_grad():
                    tp,_=teacher(x[k:k+micro]);gate=(tp.argmax(1)==y[k:k+micro].argmax(1))&(tp.max(1).values>=cfg['teacher_min_confidence']);temperature=cfg['teacher_temperature'];target=torch.softmax(tp.clamp_min(1e-12).log()/temperature,1)
                student_log=torch.log_softmax(p.clamp_min(1e-12).log()/temperature,1);kl=(target*(target.clamp_min(1e-12).log()-student_log)).sum(1)*temperature**2;term=(kl*gate).mean();assert torch.isfinite(term);loss=loss+cfg['teacher_alpha']*term;record['teacher_batches']+=1;record['teacher_active_samples']+=int(gate.sum());record['teacher_kl']+=float(term.detach())*len(p)/len(x);assert all(q.grad is None and not q.requires_grad for q in teacher.parameters())
            assert torch.isfinite(loss);(loss*len(p)/len(x)).backward();value+=float(loss.detach())*len(p)/len(x)
        return value
    if cfg['sam_radius']:
        value,sam=sam_gradients(model.parameters(),closure,cfg['sam_radius']);record.update(sam)
    else:value=closure();record.update(sam=False,gradient_passes=1,exact_restore=True)
    return value,record

def config(case):
    c=CASES[case];cfg=copy.deepcopy(c['parent_config']);cfg.update(maximum_epochs=TARGET or c['parent_epoch']+10,maximum_seconds=1800,minimum_epochs=1000,maximum_gpu_children=2,sam_radius=c['sam_radius'],input_dropout=c['input_dropout'],method_restart_epoch=c['parent_epoch']+1,method_restart_lr=c['restart_lr'],teacher_alpha=c['teacher_alpha'],teacher_temperature=c['teacher_temperature'],teacher_min_confidence=c['teacher_min_confidence'],teacher_checkpoint_sha256=P['teacher_checkpoint_sha256'],diagnosis_sha256=P['diagnosis_sha256'],cpu_gate=CPU.relative_to(ROOT).as_posix()+'/checks.json',numeric_gate=(NUMERIC/case).relative_to(ROOT).as_posix()+'/checks.json',scope=P['scope']);return cfg

def bind(case):
    parent.bind();cfg=config(case);e=s.engine;e.PROFILE=PROFILE;e.read=lambda p:copy.deepcopy(cfg) if Path(p)==PROFILE else s.read(p);e.sources=sources;e.ALLOWED=resources.GPU;e.fresh_gate=gate
    src=textwrap.dedent(inspect.getsource(e.fit));excluded=('shared_spare_memory_gib','maximum_epochs','maximum_seconds','minimum_epochs','mixup_alpha','label_smoothing','phase_smooth_weight','ema_decay','maximum_gpu_children','cpu_gate','numeric_gate','scope','refit_restart_epoch','refit_restart_lr','sam_radius','input_dropout','method_restart_epoch','method_restart_lr','teacher_alpha','teacher_temperature','teacher_min_confidence','teacher_checkpoint_sha256','diagnosis_sha256')
    changes={'20*torch.bincount(data[1],minlength=20).float()':'10*torch.bincount(data[1],minlength=10).float()',"for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib')":f"for v in (old_cfg,new_cfg):\n            for key in {excluded!r}:v.pop(key,None)","if offset+time.monotonic()-started>=cfg['maximum_seconds']":"if time.monotonic()-started>=cfg['maximum_seconds']",'old_memory_gib=3,new_memory_gib=2':"old_memory_gib=resume['config']['shared_spare_memory_gib'],new_memory_gib=cfg['shared_spare_memory_gib'],stage_additional_seconds=cfg['maximum_seconds']","if epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience']":"if False",'optimizer_reset=False,ema_reset=False,scheduler_reset=False':"optimizer_reset=False,ema_reset=False,scheduler_reset=(resume['epoch']==cfg['method_restart_epoch']-1),scheduler_restart_epoch=(cfg['method_restart_epoch'] if resume['epoch']==cfg['method_restart_epoch']-1 else None),scheduler_restart_lr=(cfg['method_restart_lr'] if resume['epoch']==cfg['method_restart_epoch']-1 else None),adam_moments_preserved=True"}
    anchor="best=best_ck['validation']['balanced_nll'];start_epoch=resume['epoch']+1;offset=history[-1]['seconds']";changes[anchor]=anchor+"\n        if start_epoch==cfg['method_restart_epoch']:\n            for group in opt.param_groups:group['lr']=cfg['method_restart_lr']\n            scheduler.best=float('inf');scheduler.num_bad_epochs=0;scheduler.cooldown_counter=0;scheduler._last_lr=[cfg['method_restart_lr']]"
    a="            for k in range(0,len(idx),cfg['microbatch']):\n                p,c=model(x[k:k+cfg['microbatch']]);loss=objective(model,p,c,y[k:k+cfg['microbatch']],weights,cfg)\n                assert torch.isfinite(loss);(loss*len(p)/len(idx)).backward();total+=float(loss.detach())*len(p)"
    changes[a]="            value,method=macro_gradients(model,x,y,weights,cfg,cfg['microbatch'],epoch,batch);total+=value*len(idx);method_records.append(method)"
    changes['mixhash=hashlib.sha256();total=0.']='mixhash=hashlib.sha256();total=0.;method_records=[]'
    changes["vm,rows=evaluate(ema,val,cfg['microbatch']);entry=dict("]= "save(a.out/('method_epoch%04d.json'%epoch),dict(epoch=epoch,records=method_records),exclusive=True)\n        vm,rows=evaluate(ema,val,cfg['microbatch']);entry=dict("
    for a,b in changes.items():assert src.count(a)==1,(a,src.count(a));src=src.replace(a,b)
    ns=dict(e.__dict__,macro_gradients=macro_gradients);exec(compile(src,__file__+':method_fit','exec'),ns);return cfg,ns['fit']

def gate(depth,rho,data,cfg,device,micro,state=None):
    base=old.gate(depth,rho,data,cfg,device,micro,state);m=parent.SubsetMoE(depth,rho).to(device);m.load_state_dict(state,strict=True);initial=s.digest(m.state_dict());weights=(len(data[1])/(10*torch.bincount(data[1],minlength=10).float())).to(device);_,theta=s.engine.epoch_plan(1,len(data[1]),cfg);checks=[]
    for name,idx in [('all10',np.array([int((data[1]==k).nonzero()[0]) for k in range(10)])),('macro16',np.arange(16)),('tail9',np.arange(9))]:
        x,y,_=s.engine.mixed_batch(data,idx,theta,1,0,cfg,device);results=[]
        for microbatch in (1,8):
            m.load_state_dict(state,strict=True);loss,record=macro_gradients(m,x,y,weights,cfg,microbatch,CASES[next(k for k,c in CASES.items() if c['depth']==depth and c['sam_radius']==cfg['sam_radius'] and c['input_dropout']==cfg['input_dropout'])]['parent_epoch']+1,0);results.append((loss,[q.grad.detach().cpu().clone() for q in m.parameters()],record));assert s.digest(m.state_dict())==initial
        rel=max(float((a-b).norm()/(a.norm()+1e-12)) for a,b in zip(results[0][1],results[1][1]));assert rel<.02 and abs(results[0][0]-results[1][0])<1e-4 and all(torch.isfinite(g).all() and g.norm()>0 for g in results[1][1]);checks.append(dict(group=name,passed=True,gradient_max_relative_l2=rel,reference=results[0][2],micro8=results[1][2]))
    if cfg['teacher_alpha']:assert s.digest(TEACHER.state_dict())==P['teacher_model_digest'] and all(not q.requires_grad and q.grad is None for q in TEACHER.parameters())
    del m;torch.cuda.empty_cache();base.update(method_checks=checks,method_optimizer_steps=0,model_state_preserved=True,teacher_parameter_state_preserved=True,teacher_checkpoint_sha256=P['teacher_checkpoint_sha256']);return base

def parent_state(case):
    c=CASES[case];arm=Path(c['parent_arm']);root=HERE/'runs/simulation'/prior.RUN;meta=s.read(root/'metadata.json');assert meta['sources']==prior.sources() and parent.base.exited(meta['pid']) and s.read(root/'status.json')['state']=='all_methods_stopped_targets_or_plateau';s.engine.verify_sources(meta['sources'],meta['git_commit']);assert s.sha(arm/'last_checkpoint.pt')==c['parent_last_sha256'] and s.sha(arm/'best_checkpoint.pt')==c['parent_best_sha256'] and s.sha(arm/'history.json')==c['parent_history_sha256'];done=s.read(arm.parent/'audit_completed.json');assert done['passed'] and done['actual_audit_returncode']==0 and done['audit_sha256']==c['parent_audit_sha256']==s.sha(arm.parent/'science/audit.json');ck=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==ck['scheduler']['last_epoch']==c['parent_epoch'] and all(int(q['step'])==48*c['parent_epoch'] for q in ck['optimizer']['state'].values());return arm,ck

def prerequisites(case=None):
    proof=s.read(P['diagnosis_path']);assert proof['passed'] and s.sha(P['diagnosis_path'])==P['diagnosis_sha256'] and proof['sources']==diagnosis.sources();s.engine.verify_sources(proof['sources'],proof['git_commit']);assert s.sha(P['teacher_checkpoint'])==P['teacher_checkpoint_sha256'];dirs=[CPU] if case is None else [CPU,NUMERIC/case]
    for out in dirs:v=s.read(out/'checks.json');assert v['passed'] and v['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit'])

def cpu(a):
    assert a.out==CPU and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';assert resources.slots()['total']==0
    import audit_caltech10_targeted_refinement as audit
    for case in CASES:parent_state(case);bind(case);assert callable(audit.independent_verifier(CASES[case])) and callable(audit.independent_scheduler(CASES[case]))
    w=torch.nn.Parameter(torch.tensor([.4,-.2],dtype=torch.float64));original=w.detach().clone()
    def closure():w.grad=None;loss=(w*w).sum();loss.backward();return float(loss.detach())
    _,v=sam_gradients([w],closure,.05);expected=2*(original+.05*original/original.norm());assert torch.equal(w.detach(),original) and torch.allclose(w.grad,expected,atol=1e-12,rtol=0)
    opt=torch.optim.AdamW([w],lr=.001,weight_decay=0);opt.step();assert int(opt.state[w]['step'])==1
    mask=mask_cpu((2,31,35),.05,17);assert torch.equal(mask,mask_cpu((2,31,35),.05,17)) and mask.shape==(2,31,35) and not mask_cpu((2,31,35),1.,17).any() and mask_cpu((2,31,35),0.,17).all()
    proof=s.read(P['diagnosis_path']);assert proof['passed'] and proof['sources']==diagnosis.sources() and s.sha(P['diagnosis_path'])==P['diagnosis_sha256'];assert s.sha(P['teacher_checkpoint'])==P['teacher_checkpoint_sha256'];teacher_ck=torch.load(P['teacher_checkpoint'],map_location='cpu',weights_only=False);assert s.digest(teacher_ck['model'])==P['teacher_model_digest']
    logits=torch.tensor([1.,.2,-.3],dtype=torch.float64,requires_grad=True);target=torch.tensor([.7,.2,.1],dtype=torch.float64);temp=2.;logp=torch.log_softmax(logits/temp,0);kl=(target*(target.log()-logp)).sum()*temp**2;gradient=torch.autograd.grad(kl,logits)[0];assert torch.allclose(gradient,temp*(logp.exp()-target),rtol=0,atol=1e-12)
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=src,automatic_retry=False),exclusive=True);a.out.mkdir(parents=True);s.save(a.out/'checks.json',dict(passed=True,sources=src,git_commit=commit,real_models=0,real_optimizer_steps=0,real_forward_calls=0,test_arrays_read=0,toy_SAM_analytic_gradient=True,toy_exact_restore=True,toy_Adam_steps=1,deterministic_dropout=True,teacher_analytic_gradient=True,teacher_checkpoint_sha256=P['teacher_checkpoint_sha256']),exclusive=True)

def numeric(a):
    c=CASES[a.case];assert a.out==NUMERIC/a.case and not a.out.exists() and a.gpu==resources.GPU[c['gpu']] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;prerequisites();resources.slots();arm,ck=parent_state(a.case);cfg,_=bind(a.case);s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);data=parent.load_fit(Path(cfg['data']),'train',parent.manifest(Path(cfg['data'])));NUMERIC.mkdir(parents=True,exist_ok=True);s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=sources(),automatic_retry=False),exclusive=True);v=gate(c['depth'],.3,data,cfg,'cuda',8,ck['model']);assert v['passed'];a.out.mkdir(parents=True);s.save(a.out/'checks.json',dict(passed=True,sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),check=v,optimizer_steps=0,test_arrays_read=0),exclusive=True)

def fit(a):
    c=CASES[a.case];assert not a.out.exists() and a.depth==c['depth'] and a.rho==.3;prerequisites(a.case);cfg,fn=bind(a.case);a.data=Path(cfg['data']);a.resume_parent=PARENT;assert s.read(PARENT/'fit_result.json')['epochs_completed']==TARGET-10;fn(a);f=s.read(a.out/'fit_result.json');f['scope']=P['scope'];s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',parent.eligibility(f,s.read(a.out/'history.json')));assert f['epochs_completed']==TARGET and f['stop_reason']=='maximum_epochs'

def audit_stage(root,case):
    s.save(root/'audit_claim.json',dict(sources=sources(),case=case,automatic_retry=False),exclusive=True)
    with (root/'audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech10_targeted_refinement.py'),'--root',str(root),'--case',case,'--out',str(root/'science')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(root/'science/audit.json')['passed'];s.save(root/'audit_completed.json',dict(passed=True,audit_sha256=s.sha(root/'science/audit.json'),actual_audit_returncode=rc),exclusive=True)

for name in ('test','test_once'):
    src=textwrap.dedent(inspect.getsource(getattr(old.b.prior,name))).replace("'caltech10_followup.py'","'caltech10_targeted_refinement.py'").replace("root/(stage+'_audit_completed.json')","root/'audit_completed.json'").replace("root/(stage+'_science')/'audit.json'","root/'science/audit.json'").replace('test_used_for_selection=False','test_used_for_selection=True').replace("scope='Validation-informed fixed10 exploratory simulation, prior test observed'","scope='Targeted distillation/qSAM test-informed development; frozen100 control'");exec(compile(src,__file__+':'+name,'exec'),globals())

def known_tests(depth):
    known=prior.known_tests(depth);root=HERE/'runs/simulation'/prior.RUN
    for case in prior.CASES:
        if not case.startswith(f'L{depth}_'):continue
        for block in sorted((root/case).glob('block*')):
            if not (block/'comparison.json').exists():continue
            c=s.read(block/'comparison.json');ck=torch.load(block/case/'best_checkpoint.pt',map_location='cpu',weights_only=False);known[s.digest(ck['model'])]=Path(c['selected_model_test_path'])
    return known

def worker(a):
    global PARENT,TARGET
    c=CASES[a.case];depth=c['depth'];known=known_tests(depth);inherited=len(known);assert inherited>=c['minimum_inherited_tests'];ctrl=s.read(old.original()/'once_tests'/f'L{depth}_rho0/result.json');arm=Path(c['parent_arm']);blockno=0;newtests=0
    while True:
        blockno+=1;PARENT=arm;TARGET=s.read(arm/'fit_result.json')['epochs_completed']+10;out=a.out/f'block{blockno:04d}';out.mkdir();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,parent=str(arm),target_epoch=TARGET,case=a.case),exclusive=True);gpu=old.wait_slot(c['gpu']);log=(out/'fit.log').open('x');child=subprocess.Popen([sys.executable,str(HERE/'caltech10_targeted_refinement.py'),'--phase','fit','--case',a.case,'--depth',str(depth),'--rho','.3','--out',str(out/a.case),'--gpu',gpu,'--target',str(TARGET),'--parent',str(arm)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);s.save(a.out/'status.json',dict(state='method_training',pid=child.pid,gpu_uuid=gpu,target_epoch=TARGET));old.b.prior.finish_job(out,dict(case=a.case,pid=child.pid,gpu_uuid=gpu,process=child,log=log));audit_stage(out,a.case);arm=out/a.case;fit=s.read(arm/'fit_result.json');ck=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);digest=s.digest(ck['model'])
        if digest in known:td=known[digest];t=s.read(td/'result.json');s.save(out/'test_reused.json',dict(model_state_sha256=digest,result_path=str(td),result_sha256=s.sha(td/'result.json'),new_forward_calls=0),exclusive=True)
        else:
            shared=a.out.parent/'test_digest_claims';shared.mkdir(exist_ok=True);claim=shared/(digest+'.json');td=out/'once_tests'/a.case
            try:s.save(claim,dict(pid=os.getpid(),digest=digest,test_path=str(td)),exclusive=True);owner=True
            except FileExistsError:owner=False
            if owner:t=test_once(out,a.case,c['gpu']);newtests+=1
            else:
                other=s.read(claim);td=Path(other['test_path']);begin=time.monotonic()
                while not (td/'completed.json').exists():assert time.monotonic()-begin<300 and not parent.base.exited(other['pid']),'Shared selected test owner failed';time.sleep(1)
                complete=s.read(td/'completed.json');assert complete['passed'] and complete['result_sha256']==s.sha(td/'result.json');t=s.read(td/'result.json');s.save(out/'test_reused.json',dict(model_state_sha256=digest,result_path=str(td),result_sha256=s.sha(td/'result.json'),new_forward_calls=0,cross_candidate_reuse=True),exclusive=True)
            known[digest]=td
        hist=s.read(arm/'history.json');significant=min(h['val']['balanced_nll'] for h in hist[:c['parent_epoch']]);stale=0
        for h in hist[c['parent_epoch']:]:
            if h['val']['balanced_nll']<significant-P['review_delta']:significant=h['val']['balanced_nll'];stale=0
            else:stale+=1
        best_acc=max(h['val']['accuracy'] for h in hist[:c['parent_epoch']]);best_train=next(h['train']['accuracy'] for h in reversed(hist[:c['parent_epoch']]) if 'train' in h);accuracy_stale=0;training_stale=0
        for h in hist[c['parent_epoch']:]:
            accuracy_stale+=1;training_stale+=1
            if h['val']['accuracy']>best_acc+1e-12:best_acc=h['val']['accuracy'];accuracy_stale=0
            if 'train' in h and h['train']['accuracy']>best_train+.005:best_train=h['train']['accuracy'];training_stale=0
        last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);lr=last['optimizer']['param_groups'][0]['lr'];gap=100*(t['test']['accuracy']-ctrl['test']['accuracy']);passed=t['test']['accuracy']>=.82 if depth==2 else gap>=10;paused=stale>=P['review_stale_epochs'] and accuracy_stale>=20 and (depth==4 or training_stale>=10);record=dict(depth=depth,method=c['method'],actual_epochs=TARGET,new_epochs=TARGET-c['parent_epoch'],selected_epoch=fit['selected_epoch'],train=fit['metrics']['train'],validation=fit['metrics']['val'],test=t['test'],control100_test=ctrl['test'],test_gap_pp=gap,target_achieved=passed,last_lr=lr,validation_stale_epochs=stale,validation_accuracy_stale_epochs=accuracy_stale,training_accuracy_stale_epochs=training_stale,current_endpoint_train=hist[-1].get('train'),current_endpoint_val=hist[-1]['val'],selected_model_test_path=str(td),new_selected_model_tests=newtests,inherited_selected_model_tests=inherited,total_selected_model_tests=len(known),test_informed_development=True,extra_residual_budget=True,total_epoch_cap=None);s.save(out/'comparison.json',record,exclusive=True);s.save(a.out/'latest_comparison.json',record)
        if passed or paused:s.save(a.out/'status.json',dict(state='target_achieved' if passed else 'strategy_review_required_target_unmet',**record));return

def campaign(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists() and a.out==HERE/'runs/simulation'/RUN
    for case in CASES:prerequisites(case);parent_state(case)
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);a.out.mkdir(parents=True);s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,profile=P),exclusive=True);s.save(a.out/'claim.json',dict(sources=src,automatic_retry=False),exclusive=True);jobs=[]
    for case in CASES:
        dest=a.out/case;dest.mkdir();log=(dest/'worker.log').open('x');child=subprocess.Popen([sys.executable,str(HERE/'caltech10_targeted_refinement.py'),'--phase','worker','--case',case,'--out',str(dest)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(case=case,pid=child.pid,process=child,log=log))
    s.save(a.out/'status.json',dict(state='targeted_refinements_running',workers=[dict(case=j['case'],pid=j['pid']) for j in jobs]))
    for j in jobs:rc=j['process'].wait();j['log'].close();s.save(a.out/j['case']/'worker_exit.json',dict(pid=j['pid'],returncode=rc,actual_exit=parent.base.exited(j['pid'])),exclusive=True);assert rc==0
    s.save(a.out/'status.json',dict(state='all_targeted_stopped_targets_or_review',cases={c:s.read(a.out/c/'status.json') for c in CASES}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','numeric','fit','test','worker','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--target',type=int);p.add_argument('--parent',type=Path);a=p.parse_args();TARGET=a.target;PARENT=a.parent
    try:globals()[a.phase](a)
    except BaseException as e:
        fail=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not fail.exists():s.save(fail,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
