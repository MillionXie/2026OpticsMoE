"""L2-only refit of an audited325 prefix; human target82 simulation test."""
import argparse,copy,inspect,os,subprocess,sys,textwrap,time
from pathlib import Path
import numpy as np
import torch
import caltech10_residual_open as old
s=old.s;parent=old.parent;baseline=old.baseline;resources=old.resources;HERE=old.HERE;ROOT=old.ROOT
PROFILE=HERE/'caltech10_l2_refit_profile.json';P=s.read(PROFILE);RUN=P['run'];CPU=HERE/'runs/smoke/caltech10_l2_refit82_state_20261015';NUMERIC=HERE/'runs/smoke/caltech10_l2_refit82_gpu2_numeric_20261015'
FILES=('caltech10_l2_refit.py','caltech10_l2_refit_profile.json','audit_caltech10_l2_refit.py','CALTECH10_L2_REFIT82_PROTOCOL_20261015.md')
CASES={'L2_rho03':(2,.3)};TARGET=350;PARENT=Path(P['parent_arm']);gate=old.gate;wait_slot=old.wait_slot;parameter_count=old.parameter_count;tensor_count=old.tensor_count
def sources():
    src=old.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src
def parent_state():
    root=HERE/'runs/simulation'/old.RUN;meta=s.read(root/'metadata.json');assert meta['sources']==old.sources();s.engine.verify_sources(meta['sources'],meta['git_commit']);assert s.read(root/'status.json')['state']=='all_depths_stopped_targets_or_plateau' and parent.base.exited(meta['pid'])
    for case in old.CASES:
        rec=s.read(root/case/'worker_exit.json');assert rec['returncode']==0 and rec['actual_exit'] and parent.base.exited(rec['pid'])
    arm=Path(P['parent_arm']);assert s.sha(arm/'last_checkpoint.pt')==P['parent_last_sha256'] and s.sha(arm/'best_checkpoint.pt')==P['parent_best_sha256'] and s.sha(arm/'history.json')==P['parent_history_sha256'];done=s.read(arm.parent/'L2_audit_completed.json');assert done['passed'] and done['audit_sha256']==P['parent_audit_sha256']==s.sha(arm.parent/'L2_science/audit.json')
    ck=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==ck['scheduler']['last_epoch']==325 and len(ck['optimizer']['state'])==6 and all(int(v['step'])==15600 for v in ck['optimizer']['state'].values()) and ck['optimizer']['param_groups'][0]['lr']==2e-5;return arm,ck
def config(case):
    assert case in CASES;c=old.b.config(case);c.update(maximum_epochs=TARGET,maximum_seconds=P['block_fit_seconds'],maximum_gpu_children=1,mixup_alpha=P['mixup_alpha'],label_smoothing=P['label_smoothing'],phase_smooth_weight=P['phase_smooth_weight'],refit_restart_epoch=326,refit_restart_lr=P['restart_lr'],cpu_gate=CPU.relative_to(ROOT).as_posix()+'/checks.json',numeric_gate=NUMERIC.relative_to(ROOT).as_posix()+'/checks.json',scope=P['scope']);return c
def bind(case):
    parent.bind();cfg=config(case);e=s.engine;e.PROFILE=PROFILE;e.read=lambda p:copy.deepcopy(cfg) if Path(p)==PROFILE else s.read(p);e.sources=sources;e.ALLOWED=resources.GPU;e.fresh_gate=gate
    src=textwrap.dedent(inspect.getsource(e.fit));changes={'20*torch.bincount(data[1],minlength=20).float()':'10*torch.bincount(data[1],minlength=10).float()',"for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib')":"for v in (old_cfg,new_cfg):\n            for k in ('shared_spare_memory_gib','maximum_epochs','maximum_seconds','minimum_epochs','mixup_alpha','label_smoothing','phase_smooth_weight','maximum_gpu_children','cpu_gate','numeric_gate','scope','refit_restart_epoch','refit_restart_lr'):v.pop(k,None)","if offset+time.monotonic()-started>=cfg['maximum_seconds']":"if time.monotonic()-started>=cfg['maximum_seconds']",'old_memory_gib=3,new_memory_gib=2':"old_memory_gib=resume['config']['shared_spare_memory_gib'],new_memory_gib=cfg['shared_spare_memory_gib'],stage_additional_seconds=cfg['maximum_seconds']","if epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience']":"if False",'optimizer_reset=False,ema_reset=False,scheduler_reset=False':"optimizer_reset=False,ema_reset=False,scheduler_reset=(resume['epoch']==325),scheduler_restart_epoch=(326 if resume['epoch']==325 else None),scheduler_restart_lr=(cfg['refit_restart_lr'] if resume['epoch']==325 else None),adam_moments_preserved=True"}
    anchor="best=best_ck['validation']['balanced_nll'];start_epoch=resume['epoch']+1;offset=history[-1]['seconds']"
    changes[anchor]=anchor+"\n        if resume['epoch']==325:\n            for group in opt.param_groups:group['lr']=cfg['refit_restart_lr']\n            scheduler.best=float('inf');scheduler.num_bad_epochs=0;scheduler.cooldown_counter=0;scheduler._last_lr=[cfg['refit_restart_lr']]"
    for a,b in changes.items():assert src.count(a)==1,(a,src.count(a));src=src.replace(a,b)
    ns=dict(e.__dict__);exec(compile(src,__file__+':L2_refit','exec'),ns);return cfg,ns['fit']
def cpu(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==CPU and not a.out.exists();resources.slots();arm,ck=parent_state();cfg,_=bind('L2_rho03');assert cfg['capture_weight']==.2 and cfg['ema_decay']==.95 and cfg['mixup_alpha']==.1 and cfg['label_smoothing']==.02 and cfg['phase_smooth_weight']==.01
    import audit_caltech10_l2_refit as auditor
    assert callable(auditor.independent_verifier()) and callable(auditor.independent_scheduler());src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=src,git_commit=commit,automatic_retry=False),exclusive=True);a.out.mkdir(parents=True);s.save(a.out/'checks.json',dict(passed=True,sources=src,git_commit=commit,parent_epoch=325,parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),parent_history_sha256=s.sha(arm/'history.json'),raw_sha256=s.digest(ck['model']),ema_sha256=s.digest(ck['ema']),optimizer_steps=15600,old_lr=2e-5,new_phase_lr=.0005,scheduler_statistics_restart_at=326,optimizer_moments_reset=False,ema_reset=False,models_constructed=0,forward_calls=0,test_arrays_read=0),exclusive=True)
def prerequisites(numeric=True):
    for folder in ((CPU,NUMERIC) if numeric else (CPU,)):
        v=s.read(folder/'checks.json');assert v['passed'] and v['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit'])
def numeric(a):
    assert a.out==NUMERIC and not a.out.exists() and a.gpu==resources.GPU[2] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;prerequisites(False);resources.slots();arm,ck=parent_state();cfg,_=bind('L2_rho03');s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);m=parent.manifest(Path(cfg['data']));data=parent.load_fit(Path(cfg['data']),'train',m);src=sources();s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=src,automatic_retry=False),exclusive=True);v=gate(2,.3,data,cfg,'cuda',8,ck['model']);assert v['passed'] and v['parameters']==479364 and len(v['phase_gradients'])==6;a.out.mkdir(parents=True);s.save(a.out/'checks.json',dict(passed=True,sources=src,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),restored_state=v,optimizer_steps=0,test_arrays_read=0),exclusive=True)
def fit(a):
    assert a.case=='L2_rho03' and a.depth==2 and a.rho==.3 and not a.out.exists();prerequisites();meta=s.read(PARENT/'metadata.json');s.engine.verify_sources(meta['sources'],meta['git_commit']);assert s.read(PARENT/'fit_result.json')['epochs_completed']==TARGET-25;cfg,fn=bind(a.case);a.data=Path(cfg['data']);a.resume_parent=PARENT;fn(a);v=s.read(a.out/'fit_result.json');v['scope']=P['scope'];s.save(a.out/'fit_result.json',v);s.save(a.out/'validation_gate.json',parent.eligibility(v,s.read(a.out/'history.json')));assert v['epochs_completed']==TARGET and v['stop_reason']=='maximum_epochs'
def audit_stage(root):
    out=root/'L2_science';assert not out.exists();s.save(root/'L2_audit_claim.json',dict(sources=sources(),cases=['L2_rho03'],automatic_retry=False),exclusive=True)
    with (root/'L2_audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech10_l2_refit.py'),'--root',str(root),'--out',str(out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Independent audit failed; preserve no retry';s.save(root/'L2_audit_completed.json',dict(passed=True,audit_sha256=s.sha(out/'audit.json'),actual_audit_returncode=rc),exclusive=True)
for name in ('test','test_once'):
    src=textwrap.dedent(inspect.getsource(getattr(old.b.prior,name))).replace("'caltech10_followup.py'","'caltech10_l2_refit.py'").replace('test_used_for_selection=False','test_used_for_selection=True').replace("scope='Validation-informed fixed10 exploratory simulation, prior test observed'","scope='Human L2 target82; test-informed development, frozen100 control'");exec(compile(src,__file__+':new_'+name,'exec'),globals())
def campaign(a):
    global TARGET,PARENT
    assert a.out==HERE/'runs/simulation'/RUN and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();prerequisites();parent_state();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);a.out.mkdir(parents=True);s.save(a.out/'claim.json',dict(pid=os.getpid(),sources=src,git_commit=commit,automatic_retry=False),exclusive=True);s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,profile=P),exclusive=True)
    arm=Path(P['parent_arm']);oldroot=HERE/'runs/simulation'/old.RUN;known={}
    for block in sorted((oldroot/'L2_rho03').glob('block*')):
        if not (block/'comparison.json').exists():continue
        c=s.read(block/'comparison.json');ck=torch.load(block/'L2_rho03/best_checkpoint.pt',map_location='cpu',weights_only=False);known[s.digest(ck['model'])]=Path(c['selected_model_test_path'])
    first=old.original()/'L2_rho03';ck=torch.load(first/'best_checkpoint.pt',map_location='cpu',weights_only=False);known[s.digest(ck['model'])]=old.original()/'once_tests/L2_rho03';inherited=len(known);assert inherited==7;ctrl=s.read(old.original()/'once_tests/L2_rho0/result.json');blockno=0;newtests=0
    while True:
        blockno+=1;PARENT=arm;TARGET=s.read(arm/'fit_result.json')['epochs_completed']+25;out=a.out/f'block{blockno:04d}';out.mkdir();s.save(out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,parent=str(arm),target_epoch=TARGET,case='L2_rho03',test_informed_development=True),exclusive=True);gpu=wait_slot(2);log=(out/'fit.log').open('x');child=subprocess.Popen([sys.executable,str(HERE/'caltech10_l2_refit.py'),'--phase','fit','--case','L2_rho03','--depth','2','--rho','.3','--out',str(out/'L2_rho03'),'--gpu',gpu,'--target',str(TARGET),'--parent',str(arm)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);s.save(a.out/'status.json',dict(state='L2_refit_training',pid=child.pid,gpu_uuid=gpu,target_epoch=TARGET,total_epoch_cap=None,target_test_accuracy=.82,control_frozen100=True));old.b.prior.finish_job(out,dict(case='L2_rho03',pid=child.pid,gpu_uuid=gpu,process=child,log=log));audit_stage(out);arm=out/'L2_rho03';fit=s.read(arm/'fit_result.json');ck=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);digest=s.digest(ck['model'])
        if digest in known:td=known[digest];test=s.read(td/'result.json');s.save(out/'test_reused.json',dict(model_state_sha256=digest,result_path=str(td),result_sha256=s.sha(td/'result.json'),new_forward_calls=0),exclusive=True)
        else:test=test_once(out,'L2_rho03',2);td=out/'once_tests/L2_rho03';known[digest]=td;newtests+=1
        hist=s.read(arm/'history.json');significant=min(h['val']['balanced_nll'] for h in hist[:325]);stale=0
        for h in hist[325:]:
            if h['val']['balanced_nll']<significant-P['plateau_delta']:significant=h['val']['balanced_nll'];stale=0
            else:stale+=1
        last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);lr=last['optimizer']['param_groups'][0]['lr'];passed=test['test']['accuracy']>=P['target_test_accuracy'];paused=stale>=P['plateau_patience'] and lr<=2e-5;c=dict(actual_epochs=TARGET,new_epochs=TARGET-325,selected_epoch=fit['selected_epoch'],train=fit['metrics']['train'],validation=fit['metrics']['val'],test=test['test'],control100_test=ctrl['test'],test_gap_pp=100*(test['test']['accuracy']-ctrl['test']['accuracy']),target_test_accuracy=.82,target_achieved=passed,last_lr=lr,validation_stale_epochs=stale,selected_model_test_path=str(td),new_selected_model_tests=newtests,inherited_selected_model_tests=inherited,total_selected_model_tests=len(known),test_informed_development=True,extra_residual_budget=True,total_epoch_cap=None);s.save(out/'comparison.json',c,exclusive=True);s.save(a.out/'latest_comparison.json',c)
        if passed or paused:s.save(a.out/'status.json',dict(state='target82_achieved' if passed else 'paused_L2_refit_plateau_target82_unmet',**c));return
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','numeric','fit','test','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');p.add_argument('--target',type=int,default=350);p.add_argument('--parent',type=Path);a=p.parse_args();TARGET=a.target;PARENT=a.parent or Path(P['parent_arm'])
    try:globals()[a.phase](a)
    except BaseException as e:
        fail=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not fail.exists():s.save(fail,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
