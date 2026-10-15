"""Human: both100, frozen control, residual continuation without total epoch cap."""
import argparse,copy,inspect,os,subprocess,sys,textwrap,time
from pathlib import Path
import numpy as np
import torch
import caltech10_equal100 as b
s=b.s;parent=b.parent;baseline=b.baseline;resources=b.resources;HERE=b.HERE;ROOT=b.ROOT
PROFILE=HERE/'caltech10_residual_open_profile.json'
FILES=('caltech10_residual_open.py','caltech10_residual_open_profile.json','audit_caltech10_residual_open.py','CALTECH10_RESIDUAL_OPEN_PROTOCOL_20261015.md')
P=s.read(PROFILE);RUN=P['run'];TARGET=125;PARENT=None
CASES={f'L{d}_rho03':(d,.3) for d in (4,3,2)}
parameter_count=b.parameter_count;tensor_count=b.tensor_count;gate=b.gate;wait_slot=b.wait_slot
def sources():
    v=b.sources()
    for n in FILES:v[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return v
def original():return HERE/'runs/simulation'/b.RUN
def config(case):
    assert case in CASES;c=b.config(case);c.update(maximum_epochs=TARGET,maximum_seconds=1800,mixup_alpha=.3,scope=P['scope']);return c
def bind(case):
    parent.bind();cfg=config(case);e=s.engine;e.PROFILE=PROFILE;e.read=lambda p:copy.deepcopy(cfg) if Path(p)==PROFILE else s.read(p);e.sources=sources;e.ALLOWED=resources.GPU;e.fresh_gate=gate
    src=textwrap.dedent(inspect.getsource(e.fit));changes={'20*torch.bincount(data[1],minlength=20).float()':'10*torch.bincount(data[1],minlength=10).float()',"for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib')":"for v in (old_cfg,new_cfg):\n            for k in ('shared_spare_memory_gib','maximum_epochs','maximum_seconds','minimum_epochs','mixup_alpha','scope'):v.pop(k)","if offset+time.monotonic()-started>=cfg['maximum_seconds']":"if time.monotonic()-started>=cfg['maximum_seconds']",'old_memory_gib=3,new_memory_gib=2':"old_memory_gib=resume['config']['shared_spare_memory_gib'],new_memory_gib=cfg['shared_spare_memory_gib'],stage_additional_seconds=cfg['maximum_seconds']"}
    # Each block saves full state; only the outer controller decides when to stop.
    changes["if epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience']"]="if False"
    for old,new in changes.items():assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
    ns=dict(e.__dict__);exec(compile(src,__file__+':open_resume','exec'),ns);return cfg,ns['fit']
def baseline_ready():
    root=original()
    for fail in list(root.glob('*failure.json'))+list(root.glob('*/*failure.json')):raise AssertionError('Baseline failure: '+str(fail))
    if not (root/'status.json').exists():return False
    if s.read(root/'status.json')['state']!='equal100_complete_science_and_frozen_tests_passed':return False
    meta=s.read(root/'metadata.json');assert meta['sources']==b.sources();s.engine.verify_sources(meta['sources'],meta['git_commit'])
    if not parent.base.exited(meta['pid']):return False
    for depth in (4,3,2):
        stage=f'L{depth}';done=s.read(root/(stage+'_audit_completed.json'));assert done['passed'] and done['audit_sha256']==s.sha(root/(stage+'_science')/'audit.json')
        for suffix in ('_rho03','_rho0'):
            case=stage+suffix;f=s.read(root/case/'fit_result.json');rec=s.read(root/case/'producer_exit.json');assert f['epochs_completed']==100 and rec['returncode']==0 and rec['actual_exit'] and parent.base.exited(rec['pid']);out=root/'once_tests'/case;done=s.read(out/'completed.json');assert done['passed'] and done['result_sha256']==s.sha(out/'result.json') and done['saved_audit_sha256']==s.sha(out/'saved_test_audit.json')
    return True
def cpu(root):
    assert baseline_ready();out=root/'cpu_admission';assert not out.exists();checks=[]
    for case in CASES:
        arm=original()/case;ck=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==ck['scheduler']['last_epoch']==100 and all(int(v['step'])==4800 for v in ck['optimizer']['state'].values());bind(case);checks.append(dict(case=case,last_sha256=s.sha(arm/'last_checkpoint.pt'),raw_sha256=s.digest(ck['model']),ema_sha256=s.digest(ck['ema']),optimizer_steps=4800))
    import audit_caltech10_residual_open as audit
    assert callable(audit.verify);s.save(root/'cpu_claim.json',dict(sources=sources(),automatic_retry=False),exclusive=True);out.mkdir();s.save(out/'checks.json',dict(passed=True,sources=sources(),checks=checks,models_constructed=0,forward_calls=0,test_arrays_read=0,reused_graph_proof_sha256=s.sha(b.CPU/'checks.json'),restored_producer_gate_required=True),exclusive=True)
def fit(a):
    assert a.case in CASES and a.rho==.3 and a.depth==CASES[a.case][0];cfg,fn=bind(a.case);assert PARENT is not None and not a.out.exists();meta=s.read(PARENT/'metadata.json');s.engine.verify_sources(meta['sources'],meta['git_commit']);old=s.read(PARENT/'fit_result.json');assert old['epochs_completed']==TARGET-25 and old['rho']==.3;a.data=Path(cfg['data']);a.resume_parent=PARENT;fn(a);f=s.read(a.out/'fit_result.json');f['scope']=P['scope'];s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',parent.eligibility(f,s.read(a.out/'history.json')));assert f['epochs_completed']==TARGET and f['stop_reason']=='maximum_epochs'
def audit_stage(root,stage,cases):
    out=root/(stage+'_science');assert not out.exists();s.save(root/(stage+'_audit_claim.json'),dict(sources=sources(),cases=cases,automatic_retry=False),exclusive=True)
    with (root/(stage+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech10_residual_open.py'),'--root',str(root),'--stage',stage,'--out',str(out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Independent audit failed, preserved no retry';s.save(root/(stage+'_audit_completed.json'),dict(passed=True,audit_sha256=s.sha(out/'audit.json'),actual_audit_returncode=rc),exclusive=True)
# New one-shot test namespaces, with explicit test-informed development labeling.
for name in ('test','test_once'):
    src=textwrap.dedent(inspect.getsource(getattr(b.prior,name)));src=src.replace("'caltech10_followup.py'","'caltech10_residual_open.py'").replace('test_used_for_selection=False','test_used_for_selection=True').replace("scope='Validation-informed fixed10 exploratory simulation, prior test observed'","scope='Test-informed development; feedback controls stopping, not independent generalization'")
    exec(compile(src,__file__+':new_'+name,'exec'),globals())
def worker(a):
    global TARGET,PARENT
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';stage=a.case[:2];depth=CASES[a.case][0];target_pp=P['depth_targets_pp'][str(depth)];arm=original()/a.case;ctl=original()/'once_tests'/(stage+'_rho0');control=s.read(ctl/'result.json');assert control['selected_epoch']<=100
    known={};old=original()/'once_tests'/a.case;ck=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);known[s.digest(ck['model'])]=old;block=0
    while True:
        block+=1;PARENT=arm;previous=s.read(arm/'fit_result.json');TARGET=previous['epochs_completed']+25;out=a.out/f'block{block:04d}';out.mkdir(exist_ok=False);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,parent=str(arm),target_epoch=TARGET,case=a.case,test_informed_development=True),exclusive=True)
        gpu=wait_slot(a.index);log=(out/'fit.log').open('x');child=subprocess.Popen([sys.executable,str(HERE/'caltech10_residual_open.py'),'--phase','fit','--out',str(out/a.case),'--case',a.case,'--depth',str(depth),'--rho','.3','--gpu',gpu,'--target',str(TARGET),'--parent',str(arm)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);s.save(a.out/'status.json',dict(state='residual_finetuning',pid=child.pid,gpu_uuid=gpu,target_epoch=TARGET,total_epoch_cap=None,control_epochs=100,test_gap_target_pp=target_pp));b.prior.finish_job(out,dict(case=a.case,pid=child.pid,gpu_uuid=gpu,process=child,log=log));audit_stage(out,stage,[a.case]);arm=out/a.case;f=s.read(arm/'fit_result.json');ck=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);digest=s.digest(ck['model'])
        if digest in known:
            test_dir=known[digest];v=s.read(test_dir/'result.json');s.save(out/'test_reused.json',dict(model_state_sha256=digest,result_path=str(test_dir),result_sha256=s.sha(test_dir/'result.json'),new_forward_calls=0),exclusive=True)
        else:
            v=test_once(out,a.case,a.index);test_dir=out/'once_tests'/a.case;known[digest]=test_dir
        gap=100*(v['test']['accuracy']-control['test']['accuracy']);hist=s.read(arm/'history.json');significant=min(h['val']['balanced_nll'] for h in hist[:100]);stale=0
        for h in hist[100:]:
            if h['val']['balanced_nll']<significant-P['plateau_minimum_improvement']:significant=h['val']['balanced_nll'];stale=0
            else:stale+=1
        last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);plateau=stale>=P['plateau_patience_epochs'] and last['optimizer']['param_groups'][0]['lr']<=2e-5
        record=dict(depth=depth,actual_epochs=TARGET,selected_epoch=f['selected_epoch'],train=f['metrics']['train'],validation=f['metrics']['val'],test=v['test'],control100_test=control['test'],test_gap_pp=gap,target_gap_pp=target_pp,target_achieved=gap>=target_pp,validation_stale_epochs=stale,selected_model_test_path=str(test_dir),test_evaluations_distinct_selected_models=len(known),test_informed_development=True,independent_generalization=False,control_training_frozen_at=100,extra_residual_budget=True,total_epoch_cap=None)
        s.save(out/'comparison.json',record,exclusive=True);s.save(a.out/'latest_comparison.json',record)
        if gap>=target_pp or plateau:
            s.save(a.out/'status.json',dict(state='target_achieved' if gap>=target_pp else 'paused_validation_plateau_target_unmet',**record));return
def campaign(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==HERE/'runs/simulation'/RUN and not a.out.exists();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);a.out.mkdir(parents=True);s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,profile=P),exclusive=True);s.save(a.out/'claim.json',dict(pid=os.getpid(),sources=src,automatic_retry=False),exclusive=True)
    while not baseline_ready():s.save(a.out/'status.json',dict(state='waiting_exact100_science_and_tests',total_epoch_cap=None));time.sleep(5)
    cpu(a.out);jobs=[]
    for case,index in zip(CASES,(4,2,0)):
        out=a.out/case;out.mkdir();log=(out/'worker.log').open('x');child=subprocess.Popen([sys.executable,str(HERE/'caltech10_residual_open.py'),'--phase','worker','--out',str(out),'--case',case,'--index',str(index)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);jobs.append((child,log,case));s.save(out/'worker_started.json',dict(pid=child.pid,case=case,index=index),exclusive=True)
    s.save(a.out/'status.json',dict(state='three_residuals_finetuning_control100_frozen',total_epoch_cap=None))
    for child,log,case in jobs:
        rc=child.wait();log.close();s.save(a.out/case/'worker_exit.json',dict(pid=child.pid,returncode=rc,actual_exit=parent.base.exited(child.pid)),exclusive=True);assert rc==0,'Depth failed; no automatic retry'
    s.save(a.out/'status.json',dict(state='all_depths_stopped_targets_or_plateau',depths={k:s.read(a.out/k/'status.json') for k in CASES},total_epoch_cap=None,test_informed_development=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['campaign','worker','fit','test'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');p.add_argument('--index',type=int);p.add_argument('--target',type=int,default=125);p.add_argument('--parent',type=Path);a=p.parse_args();TARGET=a.target;PARENT=a.parent
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
