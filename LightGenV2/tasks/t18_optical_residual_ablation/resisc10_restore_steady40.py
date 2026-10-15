"""Human restored original10: one consistent MixUp.2 continuation of audited E100."""
import argparse,copy,inspect,os,subprocess,sys,textwrap,time
from pathlib import Path
import torch
import resisc10_continue100 as original
from audit_caltech_repair234 import exited
s=original.s;base=original.base;HERE=base.HERE;ROOT=base.ROOT
PROFILE=HERE/'resisc10_restore_steady40_profile.json'
FILES=('resisc10_restore_steady40.py','resisc10_restore_steady40_profile.json','audit_resisc10_restore_steady40.py','RESISC10_RESTORE_STEADY40_PROTOCOL_20261015.md')
STAGE='steady40'

def profile():return s.read(PROFILE)

def sources():
    v=original.sources()
    for n in FILES:v[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    assert len(v)==173;return v

def parent(stage=STAGE):
    assert stage==STAGE;p=profile();root=HERE/'runs/simulation'/p['parent_run'];arm=root/p['parent_stage'];meta=s.read(root/'metadata.json')
    assert meta['git_commit']==p['parent_commit'] and meta['sources']==original.sources()
    s.engine.verify_sources(meta['sources'],meta['git_commit']);assert exited(meta['pid']) and s.read(root/'status.json')['state']=='complete'
    receipt=s.read(arm/'audit_completed.json');proof=s.read(arm/'cpu_report/audit.json');ex=s.read(arm/'producer_exit.json')
    assert receipt['passed'] and proof['passed'] and receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(arm/'cpu_report/audit.json')==p['parent_audit_sha256']
    assert ex['returncode']==0 and ex['actual_exit'] and exited(ex['pid'])
    assert s.sha(arm/'last_checkpoint.pt')==p['parent_last_sha256'] and s.sha(arm/'history.json')==p['parent_history_sha256']
    ck=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==100 and ck['scheduler']['last_epoch']==100 and all(int(v['step'])==35000 for v in ck['optimizer']['state'].values())
    assert ck['config']['mixup_alpha']==.2 and ck['config']['classes']==p['classes']
    assert ck['optimizer']['param_groups'][0]['lr']==.0005
    return arm,ck

def configuration(stage=STAGE):
    arm,ck=parent(stage);cfg=copy.deepcopy(ck['config']);p=profile()
    cfg.update(maximum_epochs=p['maximum_epochs'],maximum_seconds=p['additional_fitting_seconds'],minimum_epochs=1000)
    assert cfg['mixup_alpha']==.2 and cfg['maximum_epochs']==140
    return cfg,arm,ck

def bind(stage=STAGE):
    cfg,arm,ck=configuration(stage);base.bind();s.engine.read=lambda path:copy.deepcopy(cfg) if Path(path)==base.PLAN else s.read(path);s.engine.sources=sources
    source=textwrap.dedent(inspect.getsource(s.engine.fit))
    changes={
      '20*torch.bincount(data[1],minlength=20).float()':'10*torch.bincount(data[1],minlength=10).float()',
      "for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib')":"for v in (old_cfg,new_cfg):\n            for key in ('shared_spare_memory_gib','maximum_epochs','maximum_seconds','minimum_epochs'):v.pop(key)",
      "if offset+time.monotonic()-started>=cfg['maximum_seconds']":"if time.monotonic()-started>=cfg['maximum_seconds']",
      'old_memory_gib=3,new_memory_gib=2':"old_memory_gib=resume['config']['shared_spare_memory_gib'],new_memory_gib=cfg['shared_spare_memory_gib'],stage_additional_seconds=cfg['maximum_seconds'],stage_name=STAGE",
      "if epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience']":"if False and epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience']"
    }
    for old,new in changes.items():assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    # No manual LR factor, optimizer/EMA reset or replay of the preserved prefix.
    ns=dict(s.engine.__dict__,STAGE=stage);exec(compile(source,__file__+':sealed_consistent_continuation','exec'),ns)
    return cfg,arm,ns['fit']

def inspect_parent(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();base.slots();arm,ck=parent();bind()
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    claim=a.out.with_name(a.out.name+'.claim.json');assert not claim.exists()
    s.save(claim,dict(pid=os.getpid(),sources=src,git_commit=commit,automatic_retry=False),exclusive=True)
    a.out.mkdir(parents=True,exist_ok=False)
    s.save(a.out/'checks.json',dict(passed=True,sources=src,git_commit=commit,source_count=173,parent_epoch=100,parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),parent_history_sha256=s.sha(arm/'history.json'),optimizer_steps=35000,scheduler_epoch=100,restored_lr=.0005,mixup_alpha=.2,manual_lr_factor=1.,models_constructed=0,forward_calls=0,images_read=0,test_arrays_read=0,fit_source_compile_checked=True),exclusive=True)

def fit(a):
    assert a.stage==STAGE and a.gpu==base.GPU[2] and a.rho==.3 and a.depth==4 and not a.out.exists()
    base.slots();cfg,arm,function=bind();a.data=Path(cfg['data']);a.resume_parent=arm;function(a)
    f=s.read(a.out/'fit_result.json');f.update(scope='Human restored original10, one consistent residual-only E101-140 continuation; historical control is not a new equal-budget ablation',continuation=dict(stage=STAGE,parent=str(arm),parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),prefix_epochs=100,additional_seconds=f['seconds']-s.read(arm/'history.json')[-1]['seconds'],manual_lr_factor=1.,mixup_unchanged=True,earlystop_disabled=True))
    s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',base.eligibility(f,s.read(a.out/'history.json')))
    assert f['epochs_completed']==140 and f['stop_reason']=='maximum_epochs','Safety budget before140: preserve incomplete, no automatic extension'

def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists() and a.out.name==profile()['run']
    proof=HERE/'runs/smoke'/profile()['cpu_proof'];v=s.read(proof/'checks.json');assert v['passed'] and v['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit'])
    parent();base.slots();s.gpu_budget.ALLOWED.update(base.GPU.values());s.gpu_budget.preflight(base.GPU[2],shared=True,memory_gib=2)
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'start_claim.json',dict(pid=os.getpid(),sources=src,git_commit=commit,automatic_retry=False,test_evaluations=0),exclusive=True)
    s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,profile=profile(),command=sys.argv,test_evaluations=0),exclusive=True)
    arm=a.out/STAGE
    with (a.out/'fit.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(HERE/'resisc10_restore_steady40.py'),'--phase','fit','--stage',STAGE,'--depth','4','--rho','.3','--gpu',base.GPU[2],'--out',str(arm)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU[2]),stdout=log,stderr=subprocess.STDOUT)
        s.save(a.out/'status.json',dict(state='training',stage=STAGE,pid=child.pid,gpu_uuid=base.GPU[2],test_evaluations=0));rc=child.wait()
    s.save(arm/'producer_exit.json',dict(pid=child.pid,returncode=rc,actual_exit=exited(child.pid)),exclusive=True)
    assert rc==0 and exited(child.pid),'New continuation failure: preserve, no retry'
    assert s.read(arm/'fit_result.json')['epochs_completed']==140
    s.save(arm/'audit_claim.json',dict(pid=os.getpid(),sources=src,stage=STAGE,automatic_retry=False),exclusive=True)
    with (arm/'audit.log').open('x') as log:
        rc=subprocess.run([sys.executable,str(HERE/'audit_resisc10_restore_steady40.py'),'--base',str(arm),'--out',str(arm/'cpu_report')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(arm/'cpu_report/audit.json')['passed'],'Independent saved audit failed: no retry'
    s.save(arm/'audit_completed.json',dict(passed=True,audit_sha256=s.sha(arm/'cpu_report/audit.json'),actual_audit_returncode=rc),exclusive=True)
    s.save(a.out/'status.json',dict(state='complete_science_passed',stage=STAGE,test_evaluations=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['inspect','fit','queue'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');a=p.parse_args()
    try:inspect_parent(a) if a.phase=='inspect' else globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
        raise
