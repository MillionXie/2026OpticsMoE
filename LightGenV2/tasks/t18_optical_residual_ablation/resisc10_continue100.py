"""One authorized residual continuation; sealed169 sources, old165 untouched."""
import argparse,copy,inspect,json,os,subprocess,sys,textwrap,time
from pathlib import Path
import torch
import resisc10_optimize as opt
from audit_caltech_repair234 import exited
s=opt.s;base=opt.base;HERE=base.HERE;ROOT=base.ROOT
PROFILE=HERE/'resisc10_continue100_profile.json'
FILES=('resisc10_continue100.py','resisc10_continue100_profile.json','RESISC10_CONTINUE100_PROTOCOL_20261015.md','audit_resisc10_continue100.py')
def profile():return s.read(PROFILE)
def original():return HERE/'runs/simulation'/profile()['parent_run']
def sources():
    src=opt.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    assert len(src)==169;return src
def parent(stage):
    p=profile();b=original();meta=s.read(b/'metadata.json');state=s.read(b/'status.json');proof=s.read(b/'cpu_report/audit.json');done=s.read(b/'saved_cpu_audit_dispatch_completed_20261015.json')
    assert meta['git_commit']==p['parent_commit'] and meta['sources']==opt.sources()
    s.engine.verify_sources(meta['sources'],meta['git_commit'])
    assert proof['passed'] and done['passed'] and done['audit_sha256']==s.sha(b/'cpu_report/audit.json')==p['parent_audit_sha256']
    assert exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures']
    assert all(j['exit_code']==0 and exited(j['pid']) for j in state['completed'])
    arm=b/p['parent_case'] if stage=='complete100' else HERE/'runs/simulation'/p['run']/'complete100'
    if stage=='regularize20':
        receipt=s.read(arm/'audit_completed.json');assert receipt['passed'] and receipt['audit_sha256']==s.sha(arm/'cpu_report/audit.json')
        assert s.read(arm/'producer_exit.json')['returncode']==0 and s.read(arm/'cpu_report/audit.json')['passed']
    ck=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    expected=92 if stage=='complete100' else 100
    assert ck['epoch']==expected and s.read(arm/'fit_result.json')['epochs_completed']==expected
    if stage=='complete100':assert s.sha(arm/'last_checkpoint.pt')==p['parent_last_sha256']
    assert all(int(v['step'])==350*expected for v in ck['optimizer']['state'].values()) and ck['scheduler']['last_epoch']==expected
    return arm,ck
def configuration(stage):
    arm,ck=parent(stage);cfg=copy.deepcopy(ck['config']);v=profile()[stage]
    cfg.update(maximum_epochs=v['maximum_epochs'],maximum_seconds=v['maximum_seconds'],minimum_epochs=1000,mixup_alpha=v['mixup_alpha'])
    return cfg,arm,ck
def bind(stage):
    cfg,arm,ck=configuration(stage);base.bind();s.engine.read=lambda path:copy.deepcopy(cfg) if Path(path)==base.PLAN else s.read(path);s.engine.sources=sources
    source=textwrap.dedent(inspect.getsource(s.engine.fit))
    changes={
      "20*torch.bincount(data[1],minlength=20).float()":"10*torch.bincount(data[1],minlength=10).float()",
      "for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib')":"for v in (old_cfg,new_cfg):\n            for key in ('shared_spare_memory_gib','maximum_epochs','maximum_seconds','minimum_epochs','mixup_alpha'):v.pop(key)",
      "if offset+time.monotonic()-started>=cfg['maximum_seconds']":"if time.monotonic()-started>=cfg['maximum_seconds']",
      "old_memory_gib=3,new_memory_gib=2":"old_memory_gib=resume['config']['shared_spare_memory_gib'],new_memory_gib=cfg['shared_spare_memory_gib'],stage_additional_seconds=cfg['maximum_seconds'],stage_name=STAGE",
      "if epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience']":"if False and epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience']"
    }
    for old,new in changes.items():assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    needle="scheduler.load_state_dict(resume['scheduler'])"
    addition="\n        for group in opt.param_groups:group['lr']=max(group['lr']*RESUME_FACTOR,2e-5)\n        scheduler._last_lr=[group['lr'] for group in opt.param_groups]"
    assert source.count(needle)==1;source=source.replace(needle,needle+addition)
    ns=dict(s.engine.__dict__,STAGE=stage,RESUME_FACTOR=profile()[stage]['resume_lr_factor'])
    exec(compile(source,__file__+':sealed_stage_fit','exec'),ns)
    return cfg,arm,ns['fit']
def inspect_parent(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();p=profile();arm,ck=parent('complete100');bind('complete100');base.slots()
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    out.mkdir(parents=True,exist_ok=False)
    s.save(out/'checks.json',dict(passed=True,sources=src,git_commit=commit,parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),parent_epoch=ck['epoch'],optimizer_steps=32200,scheduler_epoch=92,source_count=169,models_constructed=0,forward_calls=0,images_read=0,test_arrays_read=0),exclusive=True)
def fit(a):
    assert a.stage in ('complete100','regularize20') and a.gpu==base.GPU[2] and a.rho==.3 and a.depth==4
    assert not a.out.exists();base.slots();cfg,arm,function=bind(a.stage);a.data=Path(cfg['data']);a.resume_parent=arm
    function(a);f=s.read(a.out/'fit_result.json');f['scope']='Human requested residual-only continuation, unchanged10 classes; validation-selected, no test'
    f['continuation']=dict(stage=a.stage,parent=str(arm),parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),prefix_epochs=92 if a.stage=='complete100' else 100,additional_seconds=f['seconds']-s.read(arm/'history.json')[-1]['seconds'],resume_lr_factor=profile()[a.stage]['resume_lr_factor'],earlystop_disabled=True)
    s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',base.eligibility(f,s.read(a.out/'history.json')))
def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();p=profile();assert a.out.name==p['run']
    proof=HERE/'runs/smoke/resisc10_resume100_state_admission_20261015';v=s.read(proof/'checks.json');assert v['passed'] and v['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit'])
    parent('complete100');base.slots();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,profile=p,command=sys.argv,test_evaluations=0),exclusive=True)
    for stage in ('complete100','regularize20'):
        arm=a.out/stage;parent(stage);base.slots();s.gpu_budget.ALLOWED.update(base.GPU.values());s.gpu_budget.preflight(base.GPU[2],shared=True,memory_gib=2)
        with (a.out/(stage+'.log')).open('x') as log:
            cmd=[sys.executable,str(HERE/'resisc10_continue100.py'),'--phase','fit','--stage',stage,'--depth','4','--rho','.3','--gpu',base.GPU[2],'--out',str(arm)]
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU[2]),stdout=log,stderr=subprocess.STDOUT)
            s.save(a.out/'status.json',dict(state='training',stage=stage,pid=child.pid,gpu_uuid=base.GPU[2],test_evaluations=0));rc=child.wait()
        s.save(a.out/(stage+'_producer_exit.json'),dict(pid=child.pid,returncode=rc,actual_exit=exited(child.pid)),exclusive=True)
        assert rc==0 and exited(child.pid),'New stage failed: preserve, no retry'
        s.save(arm/'producer_exit.json',dict(pid=child.pid,returncode=rc,actual_exit=True),exclusive=True)
        assert s.read(arm/'fit_result.json')['epochs_completed']==p[stage]['maximum_epochs'],'Stage safety budget stopped before requested epoch endpoint'
        s.save(arm/'audit_claim.json',dict(pid=os.getpid(),stage=stage,sources=src,started_unix=time.time(),automatic_retry=False),exclusive=True)
        with (arm/'audit.log').open('x') as log:
            rc=subprocess.run([sys.executable,str(HERE/'audit_resisc10_continue100.py'),'--base',str(arm),'--out',str(arm/'cpu_report')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0,'Independent saved science failed: preserve, no downstream retry'
        audit=s.read(arm/'cpu_report/audit.json');assert audit['passed'];s.save(arm/'audit_completed.json',dict(passed=True,audit_sha256=s.sha(arm/'cpu_report/audit.json'),actual_audit_returncode=rc),exclusive=True)
    s.save(a.out/'status.json',dict(state='complete',stages=['complete100','regularize20'],test_evaluations=0))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['inspect','fit','queue'],required=True);parser.add_argument('--stage');parser.add_argument('--out',type=Path,required=True);parser.add_argument('--depth',type=int);parser.add_argument('--rho',type=float);parser.add_argument('--gpu');a=parser.parse_args()
    try:
        if a.phase=='inspect':inspect_parent(a.out)
        else:globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists():
            mark=a.out/'failure.json'
            if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
        raise
