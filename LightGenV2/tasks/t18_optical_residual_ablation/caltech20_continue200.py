"""Human-authorized equal-budget continuation of current4 Caltech pair."""
import argparse,copy,inspect,os,subprocess,sys,textwrap
from pathlib import Path
import torch
import caltech20_current4 as base
s=base.s;HERE=base.HERE;ROOT=base.ROOT
PROFILE=HERE/'caltech20_continue200_profile.json'
FILES=('caltech20_continue200.py','caltech20_continue200_profile.json','audit_caltech20_continue200.py','CALTECH20_CONTINUE200_PROTOCOL_20261015.md','audit_resisc20_input_pair.py')
RUN='caltech20_current4_s17_pair_continue200_gpu25_20261015'
CASES=base.CASES;GPU=base.GPU
CPU=HERE/'runs/smoke/caltech20_continue200_state_20261015'

def profile():return s.read(PROFILE)
def original():return HERE/'runs/simulation'/base.RUN
def sources():
    src=base.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src
def parent(key):
    p=profile();root=original();meta=s.read(root/'metadata.json');done=s.read(root/'audit_completed.json')
    assert meta['git_commit']==p['parent_commit'] and meta['sources']==base.sources();s.engine.verify_sources(meta['sources'],meta['git_commit'])
    assert s.sha(HERE/'audit_resisc20_input_pair.py')==p['audit_helper_sha256']
    assert done['passed'] and done['audit_sha256']==s.sha(root/'cpu_report/audit.json')==p['parent_audit_sha256']
    state=s.read(root/'status.json');assert state['state']=='caltech_current4_complete_science_passed' and all(j['exit_code']==0 and base.exited(j['pid']) for j in state['completed'])
    assert base.exited(meta['pid']) and base.exited(p['parent_campaign_pid'])
    arm=root/key;assert s.sha(arm/'last_checkpoint.pt')==p['parent_last_sha256'][key]
    ck=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==100 and ck['scheduler']['last_epoch']==100 and all(int(v['step'])==8600 for v in ck['optimizer']['state'].values())
    return arm,ck
def config(key=None):
    arm,ck=parent(key or next(iter(CASES)));cfg=copy.deepcopy(ck['config']);cfg.update(maximum_epochs=200,maximum_seconds=3600,minimum_epochs=1000);return cfg
def bind(key):
    cfg=config(key);base.bind();s.engine.read=lambda path:copy.deepcopy(cfg) if Path(path)==base.PROFILE else s.read(path);s.engine.sources=sources
    src=textwrap.dedent(inspect.getsource(s.engine.fit))
    changes={"for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib')":"for v in (old_cfg,new_cfg):\n            for k in ('shared_spare_memory_gib','maximum_epochs','maximum_seconds','minimum_epochs'):v.pop(k)","if offset+time.monotonic()-started>=cfg['maximum_seconds']":"if time.monotonic()-started>=cfg['maximum_seconds']","old_memory_gib=3,new_memory_gib=2":"old_memory_gib=resume['config']['shared_spare_memory_gib'],new_memory_gib=cfg['shared_spare_memory_gib'],stage_additional_seconds=cfg['maximum_seconds']"}
    for old,new in changes.items():assert src.count(old)==1;src=src.replace(old,new)
    ns=dict(s.engine.__dict__);exec(compile(src,__file__+':sealed200fit','exec'),ns);return cfg,ns['fit']
def cpu(a):
    assert a.out==CPU and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';base.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'new_equal200_state')
    states={}
    for key in CASES:
        arm,ck=parent(key);bind(key);states[key]=dict(parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),parent_history_sha256=s.sha(arm/'history.json'),epoch=100,adam_steps=8600,scheduler_epoch=100,lr=ck['optimizer']['param_groups'][0]['lr'],raw_sha256=s.digest(ck['model']),ema_sha256=s.digest(ck['ema']))
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'checks.json',dict(passed=True,parents=states,sources=src,git_commit=commit,models_constructed=0,forward_calls=0,images_read=0,test_arrays_read=0),exclusive=True)
def prerequisites():
    v=s.read(CPU/'checks.json');assert v['passed'] and v['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit']);return config()
def fit(a):
    assert a.case in CASES and a.depth==4 and a.rho==CASES[a.case] and a.gpu in GPU.values();prerequisites();base.slots();cfg,fn=bind(a.case);arm,ck=parent(a.case);a.data=Path(cfg['data']);a.resume_parent=arm;fn(a)
    f=s.read(a.out/'fit_result.json');f.update(scope='Human paired continuation to200; current4/fixed20; validation only',continuation=dict(parent=str(arm),prefix_epochs=100,parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),additional_seconds=f['seconds']-s.read(arm/'history.json')[-1]['seconds'],optimizer_reset=False,ema_reset=False,scheduler_reset=False));s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',s.eligibility(f,s.read(a.out/'history.json')))
    assert f['epochs_completed']==200 and f['stop_reason']=='maximum_epochs','Incomplete200: preserve, no extension'
def claim(path,phase):
    assert not path.exists();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(path,dict(pid=os.getpid(),phase=phase,sources=src,git_commit=commit,automatic_retry=False,test_evaluations=0),exclusive=True)
_q=textwrap.dedent(inspect.getsource(base.queue));_ns=dict(base.__dict__,__file__=__file__,prerequisites=prerequisites,sources=sources,config=config,RUN=RUN);exec(compile(_q,__file__+':pairedqueue','exec'),_ns);queue=_ns['queue']
_c=textwrap.dedent(inspect.getsource(base.campaign));assert _c.count("'audit_caltech20_current4.py'")==1;assert _c.count("['epochs_completed']==100")==1
_c=_c.replace("'audit_caltech20_current4.py'","'audit_caltech20_continue200.py'").replace("['epochs_completed']==100","['epochs_completed']==200").replace("'caltech_current4_complete_science_passed'","'caltech_continue200_complete_science_passed'")
_ns=dict(base.__dict__,__file__=__file__,prerequisites=prerequisites,sources=sources,config=config,RUN=RUN,claim=claim);exec(compile(_c,__file__+':pairedcampaign','exec'),_ns);campaign=_ns['campaign']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','fit','queue','campaign'],required=True);p.add_argument('--out',type=Path,required=True)
    for k in ('case','gpu'):p.add_argument('--'+k)
    p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--data',type=Path);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
