"""Full-resolution L4 residual pair with manifest-sized fitting and six-child slots."""
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import caltech_repair234 as r
import gpu_budget
engine=r.engine;read,save,sha,digest=r.read,r.save,r.sha,r.digest
PROFILE=HERE/'resisc20_profile.json';ALLOWED=r.ALLOWED
NEW_FILES=('resisc20.py','resisc20_profile.json','resisc20_data_plan.json','prepare_resisc20.py','download_resisc45.py','audit_resisc20.py','check_resisc20.py','RESISC20_PROTOCOL_20261014.md','export_caltech_repair234.py','check_caltech_repair234_export.py')

def sources():
    src=r.sources()
    for name in NEW_FILES:src[(HERE/name).relative_to(ROOT).as_posix()]=sha(HERE/name)
    return src

def manifest(data):
    cfg=read(PROFILE);assert cfg['cache_sha256'] and cfg['manifest_sha256'],'Dataset not frozen'
    assert data==Path(cfg['data']);m=read(data.parent/'data_manifest.json')
    assert sha(data)==cfg['cache_sha256']==m['cache_sha256'] and sha(data.parent/'data_manifest.json')==cfg['manifest_sha256']
    assert m['dataset']=='NWPU_RESISC45_prespecified20_20261014' and m['classes']==read(HERE/'resisc20_data_plan.json')['classes']
    assert m['total_records']==14000 and len(m['classes'])==20 and all(min(m['supports'][s])>0 for s in ('train','val','test'))
    assert sum(sum(m['supports'][s]) for s in ('train','val','test'))==14000
    return m

def slots(include_self=True):
    # Only inspect command identity and CUDA_VISIBLE_DEVICES for project processes;
    # no unrelated process termination, environment changes or credential output.
    jobs=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or (not include_self and int(p.name)==os.getpid()):continue
        try:
            cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(HERE) not in cmd:continue
            gpu=next((v.decode().split('=',1)[1] for v in p.joinpath('environ').read_bytes().split(b'\0') if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if gpu in ALLOWED.values():jobs.append(dict(pid=int(p.name),gpu_uuid=gpu))
        except (OSError,UnicodeError):continue
    counts={gpu:sum(j['gpu_uuid']==gpu for j in jobs) for gpu in ALLOWED.values()}
    assert len(jobs)<=6 and max(counts.values(),default=0)<=2,(jobs,'Project GPU slots exceeded')
    return dict(jobs=jobs,counts=counts,total=len(jobs),maximum=6,per_card_maximum=2)

def bind():
    engine.PROFILE=PROFILE;engine.read=read;engine.DeployMoE=r.CorrectedMoE
    engine.input_amplitude=r.input_amplitude;engine.deployment_contract=r.deployment_contract
    engine.ALLOWED=ALLOWED;engine.sources=sources;engine.manifest=manifest;engine.evaluate=r.evaluate;engine.fresh_gate=r.fresh_gate
    return read(PROFILE)

def eligibility(fit,hist):
    tm,vm=fit['metrics']['train'],fit['metrics']['val'];tail=hist[-min(20,len(hist)):]
    checks=dict(val_accuracy=vm['accuracy']>=.65,val_macro=vm['balanced_accuracy']>=.60,
                no_zero_train_val=min(tm['recall'])>0 and min(vm['recall'])>0,val_each_recall=min(vm['recall'])>=.2,
                no_persistent_zero=all(sum(h['val']['recall'][k]==0 for h in tail)<=len(tail)/2 for k in range(20)),
                gap=100*(tm['accuracy']-vm['accuracy'])<=15,
                routing_all_slots=all(v==1 for split in (tm,vm) for v in split['routing']['nonzero_fraction']),
                routing_input_dependent=all(max(split['routing']['weight_std'])>1e-4 for split in (tm,vm)))
    return dict(passed=all(checks.values()),checks=checks,selected_epoch=fit['selected_epoch'],gap_pp=100*(tm['accuracy']-vm['accuracy']),test_evaluations=0)

def numeric(a):
    assert not a.out.exists() and a.gpu in ALLOWED.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    slots();cfg=bind();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit)
    gpu_budget.ALLOWED.update(ALLOWED.values());policy=gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    m=manifest(Path(cfg['data']));data=engine.load_fit(Path(cfg['data']),'train',m);a.out.mkdir(parents=True,exist_ok=False)
    checks=[r.fresh_gate(4,rho,data,cfg,'cuda',1) for rho in (0.,.3)]
    assert checks[0]['initial_parameter_sha256']==checks[1]['initial_parameter_sha256']
    save(a.out/'checks.json',dict(passed=True,checks=checks,policy=policy,sources=src,git_commit=commit,
         train_samples=len(data[1]),steps_per_epoch=math.ceil(len(data[1])/16),tail_batch=len(data[1])%16 or 16,
         active_project_slots=slots(),test_arrays_materialized=0,test_evaluations=0),exclusive=True)

def prerequisites():
    cfg=read(PROFILE);manifest(Path(cfg['data']))
    for kind in ('cpu_gate','numeric_gate'):
        proof=read(ROOT/cfg[kind]);assert proof['passed'] and proof['sources']==sources()
        engine.verify_sources(proof['sources'],proof['git_commit'])
    return cfg

def fit(a):
    assert a.depth==4 and a.rho in (0.,.3);slots();cfg=prerequisites();assert a.data==Path(cfg['data'])
    bind();a.resume_parent=None;engine.fit(a)
    f=read(a.out/'fit_result.json');save(a.out/'validation_gate.json',eligibility(f,read(a.out/'history.json')))

def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists()
    cfg=prerequisites();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,config=cfg,command=sys.argv,
         phase='bounded_RESISC20_full_resolution_L4_pair',test_evaluations=0))
    pending=[.3,0.];active=[];complete=[];failures=[]
    while pending or active:
        for job in list(active):
            rc=job['process'].poll()
            if rc is not None:
                job['log'].close();record={k:v for k,v in job.items() if k not in ('process','log')};record['exit_code']=rc
                (complete if rc==0 else failures).append(record);active.remove(job)
        if failures:pending=[]
        for index in (6,2,5):
            if not pending or failures:break
            gpu=ALLOWED[index];live=slots()
            if live['total']>=6 or live['counts'][gpu]>=2 or any(j['gpu_uuid']==gpu for j in active):continue
            gpu_budget.ALLOWED.update(ALLOWED.values())
            try:gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            rho=pending.pop(0);key='L4_rho03' if rho else 'L4_rho0';dest=a.out/key;assert not dest.exists()
            log=(a.out/(key+'.log')).open('x');cmd=[sys.executable,str(HERE/'resisc20.py'),'--phase','fit','--depth','4','--rho',str(rho),'--gpu',gpu,'--data',cfg['data'],'--out',str(dest)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
            active.append(dict(case=key,rho=rho,pid=proc.pid,gpu_uuid=gpu,process=proc,log=log))
        save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending))
        save(a.out/'status.json',dict(state='pair_failed' if failures else 'pair_running' if active or pending else 'pair_complete_awaiting_CPU_audit',
             completed=complete,failures=failures,active=len(active),pending=len(pending),test_evaluations=0))
        if active or pending:time.sleep(5)
    engine.verify_sources(src,commit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['numeric','queue','fit'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),pid=os.getpid(),automatic_retry=False,test_evaluations=0))
        raise
