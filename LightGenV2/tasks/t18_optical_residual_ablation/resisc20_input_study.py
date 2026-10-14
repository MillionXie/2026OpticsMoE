"""Authorized three fresh full-size input-detail controls; sealed fit body reused."""
import argparse,os,subprocess,sys,time
from pathlib import Path
import resisc20 as s
import resisc20_microbatch as speed
import resisc20_continue120 as prior
from resisc20_input_encoder import encode,contract

HERE,ROOT=s.HERE,s.ROOT
PROFILE=HERE/'resisc20_input_profile.json'
FILES=('resisc20_input_encoder.py','resisc20_input_study.py','resisc20_input_profile.json',
       'check_resisc20_input.py','audit_resisc20_input.py','export_resisc20_input.py','RESISC20_INPUT_PROTOCOL_20261014.md')
CASES={'L4_tile50_rho03':(50,.3),'L4_tile100_rho03':(100,.3),'L4_tile100_rho0':(100,0.)}

def sources():
    src=prior.sources()
    for name in FILES:src[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return src

def config(case):
    cfg=s.read(PROFILE);assert cfg.pop('cases')=={k:list(v) for k,v in CASES.items()}
    tile,rho=CASES[case];return dict(cfg,input_tile=tile,study_case=case)

def bind(case):
    cfg=config(case);tile,rho=CASES[case];s.bind();speed.batch.install()
    original=s.read
    def read(path):return config(case) if Path(path)==PROFILE else original(path)
    s.engine.PROFILE=PROFILE;s.engine.read=read;s.engine.sources=sources
    s.engine.input_amplitude=lambda images,theta=None:encode(images,theta,tile)
    s.engine.deployment_contract=lambda depth,value:contract(depth,value,tile)
    s.engine.fresh_gate=speed.gate
    return cfg

def prerequisites():
    s.manifest(Path(config(next(iter(CASES)))['data']))
    for name in ('resisc20_input_cpu_native_20261014','resisc20_input_gpu6_numeric_20261014'):
        proof=s.read(HERE/'runs/smoke'/name/'checks.json')
        assert proof['passed'] and proof['sources']==sources()
        s.engine.verify_sources(proof['sources'],proof['git_commit'])
    return config(next(iter(CASES)))

def numeric(a):
    assert a.gpu in s.ALLOWED.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and not a.out.exists()
    cpu=s.read(HERE/'runs/smoke/resisc20_input_cpu_native_20261014/checks.json')
    assert cpu['passed'] and cpu['sources']==sources();s.engine.verify_sources(cpu['sources'],cpu['git_commit'])
    s.slots();s.gpu_budget.ALLOWED.update(s.ALLOWED.values());policy=s.gpu_budget.configure(s.torch,a.gpu,shared=True,memory_gib=2)
    s.torch.set_num_threads(4);cfg=config(next(iter(CASES)));m=s.manifest(Path(cfg['data']))
    data=s.engine.load_fit(Path(cfg['data']),'train',m);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);checks=[]
    for case,(tile,rho) in CASES.items():
        cfg=bind(case);v=speed.gate(4,rho,data,cfg,'cuda',8);v.update(case=case,input_tile=tile);checks.append(v)
    assert len({v['initial_parameter_sha256'] for v in checks})==1
    s.save(a.out/'checks.json',dict(passed=True,checks=checks,policy=policy,sources=src,git_commit=commit,
        same_phase_initialization=True,train_samples=11198,steps_per_epoch=700,tail_batch=14,
        test_arrays_read=0,test_evaluations=0,optimizer_steps=0),exclusive=True)

def fit(a):
    assert a.case in CASES and a.depth==4 and a.rho==CASES[a.case][1]
    prerequisites();s.slots();cfg=bind(a.case);assert a.data==Path(cfg['data'])
    a.resume_parent=None;s.engine.fit(a)
    f=s.read(a.out/'fit_result.json');f['scope']='Human-authorized fresh full-size three-arm input-detail control; same data/init/order/augmentation; no test or architecture reduction.'
    s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',s.eligibility(f,s.read(a.out/'history.json')))

def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists()
    prerequisites();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'metadata.json',dict(pid=os.getpid(),git_commit=commit,sources=src,
        command=sys.argv,config=s.read(PROFILE),phase='authorized_full_size_input_detail_three_fresh_controls',test_evaluations=0))
    pending=list(CASES);active=[];complete=[];failures=[]
    while pending or active:
        for j in list(active):
            rc=j['process'].poll()
            if rc is not None:
                j['log'].close();rec={k:v for k,v in j.items() if k not in ('process','log')};rec['exit_code']=rc
                (complete if rc==0 else failures).append(rec);active.remove(j)
        if failures:pending=[]
        for index in (6,2,5):
            if not pending or failures:break
            gpu=s.ALLOWED[index];slots=s.slots()
            if slots['total']>=6 or slots['counts'][gpu]>=2 or any(j['gpu_uuid']==gpu for j in active):continue
            s.gpu_budget.ALLOWED.update(s.ALLOWED.values())
            try:s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            case=pending.pop(0);tile,rho=CASES[case];out=a.out/case;assert not out.exists();log=(a.out/(case+'.log')).open('x')
            cmd=[sys.executable,str(HERE/'resisc20_input_study.py'),'--phase','fit','--case',case,'--depth','4','--rho',str(rho),
                 '--gpu',gpu,'--data',config(case)['data'],'--out',str(out)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
            active.append(dict(case=case,input_tile=tile,rho=rho,pid=proc.pid,gpu_uuid=gpu,process=proc,log=log))
        s.save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending))
        s.save(a.out/'status.json',dict(state='input_study_failed' if failures else 'input_study_running' if active or pending else 'input_study_complete_awaiting_CPU_audit',
            completed=complete,failures=failures,active=len(active),pending=len(pending),test_evaluations=0))
        if active or pending:time.sleep(5)
    s.engine.verify_sources(src,commit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['numeric','fit','queue'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');p.add_argument('--data',type=Path);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
        raise
