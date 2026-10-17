"""Resource-only deep baseline scheduler; immutable216-file learning kernel reused."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from functools import lru_cache
import caltech20_original_equal60 as m
import caltech20_original_balancedKD as kd
FILES=('caltech20_original_deep_resource.py','check_caltech20_original_deep_resource.py','CALTECH20_ORIGINAL_DEEP_RESOURCE_PROTOCOL_20261017.md')
CPU=m.HERE/'runs/smoke/caltech20_original_deep_resource_guard_cpu_20261017'
PILOT=m.HERE/'runs/simulation/caltech20_original_fixed_current246_s17_equal60_pilot_20261017'
TUNE=m.HERE/'runs/simulation/caltech20_original_equal60_residual_balancedKD20_pilot_20261017'

def sources():
    src={**m.sources(),**kd.sources()}
    for n in FILES:src[(m.HERE/n).relative_to(m.ROOT).as_posix()]=m.s.sha(m.HERE/n)
    return src
def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=m.ROOT,text=True).strip();m.engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)
def claim(path,phase):m.s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def dead(pid,token):
    path=Path('/proc',str(pid),'cmdline');return not path.exists() or token.encode() not in path.read_bytes()

def guard(live,authorized):
    allowed=set(m.GPU.values());seen=set();assert live['total']==len(live['jobs'])<=4
    for j in live['jobs']:
        pair=(j['pid'],j['gpu_uuid']);assert pair in authorized and j['gpu_uuid'] in allowed and j['gpu_uuid'] not in seen,(j,authorized);seen.add(j['gpu_uuid'])
    assert all(c<=1 for c in live['counts'].values()) and live['counts'][m.frozen.b.FORBIDDEN]==0

def admit():
    assert m.GROUP=='deep' and m.DEPTHS==(4,6) and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    m.cpu_ready();proof=m.s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();m.engine.verify_sources(proof['sources'],proof['git_commit'])
    state=m.s.read(PILOT/'status.json');assert state['state']=='equal60_science_and_reviewed_tests_complete' and not state['active'] and not state['pending'] and not state['failures']
    meta=m.s.read(PILOT/'metadata.json');assert meta['sources']==m.sources();m.engine.verify_sources(meta['sources'],meta['git_commit'])
    cfg=dict(m.config());cfg['group']='pilot';assert meta['config']==cfg
    for receipt,folder in [('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')]:
        rec=m.s.read(PILOT/receipt);assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==m.s.sha(PILOT/folder/'audit.json') and m.s.read(PILOT/folder/'audit.json')['passed']
    assert m.s.read(PILOT/'campaign_exit.json')['returncode']==0
    started=m.s.read(PILOT.with_name(PILOT.name+'.started.json'));assert all(dead(started[k],str(PILOT)) for k in ('supervisor_pid','campaign_pid'))
    for key in ('L2_rho03','L2_rho0'):
        f=m.s.read(PILOT/key/'fit_result.json');assert f['epochs_completed']==60 and f['steps']==5160 and f['best_sha256']==m.s.sha(PILOT/key/'best.pt') and m.s.read(PILOT/'once_tests'/key/'result.json')['test_evaluations']==1

@lru_cache(None)
def verified_external_metadata(metadata_sha):
    assert m.s.sha(TUNE/'metadata.json')==metadata_sha;meta=m.s.read(TUNE/'metadata.json');m.engine.verify_sources(meta['sources'],meta['git_commit']);return meta

def external():
    if not (TUNE/'status.json').exists():return set()
    meta=verified_external_metadata(m.s.sha(TUNE/'metadata.json'));assert meta['sources']==kd.sources();cfg=dict(kd.config());cfg['group']='pilot';assert meta['config']==cfg
    assert not any((TUNE/n).exists() for n in ('failure.json','fit_audit_failure.json','test_audit_failure.json'))
    authorized=set()
    for j in m.s.read(TUNE/'status.json')['active']:
        path=Path('/proc',str(j['pid']),'cmdline')
        if not path.exists():continue
        assert j['key']=='L2_rho03' and j['gpu']==m.GPU[0] and str(TUNE).encode() in path.read_bytes() and b'caltech20_original_balancedKD.py' in path.read_bytes();authorized.add((j['pid'],j['gpu']))
    assert len(authorized)<=1;return authorized

def resource_audit(out):
    # A supplementary resource/provenance receipt, never claimed as part of216 kernel SCI.
    meta=m.s.read(out/'metadata.json');assert meta['resource_driver_identity']['sources']==sources() and meta['sources']==m.sources();events=m.s.read(out/'resource_events.json');checks=[]
    for e in events:guard(e['live'],{tuple(p) for p in e['authorized']});checks.append(e['live']['total'])
    state=m.s.read(out/'status.json');assert not state['active'] and not state['pending'] and not state['failures'] and len(state['complete'])==4
    for job in state['complete']:assert job['returncode']==0 and job['actual_exit'] and dead(job['pid'],job['out'])
    rec=m.s.read(out/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==m.s.sha(out/'science/audit.json') and m.s.read(out/'science/audit.json')['passed']
    m.s.save(out/'resource_driver_audit.json',dict(identity(),passed=True,kernel_source_count=len(m.sources()),kernel_science_sha256=rec['audit_sha256'],CPU_guard_sha256=m.s.sha(CPU/'checks.json'),resource_event_count=len(events),peak_observed_T18_GPU_children=max(checks,default=0),all_actual_producer_exits_zero=True,learning_parameters_changed=False,new_model_objects=0,new_forward_calls=0,new_TEST_calls=0),exclusive=True)

def campaign(out):
    assert out==m.OUT and not out.exists();admit();guard(m.slots(),external());claim(out.with_name(out.name+'.campaign_claim.json'),'same_kernel_deep60_with_exact_known_parallel_residual');out.mkdir();m.s.save(out/'metadata.json',dict(m.identity(),pid=os.getpid(),config=m.config(),cases=m.CASES,resource_driver_identity=identity(),kernel_source_count=len(m.sources()),resource_driver_separately_registered=True),exclusive=True)
    pending=list(m.CASES);active=[];done=[];failed=[];events=[]
    while pending or active:
        for job in list(active):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();rec={k:v for k,v in job.items() if k not in ('process','log')};rec.update(returncode=rc,actual_exit=True);m.s.save(Path(job['out']).with_name(job['case']+'.fit_exit.json'),rec,exclusive=True);(failed if rc else done).append(rec);active.remove(job)
        if failed:pending=[]
        for gpu in m.GPU.values():
            authorized=external()|{(j['pid'],j['gpu']) for j in active};live=m.slots();guard(live,authorized);events.append(dict(live=live,authorized=sorted(authorized),pending=list(pending),monotonic=time.monotonic()))
            if not pending or failed or live['total']>=4 or live['counts'][gpu] or any(j['gpu']==gpu for j in active):continue
            m.s.gpu_budget.ALLOWED.update(m.GPU.values())
            try:m.s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            key=pending.pop(0);folder=out/key;log=(out/('fit_'+key+'.log')).open('x');proc=subprocess.Popen([sys.executable,str(m.HERE/'caltech20_original_equal60.py'),'--phase','fit','--case',key,'--gpu',gpu,'--out',str(folder)],cwd=m.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);active.append(dict(case=key,gpu=gpu,pid=proc.pid,out=str(folder),process=proc,log=log))
        m.s.save(out/'resource_events.json',events);m.s.save(out/'status.json',dict(state='fit_failed' if failed else 'fit_running' if active or pending else 'fit_complete',active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending,complete=done,failures=failed))
        if active or pending:time.sleep(3)
    assert not failed and len(done)==4,failed;m.claim(out/'fit_audit_claim.json','own_saved_math_unchanged216_kernel')
    with (out/'fit_audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(m.HERE/'audit_caltech20_original_equal60.py'),'--phase','fit','--base',str(out),'--out',str(out/'science')],cwd=m.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and m.s.read(out/'science/audit.json')['passed'];m.s.save(out/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=m.s.sha(out/'science/audit.json')),exclusive=True);resource_audit(out);m.s.save(out/'status.json',dict(state='equal60_science_complete_awaiting_validation_review_no_TEST',active=[],pending=[],failures=[],complete=done,controls_frozen_at60=True))

def supervise(out):
    assert out==m.OUT and not out.exists();admit();guard(m.slots(),external());claim(out.with_name(out.name+'.supervisor_claim.json'),'one_resource_supervisor_no_retry')
    with out.with_name(out.name+'.campaign.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(out)],cwd=m.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);m.s.save(out.with_name(out.name+'.started.json'),dict(supervisor_pid=os.getpid(),campaign_pid=p.pid,resource_driver_identity=identity()),exclusive=True);rc=p.wait()
    m.s.save(out/'campaign_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['supervise','campaign'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:globals()[a.phase](a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
