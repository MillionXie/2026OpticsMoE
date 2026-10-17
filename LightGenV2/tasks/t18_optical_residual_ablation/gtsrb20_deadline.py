"""Explicit new data binding to the frozen, verified full100 hybrid engine.

Only process-local bindings change. No old source, run, config or artifact is
edited. Generic full100 fit/evaluate/numeric kernels remain byte-identical.
"""
import argparse,os,subprocess,sys,time
from pathlib import Path
import gtsrb20_deadline_data as dataset
import caltech20_alt246_richsplit as b
HERE,ROOT,s,engine=dataset.HERE,dataset.ROOT,dataset.s,dataset.engine
RUN='gtsrb20_track_current246_s17_equal100_20261017'
OUT=HERE/'runs/simulation'/RUN
CPU=HERE/'runs/smoke/gtsrb20_track_current246_cpu_20261017'
sources,identity=dataset.sources,dataset.identity
CASES={key:(depth,rho) for key,depth,rho in [('L6_rho03',6,.3),('L4_rho03',4,.3),('L6_rho0',6,0.),('L4_rho0',4,0.),('L2_rho03',2,.3),('L2_rho0',2,0.)]}
def config():
    p=dataset.profile();m=dataset.manifest();directory=Path(p['data_directory']);p.update(classes=m['classes'],data=str(directory/'gtsrb20_track_split.npz'),cache_sha256=m['cache_sha256'],manifest_sha256=s.sha(directory/'data_manifest.json'),training_samples=len(m['split_ids']['train']),steps_per_epoch=250,tail_samples=16)
    assert p['training_samples']==4000 and p['batch_size']==16 and p['microbatch']==4 and p['maximum_epochs']==100
    return p
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def launch_child(root,phase,key,gpu,out):
    log=(root/(phase+'_'+key+'.log')).open('x');cmd=[sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--case',key,'--gpu',gpu,'--out',str(out)];proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
    return dict(case=key,gpu=gpu,pid=proc.pid,process=proc,log=log,out=str(out))
def queue(root,phase):
    pending=list(CASES);active=[];done=[];failures=[]
    while pending or active:
        for job in list(active):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();rec={k:v for k,v in job.items() if k not in ('process','log')};rec.update(returncode=rc,actual_exit=True);s.save(Path(job['out']).with_name(Path(job['out']).name+'.'+phase+'_exit.json'),rec,exclusive=True);(failures if rc else done).append(rec);active.remove(job)
        if failures:pending=[]
        for card in (0,4,2,5):
            if not pending or failures:break
            gpu=b.GPU[card];live=b.slots()
            if live['total']>=4 or live['counts'][gpu] or any(j['gpu']==gpu for j in active):continue
            s.gpu_budget.ALLOWED.update(b.GPU.values())
            try:s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            key=pending.pop(0);out=root/key if phase=='fit' else root/'once_tests'/key;active.append(launch_child(root,phase,key,gpu,out))
        s.save(root/'status.json',dict(state=phase+'_failed' if failures else phase+'_running' if pending or active else phase+'_complete',active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending,complete=done,failures=failures))
        if pending or active:time.sleep(5)
    assert not failures,failures
    return done
def audit_child(root,phase):
    claim(root/(phase+'_audit_claim.json'),'GTSRB20_own_independent_'+phase);folder='science' if phase=='fit' else 'saved_test_audit'
    with (root/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_gtsrb20_deadline.py'),'--base',str(root),'--out',str(root/folder),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(root/folder/'audit.json')['passed'],'Own GTSRB audit failed; preserve/no retry';s.save(root/('audit_completed.json' if phase=='fit' else 'test_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(root/folder/'audit.json')),exclusive=True)

# Bind all dataset/source/output entry points before calling generic functions.
# Parameter graph, forward, loss, epoch generator, numeric gates, optimizer,
# full100 fit and TEST kernels are the existing protected implementation.
b.dataset=dataset;b.RUN=RUN;b.CPU=CPU;b.CASES=CASES;b.config=config;b.sources=sources;b.identity=identity;b.claim=claim;b.launch_child=launch_child;b.run_queue=queue;b.audit_child=audit_child

def campaign(a):
    assert a.out==OUT;b.campaign(a);state=s.read(OUT/'status.json');assert state['state']=='richsplit_equal100_science_and_frozen_tests_complete';state['state']='gtsrb20_equal100_own_science_and_frozen_tests_complete';s.save(OUT/'status.json',state)
def supervise(a):
    assert a.out==OUT and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists();b.cpu_ready();assert b.slots()['total']==0;claim(OUT.with_name(OUT.name+'.supervisor_claim.json'),'GTSRB20_explicit_single_supervisor')
    with OUT.with_name(OUT.name+'.campaign.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--out',str(OUT)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);s.save(OUT.with_name(OUT.name+'.started.json'),dict(supervisor_pid=os.getpid(),campaign_pid=proc.pid,command=proc.args),exclusive=True);rc=proc.wait()
    s.save(OUT/'campaign_exit.json',dict(pid=proc.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Preserve failure/no retry'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign','supervise'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');a=p.parse_args()
    try:(campaign if a.phase=='campaign' else supervise if a.phase=='supervise' else getattr(b,a.phase))(a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise
