"""Fresh residual246 after alpha co-adaptation diagnosis; complete namespaces."""
import ast,inspect,types
import caltech20_rich_alpha40_rebalance as core
import caltech20_rich_alpha40_namespace_repair as legacy
import caltech20_rich_alpha40_saved_science_repair as review
for n,v in vars(core).items():
    if not n.startswith('__'):globals()[n]=v
Alpha40MoE=legacy.Alpha40MoE
FILES=('caltech20_rich_alpha40_fresh246.py','caltech20_rich_alpha40_fresh246_profile.json','check_caltech20_rich_alpha40_fresh246.py','audit_caltech20_rich_alpha40_fresh246.py','CALTECH20_RICH_ALPHA40_FRESH246_PROTOCOL_20261018.md')
def config():
    c=core.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_alpha40_fresh246_cpu_20261018'
def sources():
    z=review.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for node in ast.parse(inspect.getsource(core)).body:
    if isinstance(node,ast.FunctionDef) and node.name not in ('config','sources','ready','campaign'):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())
def ready():
    p=s.read(CPU/'checks.json');d=s.read(diagnosis.OUT/'diagnosis.json');assert p['passed'] and p['sources']==sources() and p['all_kernel_and_audit_global_symbols_resolved'] and p['device_anchor_proof']['passed']
    assert d['passed'] and d['sources']==diagnosis.sources();identity();old.controls();return p
def legacy_release():
    meta=s.read(legacy.OUT/'metadata.json');assert meta['sources']==legacy.sources();engine.verify_sources(meta['sources'],meta['git_commit'])
    state=s.read(legacy.OUT/'status.json')
    if state['state']!='failed':return False
    assert not state['pending'] and not (legacy.OUT/'L4_fresh').exists()
    return True
def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();started=time.monotonic()
    # Original controller must first clear its queued L4 case; no duplicate fresh4.
    while not legacy_release():
        assert time.monotonic()-started<7200;time.sleep(2)
    claim(OUT.with_name(OUT.name+'.claim.json'),'fresh246_after_alpha45_coadaptation_review');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),frozen_controls=old.controls()),exclusive=True)
    pending=list(config()['cases']);jobs=[];complete=[];failures=[];reviews=[]
    while pending or jobs:
        slots=resources.slots();used={q['gpu'] for q in jobs}
        for index in config()['gpu_order']:
            gpu=resources.GPU[index]
            if not pending or gpu in used or slots['counts'][gpu]>0:continue
            key=pending.pop(0);s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            log=(OUT/(key+'.log')).open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit','--key',key,'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(key=key,gpu=gpu,pid=p.pid,process=p,log=log));used.add(gpu)
        for job in list(jobs):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();ex={k:v for k,v in job.items() if k not in ('process','log')};ex.update(returncode=rc,actual_exit=True);s.save(OUT/(job['key']+'.producer_exit.json'),ex,exclusive=True);jobs.remove(job)
            if rc:failures.append(ex);pending=[];continue
            f=OUT/job['key'];claim(f/'audit_claim.json','fresh246_independent_saved_science')
            with (f/'audit.log').open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--key',job['key']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
            if arc:failures.append(dict(key=job['key'],audit_returncode=arc));pending=[];continue
            assert s.read(f/'science/audit.json')['passed'];s.save(f/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(f/'science/audit.json')),exclusive=True);complete.append(ex)
        # Independently verify already failed legacy audits in a NEW destination.
        # Never execute the old failed audit script or mutate old receipts.
        for key in ('L2_rebalance','L4_rebalance','L6_rebalance'):
            failure=legacy.OUT/key/'audit_failure.json'
            if key in reviews or not failure.exists():continue
            log=(OUT/(key+'.legacy_review.log')).open('x');p=subprocess.run([os.sys.executable,str(HERE/review.FILE),'--key',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);log.close();reviews.append(key)
            s.save(OUT/(key+'.legacy_review_exit.json'),dict(returncode=p.returncode,actual_exit=True,review_root=str(review.OUT/key)),exclusive=True)
            if p.returncode:failures.append(dict(key=key,legacy_verification_returncode=p.returncode))
        s.save(OUT/'status.json',dict(state='failed' if failures else 'alpha40_fresh_running' if jobs or pending else 'all_alpha40_fresh_reviewed',active=[{k:v for k,v in q.items() if k not in ('process','log')} for q in jobs],pending=pending,complete=complete,failures=failures,legacy_independent_reviews=reviews))
        if pending or jobs:time.sleep(2)
    assert not failures
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        folder=OUT/a.key if a.key else OUT;mark=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
