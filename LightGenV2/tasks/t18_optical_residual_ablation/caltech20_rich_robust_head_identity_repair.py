"""Source-dictionary type repair; fresh CPU fits on immutable cached features."""
import ast,inspect
import caltech20_rich_robust_head as old
for n,v in vars(old).items():
    if not n.startswith('__') and not inspect.isfunction(v):globals()[n]=v
EXTRA=('caltech20_rich_robust_head_identity_repair.py','check_caltech20_rich_robust_head_identity_repair.py','audit_caltech20_rich_robust_head_identity_repair.py','CALTECH20_RICH_ROBUST_HEAD_IDENTITY_REPAIR_PROTOCOL_20261018.md')
FILES=(EXTRA[0],old.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
controls=robust.controls

def config():
    c=old.config();c.update(run=c['run']+'_source_identity_repair',repair='Compare source SHA string dictionaries as dictionaries; retain old failed audit; fresh CPU fits, reuse immutable features, no optical forward. Record ALL child raw exitcodes before any audit.');return c

OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_robust_head_identity_repair_cpu_20261018'

def sources():
    z=old.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

for node in ast.parse(inspect.getsource(old)).body:
    if isinstance(node,ast.FunctionDef) and node.name not in ('config','sources','extract','campaign'):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())

def reuse_preflight():
    failure=old.OUT/'L2_rho03/fit_audit_failure.json';assert s.read(failure)['error']=='''AttributeError("'str' object has no attribute 'detach'")'''
    assert not (old.OUT/'L2_rho03/science/audit.json').exists() and not (old.OUT/'L2_rho03/audit_completed.json').exists()
    ex=s.read(old.OUT/'campaign_exit.json');assert ex['returncode']==1 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
    result=dict(failure_sha256=s.sha(failure),failure_path=str(failure),cases={})
    for d in (2,4,6):
        folder=old.OUT/f'L{d}_rho03';meta=s.read(folder/'metadata.json');assert meta['sources']==old.sources();engine.verify_sources(meta['sources'],meta['git_commit'])
        ex=s.read(folder/'extract_exit.json');assert ex['returncode']==0 and ex['actual_exit'];proof=s.read(folder/'extract_result.json');assert proof['passed']
        assert not (folder/'test_claim.json').exists() and not (folder/'test_result.json').exists()
        for role in ('train','val'):assert proof['features_sha256'][role]==s.sha(folder/(role+'_features.npz'))
        assert proof['test_features_sha256']==s.sha(folder/'test_features_reused.npz')
        pid=s.read(folder/'fit_claim.json')['pid'];assert not Path('/proc',str(pid),'cmdline').exists()
        result['cases'][str(d)]=dict(extract_source_sha256=s.sha(folder/'metadata.json'),features=proof['features_sha256'],test_features_sha256=proof['test_features_sha256'],original_fit_raw_exit_recorded=(folder/'producer_exit.json').exists())
    return result

def reuse(d):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';evidence=reuse_preflight();src=old.OUT/f'L{d}_rho03';out=OUT/f'L{d}_rho03';assert not out.exists();claim(out.with_name(out.name+'.reuse_claim.json'),'reuse_passed_frozen_feature_bytes_no_forward');out.mkdir()
    for name in ('train_features.npz','val_features.npz','test_features_reused.npz'):
        before=s.sha(src/name);shutil.copyfile(src/name,out/name);assert s.sha(out/name)==before==s.sha(src/name)
    meta=s.read(src/'metadata.json');meta.update(identity());meta.update(pid=os.getpid(),gpu_uuid=None,config=config(),policy=dict(reused_feature_bytes=True,new_GPU_calls=0),reused_origin_metadata_sha256=s.sha(src/'metadata.json'),preserved_failure_sha256=evidence['failure_sha256'])
    s.save(out/'metadata.json',meta,exclusive=True);proof=s.read(src/'extract_result.json');proof.update(reused_immutable_features=True);s.save(out/'extract_result.json',proof,exclusive=True)

def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();assert s.read(CPU/'source_identity_checks.json')['passed'];origin=reuse_preflight();claim(OUT.with_name(OUT.name+'.claim.json'),'fresh_CPU_fits_source_identity_repair');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),origin=origin,frozen_controls=controls()),exclusive=True)
    for phase,receipt in [('reuse','extract_exit.json'),('fit','producer_exit.json')]:
        jobs=[]
        for d in (2,4,6):
            log=(OUT/f'L{d}.{phase}.log').open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase',phase,'--depth',str(d)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT);jobs.append((d,p,log))
        exits=[]
        for d,p,log in jobs:
            rc=p.wait();log.close();ex=dict(pid=p.pid,returncode=rc,actual_exit=True);s.save(OUT/f'L{d}_rho03'/receipt,ex,exclusive=True);exits.append(ex)
        assert all(z['returncode']==0 for z in exits),'All original exitcodes saved; preserve failure/no retry'
    for d in (2,4,6):
        out=OUT/f'L{d}_rho03';claim(out/'audit_claim.json','independent_source_dictionary_and_numeric_audit')
        with (out/'audit.log').open('x') as log:rc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--depth',str(d),'--phase','fit'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0 and s.read(out/'science/audit.json')['passed'];s.save(out/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/'science/audit.json')),exclusive=True)
        with (out/'test.log').open('x') as log:p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','test','--depth',str(d)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
        s.save(out/'test_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0;claim(out/'test_audit_claim.json','own_saved_TEST_math')
        with (out/'test_audit.log').open('x') as log:rc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--depth',str(d),'--phase','test'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0 and s.read(out/'test_science/audit.json')['passed'];s.save(out/'test_audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(out/'test_science/audit.json')),exclusive=True)
    assert reuse_preflight()['failure_sha256']==origin['failure_sha256'];s.save(OUT/'status.json',dict(state='postrobustness_samehead_complete',active=[],pending=[],failures=[],original_failed_audit_preserved=True),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['reuse','fit','test','campaign'],required=True);p.add_argument('--depth',type=int);a=p.parse_args()
    try:campaign() if a.phase=='campaign' else globals()[a.phase](a.depth)
    except BaseException as e:
        out=OUT/f'L{a.depth}_rho03' if a.depth else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
