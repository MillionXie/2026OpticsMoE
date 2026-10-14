"""Explicitly authorized additional budget; resume the sealed full-size E30 pair."""
import argparse, inspect, os, subprocess, sys, textwrap, time
from pathlib import Path
import torch
import resisc20 as s
import resisc20_microbatch as speed
from audit_caltech_repair234 import exited

HERE, ROOT = s.HERE, s.ROOT
PARENT = HERE / 'runs/simulation/resisc20_s17_mainL4_rho0rho03_fullres_microbatch_continue30_gpu256_20261014'
FILES = ('resisc20_continue120.py', 'audit_resisc20_continue120.py', 'RESISC20_CONTINUE120_PROTOCOL_20261014.md')
PARENT_COMMIT = '75be74b96e49e221c6b35a8edd94f98c0bfedac5'
MAX_EPOCHS, ADDITIONAL_SECONDS, PREFIX_EPOCHS, MICRO = 120, 7200, 30, 8

def sources():
    src = speed.sources()
    for name in FILES:
        src[(HERE / name).relative_to(ROOT).as_posix()] = s.sha(HERE / name)
    return src

def profile():
    return dict(s.read(s.PROFILE), maximum_epochs=MAX_EPOCHS,
                maximum_seconds=ADDITIONAL_SECONDS, microbatch=MICRO)

def prerequisites():
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') in ('', *s.ALLOWED.values())
    meta = s.read(PARENT / 'metadata.json'); state = s.read(PARENT / 'status.json')
    assert meta['git_commit'] == PARENT_COMMIT and len(meta['sources']) == 141
    s.engine.verify_sources(meta['sources'], meta['git_commit'])
    assert meta['sources'] == speed.sources() and exited(meta['pid'])
    assert state['active'] == state['pending'] == 0 and not state['failures']
    assert len(state['completed']) == 2 and sorted(j['rho'] for j in state['completed']) == [0., .3]
    assert not s.read(PARENT / 'current_processes.json')['active']
    audit = s.read(PARENT / 'cpu_report/audit.json')
    migration = s.read(PARENT / 'cpu_report/execution_migration_cpu_checks.json')
    receipt = s.read(PARENT / 'cpu_saved_audit_dispatch_completed_20261014.json')
    audit_sha = s.sha(PARENT / 'cpu_report/audit.json')
    assert audit['passed'] and migration['passed'] and receipt['passed']
    assert migration['audit_sha256'] == receipt['audit_sha256'] == audit_sha
    assert audit['model_forward_calls'] == audit['test_arrays_read'] == audit['test_evaluations'] == 0
    assert not (PARENT / 'cpu_report_failure.json').exists()
    assert speed.prerequisites() == MICRO
    s.manifest(Path(profile()['data']))
    records = {}
    for j in state['completed']:
        assert exited(j['pid']) and j['exit_code'] == 0
        arm = PARENT / j['case']; f = s.read(arm / 'fit_result.json')
        hist = s.read(arm / 'history.json'); grad = s.read(arm / 'gradients.json')
        assert f == audit['fits'][j['case']] and f['epochs_completed'] == PREFIX_EPOCHS
        assert s.sha(arm / 'history.json') == f['history_sha256']
        assert s.sha(arm / 'best_checkpoint.pt') == f['checkpoint_sha256']
        ck = torch.load(arm / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        assert ck['epoch'] == len(hist) == len(grad) == PREFIX_EPOCHS
        assert ck['sources'] == meta['sources'] and ck['config'] == f['config']
        assert ck['orders'] == f['orders'] and ck['transforms'] == f['transforms']
        assert ck['scheduler']['last_epoch'] == PREFIX_EPOCHS
        assert len(ck['optimizer']['state']) == 11
        assert all(int(v['step']) == 700 * PREFIX_EPOCHS for v in ck['optimizer']['state'].values())
        assert ck['config']['microbatch'] == MICRO and f['test_evaluations'] == 0
        assert all(torch.isfinite(v).all() for kind in ('model', 'ema') for v in ck[kind].values())
        records[j['case']] = dict(epoch=PREFIX_EPOCHS, last_checkpoint_sha256=s.sha(arm/'last_checkpoint.pt'),
            best_checkpoint_sha256=f['checkpoint_sha256'], history_sha256=f['history_sha256'],
            gradients_sha256=s.sha(arm/'gradients.json'), raw_sha256=s.digest(ck['model']),
            ema_sha256=s.digest(ck['ema']), cumulative_seconds=hist[-1]['seconds'],
            initial_parameter_sha256=f['initial_parameter_sha256'])
    assert len({v['initial_parameter_sha256'] for v in records.values()}) == 1
    return dict(parent=str(PARENT), audit_sha256=audit_sha, records=records,
                maximum_total_epochs=MAX_EPOCHS, additional_fit_seconds_per_arm=ADDITIONAL_SECONDS,
                model_forward_calls=0, test_arrays_read=0, test_evaluations=0)

def bind():
    s.bind(); speed.batch.install()
    original_read = s.read
    def read(path):
        return profile() if Path(path) == s.PROFILE else original_read(path)
    s.engine.read = read; s.engine.sources = sources; s.engine.fresh_gate = speed.gate
    source = textwrap.dedent(inspect.getsource(s.engine.fit))
    replacements = {
        "for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib')":
            "for v in (old_cfg,new_cfg):\n            for key in ('shared_spare_memory_gib','maximum_epochs','maximum_seconds'):v.pop(key)",
        "if offset+time.monotonic()-started>=cfg['maximum_seconds']":
            "if time.monotonic()-started>=cfg['maximum_seconds']",
        "old_memory_gib=3,new_memory_gib=2":
            "old_memory_gib=resume['config']['shared_spare_memory_gib'],new_memory_gib=cfg['shared_spare_memory_gib'],old_microbatch=resume['config']['microbatch'],new_microbatch=cfg['microbatch'],prior_cumulative_seconds=offset,additional_fit_seconds_limit=7200,maximum_total_epochs=120"
    }
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    namespace = dict(s.engine.__dict__)
    exec(compile(source, str(HERE/'resisc20_continue120.py')+':sealed_fit_new_budget_bind', 'exec'), namespace)
    return namespace['fit']

def fit(a):
    assert a.depth == 4 and a.rho in (0., .3)
    assert a.case == ('L4_rho03' if a.rho else 'L4_rho0') and not a.out.exists()
    proof = prerequisites(); s.slots()
    assert a.data == Path(profile()['data'])
    a.resume_parent = PARENT / a.case
    function = bind(); function(a)
    f = s.read(a.out/'fit_result.json')
    f['scope'] = 'Human-authorized same full-size pair continuation E31 to global120 or 7200 additional fit seconds, disclosed extra budget; no new profile/test.'
    f['budget_continuation'] = dict(parent=str(a.resume_parent), prefix_epochs=PREFIX_EPOCHS,
        parent_audit_sha256=proof['audit_sha256'], parent_snapshot=proof['records'][a.case],
        maximum_total_epochs=MAX_EPOCHS, additional_fit_seconds_limit=ADDITIONAL_SECONDS,
        additional_fit_seconds=f['seconds']-proof['records'][a.case]['cumulative_seconds'],
        optimizer_ema_scheduler_reset=False, no_completed_epochs_replayed=True)
    s.save(a.out/'fit_result.json', f)
    s.save(a.out/'validation_gate.json', s.eligibility(f, s.read(a.out/'history.json')))

def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not a.out.exists()
    proof = prerequisites(); src = sources()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    s.engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False)
    s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,
        config=profile(),command=sys.argv,parent_proof=proof,resume_parent=str(PARENT),
        phase='human_authorized_additional_budget_E31_to_120',test_evaluations=0))
    pending=[.3,0.]; active=[]; complete=[]; failures=[]
    while pending or active:
        for j in list(active):
            rc=j['process'].poll()
            if rc is not None:
                j['log'].close(); record={k:v for k,v in j.items() if k not in ('process','log')}
                record['exit_code']=rc; (complete if rc==0 else failures).append(record); active.remove(j)
        if failures: pending=[]
        for index in (6,2,5):
            if not pending or failures: break
            gpu=s.ALLOWED[index]; slots=s.slots()
            if slots['total']>=6 or slots['counts'][gpu]>=2 or any(j['gpu_uuid']==gpu for j in active): continue
            s.gpu_budget.ALLOWED.update(s.ALLOWED.values())
            try: s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError: continue
            rho=pending.pop(0); case='L4_rho03' if rho else 'L4_rho0'; dest=a.out/case
            assert not dest.exists(); log=(a.out/(case+'.log')).open('x')
            cmd=[sys.executable,str(HERE/'resisc20_continue120.py'),'--phase','fit','--depth','4',
                 '--rho',str(rho),'--case',case,'--gpu',gpu,'--data',profile()['data'],'--out',str(dest)]
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
            active.append(dict(case=case,rho=rho,pid=child.pid,gpu_uuid=gpu,process=child,log=log))
        s.save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending))
        s.save(a.out/'status.json',dict(state='continue120_failed' if failures else 'continue120_running' if active or pending else 'continue120_complete_awaiting_CPU_audit',
            completed=complete,failures=failures,active=len(active),pending=len(pending),test_evaluations=0))
        if active or pending: time.sleep(5)
    s.engine.verify_sources(src,commit)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['queue','fit'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--depth',type=int)
    p.add_argument('--rho',type=float);p.add_argument('--case');p.add_argument('--gpu');p.add_argument('--data',type=Path)
    a=p.parse_args()
    try: globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists(): s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations=0))
        raise
