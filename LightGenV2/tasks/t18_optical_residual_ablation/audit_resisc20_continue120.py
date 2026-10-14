"""Saved CPU review of the explicitly extended budget and all sealed prefixes."""
import argparse, inspect, os, textwrap
from pathlib import Path
import torch
import resisc20 as s
import resisc20_continue120 as continuation
import audit_resisc20 as audit
from audit_caltech_repair234 import exited

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not a.out.exists()
    assert not (a.base/'cpu_report_failure.json').exists()
    proof=continuation.prerequisites(); meta=s.read(a.base/'metadata.json'); state=s.read(a.base/'status.json')
    assert meta['parent_proof']==proof and meta['resume_parent']==str(continuation.PARENT)
    assert meta['sources']==continuation.sources() and len(meta['sources'])==144
    s.engine.verify_sources(meta['sources'],meta['git_commit'])
    assert exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures']
    assert len(state['completed'])==2
    checks=[]
    for j in state['completed']:
        assert exited(j['pid']) and j['exit_code']==0
        key=j['case']; parent=continuation.PARENT/key; arm=a.base/key; record=proof['records'][key]
        old=torch.load(parent/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        current=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        f=s.read(arm/'fit_result.json'); r=s.read(arm/'resume_receipt.json')
        hist=s.read(arm/'history.json'); grad=s.read(arm/'gradients.json')
        assert s.sha(parent/'last_checkpoint.pt')==record['last_checkpoint_sha256']
        assert r['parent_epoch']==30 and r['first_new_epoch']==31
        assert not r['optimizer_reset'] and not r['ema_reset'] and not r['scheduler_reset']
        assert r['last_checkpoint_sha256']==record['last_checkpoint_sha256']
        assert r['best_checkpoint_sha256']==record['best_checkpoint_sha256']
        assert r['prefix_history_sha256']==record['history_sha256']
        assert r['restored_state_sha256']=={k:s.digest(old[k]) for k in ('model','ema')}
        assert r['old_microbatch']==r['new_microbatch']==8
        assert r['prior_cumulative_seconds']==record['cumulative_seconds']
        assert r['additional_fit_seconds_limit']==7200 and r['maximum_total_epochs']==120
        assert hist[:30]==s.read(parent/'history.json') and grad[:30]==s.read(parent/'gradients.json')
        assert current['orders'][:30]==old['orders'] and current['transforms'][:30]==old['transforms']
        assert f['budget_continuation']['parent_snapshot']==record
        assert f['budget_continuation']['parent_audit_sha256']==proof['audit_sha256']
        assert f['budget_continuation']['additional_fit_seconds']==f['seconds']-record['cumulative_seconds']
        assert 30<=f['epochs_completed']<=120
        # The original engine finishes complete epochs and selected endpoint evaluation;
        # no new epoch may start after the additional-time deadline.
        assert all(h['seconds']-record['cumulative_seconds']<7200 for h in hist[30:-1])
        cfg=dict(f['config']); before=dict(old['config'])
        assert cfg['maximum_epochs']==120 and cfg['maximum_seconds']==7200 and cfg['microbatch']==8
        for v in (cfg,before):
            for k in ('maximum_epochs','maximum_seconds'): v.pop(k)
        assert cfg==before
        checks.append(dict(case=key,prefix_epochs=30,first_new_epoch=31,
            preserved_prefix_history_gradients_order_affine_mixup=True,raw_ema_restored_exact=True,
            optimizer_ema_scheduler_reset=False,parent_last_sha256=record['last_checkpoint_sha256']))
    # Reuse the independently regenerated all-epoch science checks. Only its fixed
    # old 30-epoch ceiling is made configuration-aware; the gates are unchanged.
    source=textwrap.dedent(inspect.getsource(audit.verify))
    old="f['epochs_completed']<=30"; assert source.count(old)==1
    source=source.replace(old,"f['epochs_completed']<=cfg['maximum_epochs']")
    namespace=dict(audit.__dict__);exec(compile(source,str(Path(__file__))+':saved_verify_120','exec'),namespace)
    original=audit.verify
    try:
        audit.verify=namespace['verify']; audit.audit(a)
    finally: audit.verify=original
    s.save(a.out/'budget_continuation_cpu_checks.json',dict(passed=True,checks=checks,
        parent_audit_sha256=proof['audit_sha256'],audit_sha256=s.sha(a.out/'audit.json'),
        maximum_total_epochs=120,additional_fit_seconds_per_arm=7200,
        model_forward_calls=0,test_arrays_read=0,test_evaluations=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','out','data'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        marker=a.base/'cpu_report_failure.json'
        if not marker.exists():s.save(marker,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
