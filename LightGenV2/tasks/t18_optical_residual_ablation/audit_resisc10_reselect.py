"""Independent saved original ten-output auditor bound to new human class identity."""
import argparse,os
from pathlib import Path
import resisc10_reselect as study
import audit_resisc10 as sealed

def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.base.name==study.RUN
    assert not a.out.exists() and not (a.base/'cpu_report_failure.json').exists()
    meta=study.s.read(a.base/'metadata.json');assert meta['sources']==study.sources()
    assert study.exited(meta['pid'])
    assert study.s.read(a.base/'audit_claim.json')['sources']==study.sources()
    sealed.audit(a)
    v=study.s.read(a.out/'audit.json')
    assert v['passed'] and v['common_pair_epochs']==100
    for key,f in v['fits'].items():
        assert key in study.CASES and f['epochs_completed']==100 and f['stop_reason']=='maximum_epochs'
        assert f['config']['original_indices']==list(study.INDICES) and f['config']['minimum_epochs']>100
        assert f['config']['maximum_epochs']==100 and f['config']['mixup_alpha']==.2
        assert f['config']['classes']==study.plan()['classes']
    v.update(human_two_replacements_fixed=True,equal100_both_complete=True,new_audit_source_sha256=study.s.sha(__file__),test_arrays_read=0,test_evaluations=0)
    study.s.save(a.out/'audit.json',v)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','out','data'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        mark=a.base/'cpu_report_failure.json'
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
