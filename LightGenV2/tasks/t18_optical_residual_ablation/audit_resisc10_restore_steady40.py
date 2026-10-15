"""Independent order/augmentation/LR mathematics for unchanged MixUp.2 E101-140."""
import argparse,inspect,os,textwrap
from pathlib import Path
import resisc10_restore_steady40 as study
import audit_resisc10_continue100 as sealed

def adapted(function,changes,extra):
    source=textwrap.dedent(inspect.getsource(function))
    for old,new in changes.items():assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    ns=dict(sealed.__dict__);ns.update(extra);exec(compile(source,__file__+':independent_consistent_math','exec'),ns)
    return ns[function.__name__]

independent_hashes=adapted(sealed.independent_hashes,{'alpha=.2 if epoch<=100 else .4':'alpha=.2'}, {})
lr_replay=adapted(sealed.lr_replay,{'        if epoch==101:rate=max(rate*.5,2e-5)\n':''}, {})
verify=adapted(sealed.audit,{"assert stage in ('complete100','regularize20')":"assert stage=='steady40'","prefix=92 if stage=='complete100' else 100":"prefix=100"},dict(study=study,base=study.base,independent_hashes=independent_hashes,lr_replay=lr_replay))

def audit(a):
    assert (a.base/'audit_claim.json').exists() and not (a.base/'cpu_report_failure.json').exists()
    verify(a)
    d=study.s.read(a.out/'audit.json');assert d['passed'] and d['actual_adam_steps']==49000
    d.update(unchanged_mixup02_verified=True,no_manual_lr_halving_verified=True,new_audit_source_sha256=study.s.sha(__file__),control_comparison='Historical reference only; not a new equal-budget ablation')
    study.s.save(a.out/'audit.json',d)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        mark=a.base/'cpu_report_failure.json'
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
