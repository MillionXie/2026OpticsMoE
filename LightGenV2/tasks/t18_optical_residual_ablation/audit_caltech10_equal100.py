"""Independent100-epoch saved-math audit; old source unchanged."""
import argparse,inspect,textwrap
from pathlib import Path
import audit_caltech10_followup as sealed
import caltech10_equal100 as study
src=textwrap.dedent(inspect.getsource(sealed.verify));assert src.count('else:assert len(h)==150;')==1
src=src.replace('else:assert len(h)==150;','else:assert len(h)==100;');ns=dict(sealed.__dict__,study=study);exec(compile(src,__file__+':independent100verify','exec'),ns);verify=ns['verify']
src=textwrap.dedent(inspect.getsource(sealed.audit));ns=dict(sealed.__dict__,study=study,verify=verify);exec(compile(src,__file__+':independent100audit','exec'),ns);audit=ns['audit'];s=study.s
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['L4','L3','L2'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        fail=a.root/(a.stage+'_cpu_report_failure.json')
        if not fail.exists():s.save(fail,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
