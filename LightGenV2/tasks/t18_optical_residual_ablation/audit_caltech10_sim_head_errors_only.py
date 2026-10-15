"""Same independent saved-math audit with the explicit trained E12 prefix."""
import argparse,inspect,os,textwrap
from pathlib import Path
import caltech10_sim_head_errors_only as configured
import audit_caltech10_sim_head_preserve as previous
code=textwrap.dedent(inspect.getsource(previous.audit));assert code.count("hist[0]['head_epoch']==3")==1;code=code.replace("hist[0]['head_epoch']==3","hist[0]['head_epoch']==12");exec(compile(code,__file__+':audit','exec'),previous.__dict__)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:previous.audit(a.out)
    except BaseException as error:
        if not (a.out/'audit_failure.json').exists():configured.s.save(a.out/'audit_failure.json',dict(error=repr(error),automatic_retry=False),exclusive=True)
        raise
