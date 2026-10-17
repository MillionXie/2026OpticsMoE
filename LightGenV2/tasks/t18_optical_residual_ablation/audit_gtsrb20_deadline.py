"""Own GTSRB receipt; reuse the frozen independent generic 20-class math audit."""
import argparse
from pathlib import Path
import gtsrb20_deadline as study
from audit_caltech20_alt246_richsplit import audit
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:
        assert a.base==study.OUT;audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
