"""Genuine saved independent ten-class mathematics under the .2 MixUp profile."""
import argparse
from pathlib import Path
import resisc10_optimize as opt
import audit_resisc10 as sealed
sealed.study=opt.base;sealed.__file__=__file__
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','out','data'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();opt.admit_parent()
    try:sealed.audit(a)
    except BaseException as e:
        mark=a.base/'cpu_report_failure.json'
        if not mark.exists():opt.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
