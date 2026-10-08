"""Six-stage LGVQ entry to the shared, audited phase-owner supervisor."""
import argparse
from pathlib import Path
from LightGenV2.tasks.t03_saliency.lab_supervise import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bench-root','link-config','phases','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('project','session','config'):p.add_argument('--'+name,required=True)
    p.add_argument('--resume-completed',action='store_true')
    p.add_argument('--verify-phase-optically',action='store_true')
    p.add_argument('--phase-reference-dir',type=Path)
    a=p.parse_args();a.task='lgvq';run(a)


if __name__=='__main__':main()
