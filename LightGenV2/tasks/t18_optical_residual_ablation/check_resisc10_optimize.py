"""Actual new CPU native/label gate for the declared .2 MixUp pure profile."""
import argparse
from pathlib import Path
import resisc10_optimize as opt
import check_resisc10 as sealed
sealed.study=opt.base
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();opt.admit_parent();sealed.check(a.out)
