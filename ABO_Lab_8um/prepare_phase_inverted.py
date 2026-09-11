"""Create an opt-in inverted-LUT profile and new BMP tree; preserve old sessions."""
import argparse
import copy
from common import ROOT,config,write
from phase_encoding import mode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config')
    p.add_argument('--calibration-only',action='store_true')
    a=p.parse_args();c,source=config(a.config);c=copy.deepcopy(c)
    c['phase_slm']['gray_encoding']='inverted_255_minus_g'
    target=ROOT/'LAB.phase_inverted.json'
    if target.exists() and config(target)[0]!=c:raise FileExistsError('Different inverted profile exists: '+str(target))
    write(target,c)
    from patterns import calibration,export
    calibration(c)
    from dual_patterns import generate
    generate(c)
    if not a.calibration_only:export(c)
    print('New profile:',target,'; use NEW sessions with --config',target.name)


if __name__=='__main__':main()
