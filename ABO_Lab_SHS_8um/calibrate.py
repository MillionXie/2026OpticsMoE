"""Generate ABO-compatible 8 um / 8 um alignment and inverted phase BMPs offline."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def helpers():
    folder=ROOT/'compat_abo'
    if not folder.is_dir():folder=ROOT.parent/'ABO_Lab_8um'
    sys.path.insert(0,str(folder))
    import patterns,dual_patterns,fresnel,common
    for m in (patterns,dual_patterns,fresnel,common):m.ROOT=ROOT
    return patterns,dual_patterns


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(ROOT/'config.json'))
    p.add_argument('--phases',action='store_true',help='Also export the six fixed ABO phase NPYs from assets/phases')
    a=p.parse_args();c=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    if c['model_active_pixels']!=478 or c['model_pitch_um']!=17:raise ValueError('Fixed ABO checkpoint requires 478 x 17 um physical aperture')
    patterns,dual=helpers()
    from diagnostic_config import configure_generated
    configure_generated(c)
    patterns.calibration(c);dual.generate(c)
    if a.phases:patterns.export(c)
    print('Phase LUT direction:',c['phase_slm']['gray_encoding'],'; amplitude is NOT inverted.')


if __name__=='__main__':main()
