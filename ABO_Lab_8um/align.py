"""Hold one native-size amplitude BMP; leave the CCD free for its own GUI."""
import argparse
from common import ROOT,config,setup_imports,sha
from phase_encoding import generated_root

def main():
    p=argparse.ArgumentParser(description=__doc__)
    choice=p.add_mutually_exclusive_group()
    choice.add_argument('--bmp')
    choice.add_argument('--pair',help='Folder containing matched A.bmp and P.bmp')
    p.add_argument('--opposite-vertical',action='store_true',help='With --pair, request P_V.bmp instead of P.bmp')
    a=p.parse_args(); c,_=config(); setup_imports()
    if a.opposite_vertical and a.bmp: p.error('--opposite-vertical requires a pair, not --bmp')
    if not a.bmp:
        pair=ROOT/a.pair if a.pair else generated_root(ROOT,c)/'dual/01_check64'
        phase=pair/('P_V.bmp' if a.opposite_vertical else 'P.bmp')
        if not phase.is_file(): raise FileNotFoundError(phase)
        print('MANUALLY LOAD matched phase:',phase.resolve(),flush=True)
        print('Phase SHA256:',sha(phase),flush=True)
        a.bmp=str(pair/'A.bmp')
    from experiments.hardware_sdk.devices import build_slm
    print('Amplitude only. Phase unchanged; CCD is NOT opened. Close other amplitude players first.',flush=True)
    with build_slm(c['amplitude_slm'],ROOT) as slm:
        print(slm.device_info(),flush=True)
        path=(ROOT/a.bmp).resolve()
        slm.preload_files([path]); slm.display_file(path)
        try:
            input('BMP held at native size. Use CCD software to align. Press Enter to end: ')
        finally:
            black=generated_root(ROOT,c)/'cal/A_BLACK.bmp'
            slm.preload_files([black]); slm.display_file(black)

if __name__=='__main__': main()
