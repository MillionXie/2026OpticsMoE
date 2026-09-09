"""Hold one native-size amplitude BMP; leave the CCD free for its own GUI."""
import argparse
from common import ROOT,config,setup_imports

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bmp',default='generated/cal/A_CHECK_32.bmp')
    a=p.parse_args(); c,_=config(); setup_imports()
    from experiments.hardware_sdk.devices import build_slm
    print('Amplitude only. Phase unchanged; CCD is NOT opened. Close other amplitude players first.',flush=True)
    with build_slm(c['amplitude_slm'],ROOT) as slm:
        print(slm.device_info(),flush=True)
        path=(ROOT/a.bmp).resolve()
        slm.preload_files([path]); slm.display_file(path)
        try:
            input('BMP held at native size. Use CCD software to align. Press Enter to end: ')
        finally:
            black=ROOT/'generated/cal/A_BLACK.bmp'
            slm.preload_files([black]); slm.display_file(black)

if __name__=='__main__': main()
