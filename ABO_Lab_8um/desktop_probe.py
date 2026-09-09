"""Bounded hardware smoke test; can run as an interactive scheduled task.

The SSH service desktop cannot present HDMI/DisplayPort SLM images. This
worker must run in the already logged-in console session. No credentials.
"""
import argparse
import sys
import traceback
from common import ROOT,config,write

def main():
    p=argparse.ArgumentParser(); p.add_argument('--bmp',default='generated/cal/A_L.bmp'); args=p.parse_args()
    out=ROOT/'results/desktop_probe'; out.mkdir(parents=True,exist_ok=True)
    write(out/'status.json',{'status':'running','bmp':args.bmp})
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log:
        sys.stdout=log; sys.stderr=log
        try:
            from hardware import Bench
            c,_=config()
            print('Opening Holoeye and DVP in the interactive desktop...',flush=True)
            with Bench(c) as bench:
                bench.capture(ROOT/args.bmp,out/'raw',rectify=False)
                # Leave the amplitude screen black on a normal exit.
                black=ROOT/'generated/cal/A_BLACK.bmp'
                bench.slm.preload_files([black]); bench.slm.display_file(black)
            write(out/'status.json',{'status':'passed','bmp':args.bmp,'real_hardware':True})
        except BaseException as e:
            traceback.print_exc(); write(out/'status.json',{'status':'failed','error':str(e)}); raise

if __name__=='__main__': main()
