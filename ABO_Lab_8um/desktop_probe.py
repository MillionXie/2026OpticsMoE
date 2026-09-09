"""Bounded hardware smoke test; can run as an interactive scheduled task.

The SSH service desktop cannot present HDMI/DisplayPort SLM images. This
worker must run in the already logged-in console session. No credentials.
"""
import argparse
import sys
import traceback
import time
import numpy as np
from common import ROOT,config,write

def main():
    p=argparse.ArgumentParser(); p.add_argument('--bmp',default='generated/cal/A_L.bmp')
    p.add_argument('--sequence',action='store_true',help='Short black-white-black raw CCD response test')
    args=p.parse_args()
    out=ROOT/'results/desktop_probe'; out.mkdir(parents=True,exist_ok=True)
    capture_dir=out/time.strftime('%Y%m%d_%H%M%S'); capture_dir.mkdir(exist_ok=False)
    write(out/'status.json',{'status':'running','bmp':args.bmp})
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log:
        sys.stdout=log; sys.stderr=log
        try:
            from hardware import Bench
            c,_=config()
            if args.sequence: c['settle_delay_ms']=1000 # Diagnostic only, not the formal timing.
            print('Opening Holoeye and DVP in the interactive desktop...',flush=True)
            with Bench(c) as bench:
                files=['generated/cal/A_BLACK.bmp','generated/cal/A_WHITE.bmp','generated/cal/A_BLACK.bmp'] if args.sequence else [args.bmp]
                stats=[]
                for i,bmp in enumerate(files):
                    raw=bench.capture(ROOT/bmp,capture_dir/f'{i:02d}_{__import__("pathlib").Path(bmp).stem}',rectify=False)
                    stats.append({'bmp':bmp,'mean':float(raw.mean()),'p99':float(np.percentile(raw,99)),
                                  'max':int(raw.max()),'saturated_fraction':float(np.mean(raw==np.iinfo(raw.dtype).max))})
                # Leave the amplitude screen black on a normal exit.
                black=ROOT/'generated/cal/A_BLACK.bmp'
                bench.slm.preload_files([black]); bench.slm.display_file(black)
            report={'status':'passed','bmp':args.bmp,'real_hardware':True,'captures':str(capture_dir),'statistics':stats,'devices':bench.info}
            write(out/'status.json',report); write(capture_dir/'report.json',report)
        except BaseException as e:
            traceback.print_exc(); write(out/'status.json',{'status':'failed','error':str(e)}); raise

if __name__=='__main__': main()
