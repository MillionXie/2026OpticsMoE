"""Run an explicit project command in the logged-in Windows desktop with a log."""
import argparse,os,runpy,sys,traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--log',required=True,type=Path)
    p.add_argument('script',choices=['joint_diagnostic.py','display_probe.py','digit_timing.py','slm_camera.py','run.py'])
    p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    root=Path(__file__).resolve().parent;os.chdir(root);a.log.parent.mkdir(parents=True,exist_ok=True)
    with a.log.open('x',encoding='utf-8',buffering=1) as stream:
        sys.stdout=sys.stderr=stream
        sys.argv=[str(root/a.script)]+a.arguments
        try:runpy.run_path(str(root/a.script),run_name='__main__')
        except BaseException:
            traceback.print_exc();raise

if __name__=='__main__':main()
