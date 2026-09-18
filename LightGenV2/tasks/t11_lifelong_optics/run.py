from pathlib import Path
import argparse, json, time
from .continual import run

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default=str(Path(__file__).parent/'configs/smoke.json')); p.add_argument('--out',default=None); a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text()); out=Path(a.out or Path(__file__).parent/'runs/smoke'/('lifelong_'+time.strftime('%Y%m%d_%H%M%S'))); result=run(cfg,out); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
