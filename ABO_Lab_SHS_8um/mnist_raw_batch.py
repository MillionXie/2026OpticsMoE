"""One fixed, coordinator-owned phase; bounded sequential raw CCD acquisition."""
import argparse,hashlib,json,re
from pathlib import Path
from PIL import Image
from slm_camera import Controller
ROOT=Path(__file__).resolve().parent
def load_spec(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'results/dual_jobs'):raise ValueError('Spec outside controlled jobs')
    s=json.loads(path.read_text(encoding='utf-8'));rows=s['mnist_rows']
    if not 1<=len(rows)<=128 or len({r['name'] for r in rows})!=len(rows):raise ValueError('Invalid count/identity')
    if not re.fullmatch('[0-9a-f]{64}',s['phase_receipt']['phase_sha256']):raise ValueError('Missing phase receipt')
    dest=(ROOT/s['out']).resolve()
    if not dest.is_relative_to(ROOT/'results'):raise ValueError('Output path escape')
    for r in rows:
        if not re.fullmatch('[A-Za-z0-9_-]+',r['name']):raise ValueError('Unsafe name')
        bmp=(ROOT/r['bmp']).resolve()
        if not bmp.is_relative_to(ROOT/'generated'):raise ValueError('BMP outside generated')
        if hashlib.sha256(bmp.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('BMP mismatch')
        if r['phase_sha256']!=s['phase_receipt']['phase_sha256']:raise ValueError('Batch phases differ')
    return s,dest
def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args()
    spec,dest=load_spec(a.spec);c=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    dest.mkdir(parents=True,exist_ok=False)
    with Controller(c) as hw:
        for i,r in enumerate(spec['mnist_rows']):
            frame,meta=hw.capture(ROOT/r['bmp']);sub=dest/r['name'];sub.mkdir()
            Image.fromarray(frame).save(sub/'raw.png')
            meta['phase_receipt']=spec['phase_receipt'];meta['phase_control']='local coordinator RGBA SDK; held throughout batch'
            (sub/'capture.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            print(f"Captured MNIST {i+1}/{len(spec['mnist_rows'])} {r['name']}",flush=True)
if __name__=='__main__':main()
