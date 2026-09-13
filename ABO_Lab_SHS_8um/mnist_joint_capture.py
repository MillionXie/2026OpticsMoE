"""Bounded raw-image diagnostic; does not approve ABO ROI or alter LAB.local.json.

Local phase owner + remote logged-in desktop amplitude/camera. Manifest pins
each native BMP and its phase. Existing camera configuration is recorded, never
silently replaced; numeric analysis is a separate operation on immutable PNGs.
"""
import argparse, hashlib, json, subprocess
from pathlib import Path
from PIL import Image
from phase_owner import PhaseOwner
from dual_run import Remote

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def run(manifest, link_path, out):
    m=read(manifest);link=read(link_path)
    rows=m['rows']
    if not 1<=len(rows)<=128: raise ValueError('Bounded diagnostic requires 1..128 captures')
    if len({r['name'] for r in rows})!=len(rows): raise ValueError('Duplicate capture identity')
    import re
    for row in rows:
        if not re.fullmatch('[A-Za-z0-9_-]+',row['name']): raise ValueError('Unsafe name')
        for key, size in [('amplitude',(1920,1080)),('phase',(1920,1200))]:
            path=Path(row[key]).resolve()
            if not path.is_relative_to(ROOT): raise ValueError('Asset outside project')
            if sha(path)!=row[key+'_sha256']: raise ValueError('Asset hash mismatch')
            with Image.open(path) as im:
                if im.mode!='L' or im.format!='BMP' or im.size!=size: raise ValueError('Native BMP required')
    link.update(phase_startup_cycles=0,phase_pixel_format='rgba',phase_display_align_top=True)
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT/'results'): raise ValueError('Output outside results')
    out.mkdir(parents=True,exist_ok=False)
    report={'complete':False,'kind':'MNIST raw diagnostic, not full dataset accuracy',
            'manifest':m,'manifest_sha256':sha(manifest),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'camera_config':'LAB.local.json','rows':[]}
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    try:
        with Remote(link) as remote:
            remote.ps("$busy=Get-Process | Where-Object {$_.ProcessName -match 'FastStream|Slideshow|python|Viewer'};if($busy){throw 'Remote hardware busy'}")
            report['remote_config_snapshot']=remote.read('LAB.local.json');save()
            for row in rows:
                p=Path(row['amplitude']).resolve();rel=p.relative_to(ROOT).as_posix()
                parent=(remote.root+'/'+rel).rsplit('/',1)[0]
                remote.ps(f"New-Item -ItemType Directory -Force -Path '{parent}' | Out-Null")
                remote.sftp.put(str(p),remote.root+'/'+rel)
                with remote.sftp.open(remote.root+'/'+rel,'rb') as f:
                    if hashlib.sha256(f.read()).hexdigest()!=row['amplitude_sha256']: raise ValueError('Uploaded BMP corrupted')
            flat=ROOT/'generated/phase_response_strong_20260913/P_flat_0.bmp'
            lens=ROOT/'generated/phase_response_strong_20260913/P_lens_10cm_inverse.bmp'
            with PhaseOwner(link,flat,lens) as owner:
                for row in rows:
                    receipt=owner.show(row['phase'],row['phase_sha256'])
                    dest=out.relative_to(ROOT).as_posix()+'/'+row['name']
                    remote.job({'action':'probe','bmp':Path(row['amplitude']).resolve().relative_to(ROOT).as_posix(),
                                'out':dest,'phase_receipt':receipt})
                    remote.download(dest+'/raw.png',out/(row['name']+'.png'))
                    remote.download(dest+'/capture.json',out/(row['name']+'.json'))
                    report['rows'].append(dict(row,receipt=receipt,raw_sha256=sha(out/(row['name']+'.png'))))
                    save();print('CAPTURED',row['name'],flush=True)
            report['display_audit']=owner.display.audit
        report['complete']=True
    except BaseException as e:
        report['error']=str(e);raise
    finally: save()
    print('RESULT',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--link',type=Path,default=ROOT/'dual.local.json');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.manifest,a.link,a.out)
