"""Bounded raw-image diagnostic; does not approve ABO ROI or alter LAB.local.json.

Local phase owner + remote logged-in desktop amplitude/camera. Manifest pins
each native BMP and its phase. Existing camera configuration is recorded, never
silently replaced; numeric analysis is a separate operation on immutable PNGs.
"""
import argparse, hashlib, json, subprocess,zipfile
from pathlib import Path
from PIL import Image
from phase_owner import PhaseOwner
from dual_run import Remote

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def run(manifest, link_path, out, batch=False, config_rel='LAB.local.json'):
    m=read(manifest);link=read(link_path)
    rows=m['rows']
    if batch and len({r['phase_sha256'] for r in rows})!=1:raise ValueError('Batch needs exactly one fixed phase')
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
    report={'complete':False,'kind':m.get('kind','MNIST raw diagnostic, not full dataset accuracy'),
            'manifest':m,'manifest_sha256':sha(manifest),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'camera_config':config_rel,'rows':[]}
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    try:
        with Remote(link) as remote:
            remote.ps("$busy=Get-Process | Where-Object {$_.ProcessName -match 'FastStream|Slideshow|python|Viewer'};if($busy){throw 'Remote hardware busy'}")
            report['remote_config_snapshot']=remote.read(config_rel);save()
            # Native BMPs are highly compressible. Transfer one data-only ZIP,
            # never regenerate/rescale or send source files through this path.
            assets={Path(r['amplitude']).resolve().relative_to(ROOT).as_posix():r['amplitude_sha256'] for r in rows}
            zpath=out/'amplitude_transfer.zip'
            with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
                for rel in assets:z.write(ROOT/rel,rel)
            relzip=out.relative_to(ROOT).as_posix()+'/amplitude_transfer.zip'
            remote.ps(f"New-Item -ItemType Directory -Force -Path '{remote.root}/{out.relative_to(ROOT).as_posix()}' | Out-Null")
            remote.sftp.put(str(zpath),remote.root+'/'+relzip)
            # A hash manifest avoids Windows' encoded-command length limit
            # when validating dozens of BMPs. The manifest itself is pinned.
            hashfile=out/'amplitude_hashes.json';hashfile.write_text(json.dumps(assets),encoding='utf-8')
            relhash=out.relative_to(ROOT).as_posix()+'/amplitude_hashes.json'
            remote.sftp.put(str(hashfile),remote.root+'/'+relhash)
            # Every entry is from validated project-local files generated for
            # this run; extraction rejects overwrites and escaping entries.
            remote.ps(f"""$ErrorActionPreference='Stop'
if((Get-FileHash -LiteralPath '{remote.root}/{relzip}').Hash.ToLower() -ne '{sha(zpath)}'){{throw 'ZIP mismatch'}}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$z=[IO.Compression.ZipFile]::OpenRead('{remote.root}/{relzip}')
try{{foreach($e in $z.Entries){{
 if($e.FullName.Contains('..') -or $e.FullName.Contains(':') -or $e.FullName.StartsWith('/')){{throw 'Unsafe data entry'}}
 $p=Join-Path '{remote.root}' $e.FullName
 New-Item -ItemType Directory -Force -Path (Split-Path $p) | Out-Null
 if(-not (Test-Path -LiteralPath $p)){{[IO.Compression.ZipFileExtensions]::ExtractToFile($e,$p,$false)}}
}}}}finally{{$z.Dispose()}}
if((Get-FileHash -LiteralPath '{remote.root}/{relhash}').Hash.ToLower() -ne '{sha(hashfile)}'){{throw 'Hash manifest mismatch'}}
$hashes=Get-Content -LiteralPath '{remote.root}/{relhash}' -Raw | ConvertFrom-Json
foreach($entry in $hashes.PSObject.Properties){{
 if((Get-FileHash -LiteralPath (Join-Path '{remote.root}' $entry.Name)).Hash.ToLower() -ne $entry.Value){{throw 'BMP mismatch'}}
}}
""")
            flat=ROOT/'generated/phase_response_strong_20260913/P_flat_0.bmp'
            lens=ROOT/'generated/phase_response_strong_20260913/P_lens_10cm_inverse.bmp'
            with PhaseOwner(link,flat,lens) as owner:
                if batch:
                    receipt=owner.show(rows[0]['phase'],rows[0]['phase_sha256'])
                    batch_dest=out.relative_to(ROOT).as_posix()+'/batch'
                    remote.job({'action':'mnist_batch','config':config_rel,'out':batch_dest,'phase_receipt':receipt,'mnist_rows':[
                        dict(name=r['name'],bmp=Path(r['amplitude']).resolve().relative_to(ROOT).as_posix(),sha256=r['amplitude_sha256'],phase_sha256=r['phase_sha256']) for r in rows]})
                for row in rows:
                    dest=out.relative_to(ROOT).as_posix()+('/batch/' if batch else '/')+row['name']
                    if not batch:
                        receipt=owner.show(row['phase'],row['phase_sha256'])
                        remote.job({'action':'probe','config':config_rel,'bmp':Path(row['amplitude']).resolve().relative_to(ROOT).as_posix(),
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
    p.add_argument('--batch',action='store_true',help='One fixed phase; open hardware once for all inputs')
    p.add_argument('--remote-config',default='LAB.local.json')
    a=p.parse_args();run(a.manifest,a.link,a.out,a.batch,a.remote_config)
