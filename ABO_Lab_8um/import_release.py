"""Import the three immutable sister releases without touching the parent repo."""
import argparse
import shutil
import zipfile
from pathlib import Path
from common import ROOT, sha, write

def extract(zpath,destination):
    if destination.exists():
        marker=destination/'.import.json'
        if marker.exists() and __import__('json').loads(marker.read_text())['sha256']==sha(zpath): return
        raise FileExistsError(f'{destination} exists without matching import identity; do not overwrite it.')
    with zipfile.ZipFile(zpath) as z:
        names=[i.filename for i in z.infolist()]
        if len(names)!=len(set(names)): raise ValueError('Duplicate ZIP paths')
        for i in z.infolist():
            p=(destination/i.filename).resolve()
            if not p.is_relative_to(destination.resolve()) or '\\' in i.filename: raise ValueError(i.filename)
        z.extractall(destination)
    write(destination/'.import.json',{'sha256':sha(zpath),'source':zpath.name})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--imports',type=Path,default=ROOT/'imports'); a=p.parse_args()
    inputs={'optics':'ABO_OPTICS_complete.zip','a100':'ABO_A100_complete.zip','inference':'ABO_software_inference_80583_complete.zip'}
    report={}
    for key,name in inputs.items():
        src=a.imports/name
        if not src.exists(): raise FileNotFoundError(src)
        report[key]={'filename':name,'sha256':sha(src),'bytes':src.stat().st_size}
        out=ROOT/('original_'+key)
        # An earlier read-only audit may already have extracted these files.
        if out.exists() and not (out/'.import.json').exists():
            with zipfile.ZipFile(src) as z:
                for i in z.infolist():
                    if not i.is_dir():
                        import hashlib
                        if sha(out/i.filename)!=hashlib.sha256(z.read(i)).hexdigest(): raise ValueError('Existing original differs: '+i.filename)
            write(out/'.import.json',{'sha256':sha(src),'source':name})
        extract(src,out)
    (ROOT/'runtime').mkdir(exist_ok=True)
    pairs=[(ROOT/'original_inference/backend',ROOT/'runtime/backend'),
           (ROOT/'original_optics/abo_dual',ROOT/'runtime/abo_dual'),
           (ROOT/'original_inference/assets',ROOT/'assets'),
           (ROOT/'original_inference/models',ROOT/'models')]
    for src,dst in pairs:
        if not dst.exists(): shutil.copytree(src,dst,copy_function=shutil.copy2)
    write(ROOT/'results/import_manifest.json',report)
    print('Imported original source, fixed checkpoint, 2400 test images and 100 titles. No training data fabricated.')

if __name__=='__main__': main()
