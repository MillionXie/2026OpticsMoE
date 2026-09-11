"""Versioned source-only deployment; vendor SDK, secrets and captures excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--code-only',action='store_true',help='Small source patch for an already installed full package')
    a=p.parse_args()
    root=Path(__file__).resolve().parent;repo=root.parent
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'--',root.name],cwd=repo,text=True).splitlines()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    manifest={'commit':commit,'contains':'committed code + six phase NPYs + calibration BMPs + private vendor manuals; NO full model/dataset/sessions/SDK binaries','files':{}}
    with zipfile.ZipFile(a.out,'x',zipfile.ZIP_DEFLATED) as z:
        def add(rel,data):
            z.writestr(rel,data);manifest['files'][rel]=hashlib.sha256(data).hexdigest()
        for name in files:
            data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
            rel=str(Path(name).relative_to(root.name)).replace('\\','/')
            add(rel,data)
        # Bundle exact committed compatibility sources, never SCP working files.
        compat=['common.py','patterns.py','phase_encoding.py','registration.py','dual_patterns.py','fresnel.py',
                'run.py','backend.py','hardware.py','memory.py','raw_cleanup.py','router_quality.py']
        for name in compat:
            add('compat_abo/'+name,subprocess.check_output(['git','show',commit+':ABO_Lab_8um/'+name],cwd=repo))
        add('vendor_driver.py',subprocess.check_output(['git','show',commit+':experiments/hardware_sdk/devices.py'],cwd=repo))
        # Model phase/digit assets stay out of Git but are explicitly hash-pinned.
        for folder in (() if a.code_only else ('phases','direction_digits')):
            for path in sorted((root/'assets'/folder).glob('*')):
                if path.is_file():add('assets/'+folder+'/'+path.name,path.read_bytes())
        for folder in (() if a.code_only else ('generated/phase_inverted','vendor/manuals')):
            for path in sorted((root/folder).rglob('*')):
                if path.is_file():add(path.relative_to(root).as_posix(),path.read_bytes())
        manifest['code_only_patch']=a.code_only
        z.writestr('CODE_MANIFEST.json',json.dumps(manifest,indent=2))
    h=hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(a.out.resolve(),h,flush=True)


if __name__=='__main__':main()
