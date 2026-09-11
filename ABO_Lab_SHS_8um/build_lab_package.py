"""Versioned source-only deployment; vendor SDK, secrets and captures excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;repo=root.parent
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'--',root.name],cwd=repo,text=True).splitlines()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    manifest={'commit':commit,'source_only':True,'files':{}}
    with zipfile.ZipFile(a.out,'x',zipfile.ZIP_DEFLATED) as z:
        for name in files:
            data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
            rel=str(Path(name).relative_to(root.name)).replace('\\','/')
            z.writestr(rel,data);manifest['files'][rel]=hashlib.sha256(data).hexdigest()
        z.writestr('CODE_MANIFEST.json',json.dumps(manifest,indent=2))
    h=hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(a.out.resolve(),h,flush=True)


if __name__=='__main__':main()
