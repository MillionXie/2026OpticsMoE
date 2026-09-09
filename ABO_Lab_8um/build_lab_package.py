"""Build an immutable Windows release, excluding original TF weights/runs."""
import argparse
import subprocess
import zipfile
from pathlib import Path
from common import ROOT,read,write,sha

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'releases/ABO_Holoeye8_DVP.zip'); a=p.parse_args()
    repo=ROOT.parent
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    paths={}
    def add(src,arc):
        if src.is_file(): paths[arc.replace('\\','/')]=src
    # Only tracked new adapter files; original imported files keep their audit.
    tracked=subprocess.check_output(['git','ls-files','ABO_Lab_8um'],cwd=repo,text=True).splitlines()
    for name in tracked:
        pth=Path(name); add(repo/pth,str(pth.relative_to('ABO_Lab_8um')))
    for folder in ('runtime','assets','models','generated'):
        for f in (ROOT/folder).rglob('*'):
            if not f.is_file() or '__pycache__' in f.parts: continue
            rel=f.relative_to(ROOT)
            if folder=='models' and f.suffix=='.safetensors' and f.name!='native_student.safetensors': continue
            if folder=='models' and f.name=='ASSET_MANIFEST.json': continue # Full source manifest does not describe compact export.
            add(f,str(rel))
    # Include audited current common driver/legacy worker without bringing old data.
    sdk=repo/'experiments/hardware_sdk'
    for f in sdk.rglob('*.py'):
        if any(x in f.parts for x in ('vendor_sdk','artifacts','tests','dvp_runtime','__pycache__')): continue
        add(f,'runtime/backend/experiments/hardware_sdk/'+str(f.relative_to(sdk)))
    for name in ('SOURCE_AUDIT.json','reference/FINAL_VERIFICATION.json','reference/final_report.json'):
        add(ROOT/'original_inference'/name,'provenance/'+name)
    for f in (ROOT/'results').rglob('*.json'):
        if any(x.startswith('compact') or x=='import_manifest.json' for x in f.relative_to(ROOT/'results').parts):
            add(f,'evidence/'+str(f.relative_to(ROOT/'results')))
    compact=ROOT/'models/Qwen3-VL-Embedding-2B/native_student.safetensors'
    if not compact.is_file(): raise FileNotFoundError('Run simulate.py --export-native before packaging')
    manifest={'schema':1,'git_commit':commit,'files':{n:sha(f) for n,f in sorted(paths.items())},
        'hardware':'Holoeye8um + manual phase8um + DVP legacy','sister_checkpoint_reference':80.58333333333333,
        'not_included':['training dataset','unused frozen Transformer weights','sister-lab LUT and CCD homography','old sessions']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,f in sorted(paths.items()): z.write(f,name)
        import json
        z.writestr('RELEASE_MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    write(a.output.with_suffix('.sha256.json'),{'zip':a.output.name,'sha256':sha(a.output),'bytes':a.output.stat().st_size,'git_commit':commit})
    # Small source-only update for a machine that already has these same assets.
    # Full manifest is verified after extraction: mismatched old assets still fail.
    with zipfile.ZipFile(a.output.with_name('ABO_source_update.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for name,f in sorted(paths.items()):
            if name.endswith(('.py','.md','.txt','.json','.yaml','.ps1')) and not name.startswith(('models/','assets/')):
                z.write(f,name)
        z.writestr('RELEASE_MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    print(a.output,sha(a.output),flush=True)

if __name__=='__main__': main()
