"""Build a standalone simulation/fine-tuning release (not a camera/SLM SDK bundle)."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

TASK=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not (args.data/'data/abo_similarity10_manifest.csv').is_file():raise FileNotFoundError('Wrong data root')
    if __package__:
        from .standalone.io import verify_assets
    else:
        from standalone.io import verify_assets
    verify_assets(args.assets)
    files={f'standalone/{f.name}':f for f in sorted((TASK/'standalone').glob('*.py'))}
    for name in ['run.py','README.md','COMMAND.md','requirements.txt']:files[name]=TASK/name
    for root,label in [(args.assets,'assets'),(args.data,'data')]:
        for f in sorted(root.rglob('*')):
            if f.is_file():
                if not f.resolve().is_relative_to(root.resolve()):raise ValueError('External symlink in release data')
                files[label+'/'+f.relative_to(root).as_posix()]=f
    for name in ['final_report.json','execution.json','retrieval_predictions.csv','per_category_metrics.csv','phase_masks.png']:
        f=args.reference/name
        if not f.is_file():raise FileNotFoundError(f)
        files['reference/'+name]=f
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip()
    manifest={'source_commit':commit,'hardware_sdk_included':False,
              'description':'Standalone simulation + all-optical/electronic fine-tuning, fixed prompt, no full Qwen.',
              'files':{k:{'sha256':digest(v),'bytes':v.stat().st_size} for k,v in files.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for name,path in files.items():archive.write(path,name)
        archive.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    result={'zip':str(args.output),'bytes':args.output.stat().st_size,'sha256':digest(args.output),'source_commit':commit}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
