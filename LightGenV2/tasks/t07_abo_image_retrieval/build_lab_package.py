"""Build a standalone simulation/fine-tuning release (not a camera/SLM SDK bundle)."""
import argparse
import hashlib
import json
import subprocess
import shutil
import zipfile
from pathlib import Path

TASK=Path(__file__).resolve().parent
BEST_83125='c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0'
PROTOCOL_83125='f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde'


def enrolled_files(assets, data, reference, protocol):
    """Pinned release: no legacy checkpoint, teacher targets or obsolete split."""
    if digest(reference/'best.pt') != BEST_83125 or digest(protocol) != PROTOCOL_83125:
        raise ValueError('83.125 release requires the pinned best and enrolled protocol')
    report=json.loads((reference/'verification/final_report.json').read_text(encoding='utf-8'))
    if (report['checkpoint_sha256'] != BEST_83125 or report['manifest_sha256'] != PROTOCOL_83125
            or report['metrics']['normal']['hit_at_1'] != .83125):
        raise ValueError('Independent raw-image verification does not match release')
    files={f'standalone/{f.name}':f for f in sorted((TASK/'standalone').iterdir())
           if f.is_file() and f.suffix in ('.py','.json')}
    files['requirements.txt']=TASK/'requirements.txt'
    for name in ('README.md','AI_HANDOFF.md','delivery.py'):
        files[name]=TASK/'delivery_83125'/name
    files['assets/best.pt']=reference/'best.pt'
    for f in sorted((assets/'processor').rglob('*')):
        if f.is_file():
            if not f.resolve().is_relative_to(assets.resolve()):raise ValueError('External processor symlink')
            files['assets/'+f.relative_to(assets).as_posix()]=f
    files['protocol.json']=protocol
    rows=json.loads(protocol.read_text(encoding='utf-8'))['rows']
    for row in rows:
        f=(data/row['image_path']).resolve()
        if not f.is_relative_to(data.resolve()) or not f.is_file() or digest(f)!=row['image_sha256']:
            raise ValueError('Missing/changed protocol image: '+row['sample_id'])
        files['data/'+row['image_path']]=f
    for name in ('final_report.json','execution.json','normal_predictions.csv','remove_optical_predictions.csv','weight_train_audit.json'):
        files['reference/'+name]=reference/'verification'/name
    files['reference/readout_fit_report.json']=reference/'final_report.json'
    generated={'assets/manifest.json':json.dumps({'schema':1,'files':{
        k.removeprefix('assets/'):digest(v) for k,v in files.items() if k.startswith('assets/')}},indent=2).encode()}
    return files,generated


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
    parser.add_argument('--output',type=Path,help='Final ZIP; cannot overwrite')
    parser.add_argument('--stage',type=Path,help='Optional clean directory for isolated acceptance before ZIP')
    parser.add_argument('--profile',choices=['legacy','enrolled83125'],default='legacy')
    parser.add_argument('--protocol',type=Path,help='Pinned enrolled protocol, required for enrolled83125')
    args=parser.parse_args()
    if args.output is None and args.stage is None:parser.error('Provide --output and/or --stage')
    if args.output is not None and args.output.exists():raise FileExistsError(args.output)
    if args.stage is not None and args.stage.exists():raise FileExistsError(args.stage)
    if args.profile=='enrolled83125' and args.protocol is None:parser.error('enrolled83125 requires --protocol')
    if not (args.data/'data/abo_similarity10_manifest.csv').is_file():raise FileNotFoundError('Wrong data root')
    if __package__:
        from .standalone.io import verify_assets
    else:
        from standalone.io import verify_assets
    verify_assets(args.assets)
    files={f'standalone/{f.name}':f for f in sorted((TASK/'standalone').glob('*.py'))}
    generated={}
    files.update({f'standalone/{f.name}':f for f in sorted((TASK/'standalone').glob('*.json'))})
    for name in ['run.py','README.md','COMMAND.md','ACCEPTANCE.md','requirements.txt']:files[name]=TASK/name
    for root,label in [(args.assets,'assets'),(args.data,'data')]:
        for f in sorted(root.rglob('*')):
            if f.is_file():
                if not f.resolve().is_relative_to(root.resolve()):raise ValueError('External symlink in release data')
                files[label+'/'+f.relative_to(root).as_posix()]=f
    if args.profile=='enrolled83125':
        files,generated=enrolled_files(args.assets,args.data,args.reference,args.protocol)
    else:
        for name in ['final_report.json','execution.json','retrieval_predictions.csv','per_category_metrics.csv','phase_masks.png']:
            f=args.reference/name
            if not f.is_file():raise FileNotFoundError(f)
            files['reference/'+name]=f
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip()
    manifest={'source_commit':commit,'profile':args.profile,'hardware_sdk_included':False,
              'description':'Standalone simulation + all-optical/electronic fine-tuning, fixed prompt, no full Qwen.',
              'files':{k:{'sha256':digest(v),'bytes':v.stat().st_size} for k,v in files.items()}}
    manifest['files'].update({k:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for k,v in generated.items()})
    if args.stage is not None:
        args.stage.mkdir(parents=True)
        for name,path in files.items():
            target=args.stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        for name,content in generated.items():
            target=args.stage/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        (args.stage/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'stage':str(args.stage),'source_commit':commit,'files':len(files)},indent=2))
    if args.output is None:return
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for name,path in files.items():archive.write(path,name)
        for name,content in generated.items():archive.writestr(name,content)
        archive.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    result={'zip':str(args.output),'bytes':args.output.stat().st_size,'sha256':digest(args.output),'source_commit':commit}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
