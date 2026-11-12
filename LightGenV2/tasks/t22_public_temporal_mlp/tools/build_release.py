"""Build and verify the standalone public reference distribution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import torch

if __package__:
    from . import _paths
    from .checkpoint_utils import export_public_checkpoint, file_sha256
else:
    import _paths
    from checkpoint_utils import export_public_checkpoint, file_sha256

if __package__ and __package__!='tools':
    from ..model import load_lightgen
    from ..infer import iter_video_fields
else:
    from model import load_lightgen
    from infer import iter_video_fields

ROOT=_paths.ROOT
RUNTIME_FILES=(
    'infer.py','model.py','requirements.txt','requirements-visualization.txt',
    'LICENSE','NOTICE','licenses/QWEN_LICENSE.txt','examples/make_example.py',
    'tools/__init__.py','tools/_paths.py','tools/data.py','tools/plot_style.py',
    'tools/phase_plotting.py','tools/visualize_fields.py','tools/plot_global_ablation.py',
)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n','utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def copy_results(run,output):
    output.mkdir()
    summaries={}
    conditions={}
    for directory in (run/'global_ablation',run/'global_ablation/initialization_control'):
        if json.loads((directory/'status.json').read_text())['state']!='completed':
            raise ValueError('Global ablation is incomplete')
        for row in read_csv(directory/'summary.csv'):
            summaries[row['kind'],row['stage']]=row
        for row in read_csv(directory/'all_conditions.csv'):
            conditions[row['kind'],row['stage'],row['seed']]=row
    write_csv(output/'global_ablation_summary.csv',list(summaries.values()))
    write_csv(output/'global_ablation_seeds.csv',list(conditions.values()))
    original=json.loads((run/'global_ablation/protocol.json').read_text())
    write_json(output/'global_ablation_protocol.json',{
        'role':'post_training_forward_interventions_on_original_TEST_development',
        'videos':558,'retraining':False,'random_seeds':original['seeds'],
        'grouping':'original_order_with_cyclic_tail_padding_and_valid_sample_mask',
        'bypass':'identity bypass of the complete GlobalStage',
        'random_phase':'uniform physical phase in [0,2*pi), with trained mappings',
        'random_raw_phase':'raw N(0,0.25^2) through the original sigmoid, with trained mappings',
        'random_global':'randomly initialized mappings/readout and uniform physical phase',
        'random_global_default':'complete default GlobalStage initialization',
        'standard_deviation':'sample SD over ten seeds',
    })
    for name in ('phase_masks.png','phase_training_update.png','optical_expert_share.png'):
        shutil.copyfile(run/'figures'/name,output/name)


def verify_manifest(directory):
    manifest=json.loads((directory/'MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path)!=expected:
            raise ValueError('Release file checksum differs: '+name)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'releases/current')
    args=parser.parse_args()
    output=args.out.resolve()
    release_root=(ROOT/'releases').resolve()
    if output.parent!=release_root or output.name!='current':
        raise ValueError('The generated distribution uses releases/current')
    release_root.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.build-',dir=release_root)).resolve()
    if not stage.is_relative_to(release_root):
        raise ValueError('Unexpected build directory')
    profile=json.loads((ROOT/'configs/selected.json').read_text())
    source=ROOT/profile['checkpoint']
    if file_sha256(source)!=profile['checkpoint_sha256']:
        raise ValueError('Selected research checkpoint checksum differs')
    for name in RUNTIME_FILES:
        target=stage/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    (stage/'README.md').write_text((ROOT/'README.md').read_text().split('\n## Author workspace',1)[0],'utf-8')
    shutil.copyfile(ROOT/'docs/MODEL_CARD.md',stage/'MODEL_CARD.md')
    exported=export_public_checkpoint(source,stage/'weights/LightGen.pt')
    (stage/'configs').mkdir()
    write_json(stage/'configs/selected.json',{
        'checkpoint':'weights/LightGen.pt','checkpoint_sha256':exported['public_checkpoint_sha256'],
        'architecture':profile['architecture'],'model_parameters':profile['model_parameters'],
        'evaluation':profile['evaluation'],
    })
    copy_results(source.parent,stage/'results')
    environment=os.environ.copy()
    environment.update(PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
    commands=[
        [sys.executable,str(stage/'examples/make_example.py'),'--output','examples/demo.mp4'],
        [sys.executable,str(stage/'infer.py'),'--weights','weights/LightGen.pt','--videos','examples/demo.mp4',
         '--device','cpu','--output','examples/expected_predictions.json'],
        [sys.executable,str(stage/'tools/plot_global_ablation.py'),'--summary','results/global_ablation_summary.csv',
         '--output','results/global_ablation.png'],
    ]
    for command in commands:
        result=subprocess.run(command,cwd=stage,env=environment,capture_output=True,text=True,timeout=180)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
    torch.set_num_threads(4)
    original,_=load_lightgen(source)
    _,pixels=next(iter_video_fields([stage/'examples/demo.mp4']))
    with torch.inference_mode():
        expected=float(original(pixels=pixels)['prediction'][0,0])
    predictions=json.loads((stage/'examples/expected_predictions.json').read_text())['predictions']
    if len(predictions)!=1 or abs(predictions[0]['MOS']-expected)>1e-4:
        raise ValueError('Standalone release inference differs from the source model')
    for path in stage.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.md','.json','.csv','.txt'):
            text=path.read_text('utf-8-sig')
            for marker in ('Xml12','OneDrive','/DATA/','/home/','xwechat_files'):
                if marker in text:
                    raise ValueError('Local/private text in release file: '+path.relative_to(stage).as_posix())
    payload=torch.load(stage/'weights/LightGen.pt',map_location='cpu',weights_only=True)
    metadata_text=json.dumps(payload['metadata'])
    if any(marker in metadata_text for marker in ('Xml12','OneDrive','/DATA/','/home/')):
        raise ValueError('Private checkpoint metadata remains')
    files={path.relative_to(stage).as_posix():file_sha256(path) for path in sorted(stage.rglob('*')) if path.is_file()}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest={'project':'LightGen-II temporal quality reference','license':'Apache-2.0',
              'source_commit':commit,'checkpoint_export':exported,'files':files,
              'verification':{'standalone_empty_PYTHONPATH':True,'source_tensor_identity':True,
                              'example_MOS':predictions[0]['MOS'],'source_example_MOS':expected,
                              'public_metadata_checked':True}}
    write_json(stage/'MANIFEST.json',manifest)
    verify_manifest(stage)
    archive=release_root/'.current.zip.building'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as zipped:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                zipped.write(path,'lightgen-temporal/'+path.relative_to(stage).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        for name,digest in files.items():
            if hashlib.sha256(zipped.read('lightgen-temporal/'+name)).hexdigest()!=digest:
                raise ValueError('Distributable ZIP checksum differs: '+name)
    backup=None
    if output.exists():
        verify_manifest(output)
        backup=Path(tempfile.mkdtemp(prefix='lightgen-release-backup-')).resolve()
        if backup.parent!=Path(tempfile.gettempdir()).resolve():
            raise ValueError('Unexpected backup directory')
        shutil.move(str(output),str(backup/'current'))
        if (release_root/'current.zip').exists():
            shutil.copyfile(release_root/'current.zip',backup/'current.zip')
    try:
        stage.rename(output)
    except BaseException:
        if backup is not None:
            shutil.move(str(backup/'current'),str(output))
        raise
    archive.replace(release_root/'current.zip')
    digest=file_sha256(release_root/'current.zip')
    (release_root/'current.zip.sha256').write_text(digest+'  current.zip\n','utf-8')
    print(json.dumps({'directory':str(output),'archive':str(release_root/'current.zip'),
                      'archive_sha256':digest,'files':len(files)+1,'weights':exported,
                      'standalone_verified':True},indent=2))


if __name__=='__main__':
    main()
