"""Check SHA-pinned shared-frontend CLI imports on CPU without running experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def check(root, manifest, dependencies):
    root=Path(root).resolve()
    rows=manifest['paths']+dependencies['source_files']
    checked=[]
    for row in rows:
        path=root/row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError('Unsafe source fixture')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise RuntimeError('Source identity changed: '+row['path'])
        if 'bytes' in row and len(raw)!=row['bytes']:
            raise RuntimeError('Source length changed')
        if path.suffix=='.py':compile(raw,path.as_posix(),'exec')
        elif path.suffix=='.json':json.loads(raw)
        checked.append(row['path'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    results=[]
    for relative in ('shared_frontend/run.py','shared_frontend/continue_training.py',
                     'shared_frontend/evaluate_holdout.py','pure_optical/run.py'):
        path=root/'LightGenV2/demo_check'/relative
        process=subprocess.run([sys.executable,str(path),'--help'],cwd=root,env=env,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
        if process.returncode!=0 or '--data' not in process.stdout or '--out' not in process.stdout:
            raise RuntimeError('CLI import/contract failed: '+relative+'\n'+process.stderr)
        results.append({'entry':relative,'help_exit_code':process.returncode,'imports_passed':True})
    cfg=json.loads((root/'LightGenV2/demo_check/shared_frontend/config.json').read_text())
    continuation=json.loads((root/'LightGenV2/demo_check/shared_frontend/continuation.json').read_text())
    assert cfg['no_alpha'] and cfg['no_output_residual'] and cfg['no_trainable_electronic_adapter_or_head']
    assert cfg['test_set_used'] is False and continuation['test_set_used'] is False
    assert continuation['resume_epoch']==cfg['epochs']==20
    return {'source_paths_verified':checked,'cli_results':results,'cuda_visible_devices':'',
            'dataset_read':False,'training_or_evaluation_run':False,'output_directory_created':False,
            'historical_accuracy_revalidated':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--dependencies',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(check(args.root,json.loads(args.manifest.read_text()),json.loads(args.dependencies.read_text())),indent=2))
