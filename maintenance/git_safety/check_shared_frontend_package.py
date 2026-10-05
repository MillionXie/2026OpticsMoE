"""Audit shared-frontend preparation/package sources without downloading or building."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def check(root,manifest):
    root=Path(root).resolve()
    sys.dont_write_bytecode=True
    sources=[]
    for row in manifest['paths']:
        path=root/row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError('Unsafe fixture path')
        raw=path.read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise RuntimeError('Source fixture changed: '+row['path'])
        if path.suffix=='.py':compile(raw,str(path),'exec')
        elif path.suffix=='.json':json.loads(raw)
        sources.append(row['path'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    results=[]
    for relative in ('pure_optical/prepare.py','shared_frontend/prepare_holdout.py',
                     'EuroSAT_MoE_D2NN/code/download_archives.py','build_lab_package.py'):
        process=subprocess.run([sys.executable,str(root/'LightGenV2/demo_check'/relative),'--help'],
                               cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
        if process.returncode or 'usage:' not in process.stdout:
            raise RuntimeError('CLI import failed: '+relative+'\n'+process.stderr)
        results.append({'entry':relative,'help_exit_code':0})
    path=root/'LightGenV2/demo_check/shared_frontend/package_builder.py'
    spec=importlib.util.spec_from_file_location('audit_package_builder',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.build)
    canonical=[]
    for relative in ('pure_optical/models.py','pure_optical/run.py','frozen_electronic/model.py',
                     'shared_frontend/model.py','shared_frontend/run.py',
                     'adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py'):
        path='LightGenV2/demo_check/'+relative
        raw=subprocess.check_output(['git','-C',str(root),'show',module.EXPORT_SOURCE+':'+path])
        ast.parse(raw)
        canonical.append({'path':path,'sha256':hashlib.sha256(raw).hexdigest()})
    assert module.TRAINING=='8cb125cfaa4829e7714b5b23389f0c96e4844a72'
    return {'verified_paths':sources,'cli_results':results,'builder_imported':True,
            'canonical_export_source':module.EXPORT_SOURCE,'canonical_sources':canonical,
            'package_built':False,'download_started':False,'dataset_read':False,
            'other_package_variants_validated':False,'cuda_visible_devices':''}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(check(args.root,json.loads(args.manifest.read_text())),indent=2))
