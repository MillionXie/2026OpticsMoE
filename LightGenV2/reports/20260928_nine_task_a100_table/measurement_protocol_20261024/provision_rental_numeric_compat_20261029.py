"""Pin original NumPy before first model, isolate modern SciPy dependency drift.

Never edit a forward or active measurement. Exact original NumPy wheel already
SHA-verified; compatible SciPy1.15.3 supports it. One finite environment overlay.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE=Path('/root/data/lgvq_20261029')

def main():
    setup=json.loads((BASE/'python_setup_started04.json').read_text());deadline=setup['deadline']
    if not Path('/proc/'+str(setup['pid'])).exists():raise ValueError('Only unfinished pre-model installer')
    if (BASE/'formal_controller_started01.json').exists():raise ValueError('No active formal environment changes')
    target=BASE/'original_numeric_compat';assert not target.exists()
    records=json.loads((BASE/'python_wheel_manifest03.json').read_text())
    item=next(r for r in records if (r['package'],r['version'])==('numpy','1.26.4'))
    wheel=BASE/'python_wheels03'/item['name']
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Original NumPy wheel identity')
    env=dict(os.environ,TMPDIR=str(BASE/'tmp'),PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run(['/root/miniconda3/bin/python','-m','pip','install','--no-deps','--no-cache-dir',
        '--retries','0','--timeout','20','--target',str(target),'-i','https://mirrors.aliyun.com/pypi/simple',
        str(wheel),'scipy==1.15.3'],env=env,check=True,timeout=min(900,max(1,deadline-time.time())))
    checked={}
    for model in ('kimi','deepseek'):
        site=BASE/'venvs'/model/'lib/python3.12/site-packages'
        while not site.exists():
            if time.time()>=deadline or not Path('/proc/'+str(setup['pid'])).exists():raise TimeoutError('Original setup cutoff')
            time.sleep(1)
        # Search path only: restores exact original numerical libraries without
        # altering old downloaded/installed evidence or a model implementation.
        pth=site/'00_original_numeric_versions.pth'
        with pth.open('x') as f:f.write('import sys;sys.path.insert(0,'+repr(str(target))+')\n')
        code="import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ('numpy','scipy')}))"
        versions=json.loads(subprocess.check_output([str(BASE/'venvs'/model/'bin/python'),'-c',code],text=True,timeout=20))
        if versions!={'numpy':'1.26.4','scipy':'1.15.3'}:raise ValueError('Original numeric namespace not selected')
        checked[model]=versions
    with (BASE/'original_numeric_compat_receipt01.json').open('x') as f:
        json.dump({'status':'original_numpy_restored_pre_model','versions':checked,'numpy_original_wheel':item,
                   'time':time.time(),'no_model_or_CUDA_import':True,'prior_common_packages_preserved':True},f,ensure_ascii=True,indent=2)
    print(json.dumps({'status':'original_numpy_namespace_ready','versions':checked}),flush=True)

if __name__=='__main__':main()
