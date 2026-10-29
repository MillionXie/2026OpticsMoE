"""Single finite readiness gate, then original strict twelve-point controller.

No model smoke, training, retry or second suite. Package/data preparation must
finish first; formal inference starts immediately after CPU identity checks.
"""
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import run_rental_fullscope_20261029 as entry

BASE=Path('/root/data/lgvq_20261029')
DEADLINE=1793278441.7376518

def write(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=True,indent=2)

def run(command,**kw):
    return subprocess.run(command,check=True,timeout=min(90,max(1,DEADLINE-time.time())),**kw)

def symlink(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.resolve()!=target.resolve():raise ValueError('Existing path differs: '+str(path))
    else:path.symlink_to(target,target_is_directory=True)

def main():
    if entry.REPO!=BASE/'repo' or time.time()>=DEADLINE-120:raise ValueError('Exact rental/finite deadline only')
    marker=BASE/'readiness_launcher_started01.json'
    if marker.exists() or entry.ROOT.exists():raise FileExistsError('Never launch a second canonical chain')
    commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT']
    write(marker,{'pid':os.getpid(),'started':time.time(),'deadline':DEADLINE,'source_commit':commit})
    state={'status':'waiting_for_packages_only','pid':os.getpid()}
    try:
        setup=json.loads((BASE/'python_setup_started03.json').read_text())
        ready=BASE/'python_setup_receipt03.json'
        while not ready.exists():
            if time.time()>=min(DEADLINE-120,setup['deadline']):raise TimeoutError('Fixed setup/task cutoff; no extension')
            if not Path('/proc/'+str(setup['pid'])).exists():raise RuntimeError('Package installer exited without receipt; preserve log')
            time.sleep(3)
        while Path('/proc/'+str(setup['pid'])).exists():
            if time.time()>=DEADLINE-120:raise TimeoutError('No new worker time')
            time.sleep(1)
        vendor=BASE/'vendor/DeepSeek-VL2'
        vendor_receipt=json.loads((BASE/'vendor_git_source_receipt01.json').read_text())
        if vendor_receipt['vendor_git_commit']!='ef9f91e2b6426536b83294c11742c27be66361b1':raise ValueError('Exact official vendor Git')
        old=entry.base.TASK/'runs/simulation/lgvq_full_deepseek_kimi_rental_20261017'
        for model in ('kimi','deepseek'):
            receipt=json.loads((old/model/'train/training_receipt.json').read_text())
            for name,digest in receipt['identity']['source_sha256'].items():
                data=subprocess.check_output(['git','show',receipt['identity']['git_commit']+':'+name.replace('\\','/')],cwd=entry.REPO,timeout=10)
                if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Original historical source identity')
            if json.loads((BASE/'models'/model/'asset_receipt.json').read_text())['status']!='complete':raise ValueError('Model assets unsealed')
            symlink(Path('/data/models')/model,BASE/'models'/model)
            symlink(Path('/data/venvs')/model,BASE/'venvs'/model)
        # The official DeepSeek source imports AttrDict. The Python3.12
        # compatible maintained distribution is a mapping utility, not a
        # numerical model/forward replacement. Preserve exact installed source.
        dp=BASE/'venvs/deepseek/bin/python'
        run([str(dp),'-m','pip','install','--no-deps','--retries','0','--timeout','20',
             '--no-cache-dir','-i','https://mirrors.aliyun.com/pypi/simple','attrdict3==2.0.2'])
        pth=BASE/'venvs/deepseek/lib/python3.12/site-packages/official_deepseek_git.pth'
        with pth.open('x') as f:f.write(str(vendor)+'\n')
        expected=json.loads((old/'deepseek/train/training_receipt.json').read_text())['identity']['runtime']['official_runtime_source_sha256']
        for name,digest in expected.items():
            if hashlib.sha256((vendor/'deepseek_vl2/models'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Official DeepSeek source differs')
        checks={}
        for model in ('kimi','deepseek'):
            py=BASE/'venvs'/model/'bin/python'
            expected=json.loads((old/model/'train/training_receipt.json').read_text())['identity']['runtime']['packages']
            code="import importlib.metadata as m,json;names=('torch','torchvision','transformers','accelerate','xformers','flash-attn','numpy','opencv-python');out={};\nfor n in names:\n try:out[n]=m.version(n)\n except m.PackageNotFoundError:pass\nprint(json.dumps(out))"
            actual=json.loads(subprocess.check_output([str(py),'-c',code],text=True,timeout=20))
            if actual!=expected:raise ValueError((model,'original native packages differ',actual,expected))
            imports="import torch,cv2;assert not torch.cuda.is_initialized();"
            imports+=("import flash_attn;" if model=='kimi' else "import deepseek_vl2.models;from attrdict import AttrDict;assert AttrDict(x=1).x==1;")
            imports+="assert not torch.cuda.is_initialized();print('CPU_imports_passed_no_CUDA_initialization')"
            run([str(py),'-c',imports])
            run([str(py),str(entry.HERE/'test_rental_fullscope_20261029.py')])
            checks[model]=actual
        chunks=[json.loads((BASE/'input_data'/f'chunk{i:02d}_receipt.json').read_text()) for i in range(8)]
        if (sum(c['videos'] for c in chunks)!=512 or {c['chunk'] for c in chunks}!=set(range(8))
                or any(c['status']!='verified_original512_chunk' for c in chunks)):
            raise ValueError('All original512 data chunks required')
        uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
        if uuid!=entry.GPU:raise ValueError('Sole GPU changed; do not change binding during the chain')
        entry.configure();entry.identity.verify(commit,entry.sources(),entry.REPO)
        write(BASE/'readiness_passed01.json',{'status':'CPU_identity_passed_not_GPU_proof','time':time.time(),
              'packages':checks,'gpu_uuid':uuid,'source_commit':commit,'vendor_git_commit':vendor_receipt['vendor_git_commit']})
        manifest=old/'provisioning/original_manifest.csv'
        command=[str(BASE/'venvs/kimi/bin/python'),'-u',str(entry.ENTRY),'--phase','suite','--profile','rental',
            '--manifest',str(manifest),'--output',str(entry.ROOT),'--gpu-uuid',entry.GPU,'--deadline',str(DEADLINE)]
        state.update(status='running_single_controller',started=time.time(),command=command)
        with (BASE/'formal_controller01.log').open('x') as log:
            child=subprocess.Popen(command,cwd=entry.REPO,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['controller_pid']=child.pid
            write(BASE/'formal_controller_started01.json',state)
            try:code=child.wait(timeout=max(1,DEADLINE-time.time()+30))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise
        state.update(status='controller_exited',exit_code=code,finished=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error),finished=time.time());raise
    finally:write(BASE/'readiness_launcher_exit01.json',state)

if __name__=='__main__':main()
