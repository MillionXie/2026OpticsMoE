"""Bounded new-rental Python setup; preserve old partials, exact official wheels.

No CUDA/model import or inference. Eight exact HTTPS ranges across wheel files,
whole original PyPI SHA required, then local installation. No retry loop.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import venv

BASE=Path('/root/data/lgvq_20261029')
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
OLD=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_full_deepseek_kimi_rental_20261017'
FLASH='flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'
FLASH_SHA='7e0b07913d56782d0f3f2ee76bd39587557742628983f6ab9ec2527c99437476'
PINS={'torch':'2.6.0','torchvision':'0.21.0','numpy':'1.26.4',
 'nvidia-cuda-nvrtc-cu12':'12.4.127','nvidia-cuda-runtime-cu12':'12.4.127',
 'nvidia-cuda-cupti-cu12':'12.4.127','nvidia-cudnn-cu12':'9.1.0.70',
 'nvidia-cublas-cu12':'12.4.5.8','nvidia-cufft-cu12':'11.2.1.3',
 'nvidia-curand-cu12':'10.3.5.147','nvidia-cusolver-cu12':'11.6.1.9',
 'nvidia-cusparse-cu12':'12.3.1.170','nvidia-cusparselt-cu12':'0.6.2',
 'nvidia-nccl-cu12':'2.21.5','nvidia-nvtx-cu12':'12.4.127',
 'nvidia-nvjitlink-cu12':'12.4.127','triton':'3.2.0'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def write(p,j):
    with p.open('x') as f:json.dump(j,f,ensure_ascii=True,indent=2)

def get_json(url):
    with urllib.request.urlopen(url,timeout=25) as r:return json.load(r)

def wheel_record(item):
    name,version=item
    j=get_json(f'https://pypi.org/pypi/{name}/{version}/json')
    candidates=[f for f in j['urls'] if f['packagetype']=='bdist_wheel'
                and not f['yanked'] and f['filename'].endswith('x86_64.whl')
                and ('manylinux' in f['filename'] or 'linux_x86_64' in f['filename'])
                and ('-cp312-cp312-' in f['filename'] or '-py3-none-' in f['filename'])]
    # Torch/numpy may offer distinct manylinux ABI builds. This exact platform
    # is cp312 Linux x86_64; require a unique official compatible wheel.
    if len(candidates)!=1:raise ValueError((name,'nonunique exact wheel',len(candidates)))
    f=candidates[0]
    return dict(package=name,version=version,name=f['filename'],bytes=f['size'],
                sha256=f['digests']['sha256'],url=f['url'].replace('https://files.pythonhosted.org/',
                'https://mirrors.aliyun.com/pypi/'),official_json=f'https://pypi.org/pypi/{name}/{version}/json')

def download_all(records,deadline):
    cache=BASE/'python_wheels03';cache.mkdir(exist_ok=True)
    jobs=[]
    for r in records:
        dst=cache/r['name']
        if dst.exists():
            if dst.stat().st_size!=r['bytes'] or sha(dst)!=r['sha256']:raise ValueError('Existing wheel differs')
            continue
        # Reuse complete old downloads only after independent official hash.
        candidates=list(BASE.glob('tmp/pip-unpack-*/'+r['name']))
        valid=next((p for p in candidates if p.stat().st_size==r['bytes'] and sha(p)==r['sha256']),None)
        if valid:
            os.link(valid,dst);continue
        parts=cache/(r['name']+'.ranges');parts.mkdir(exist_ok=False)
        width=(r['bytes']+7)//8
        jobs.extend((r,parts,i,i*width,min(r['bytes'],(i+1)*width)-1) for i in range(8) if i*width<r['bytes'])
    def segment(job):
        r,parts,i,start,end=job
        if time.time()>=deadline:raise TimeoutError('Finite Python setup deadline')
        req=urllib.request.Request(r['url'],headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=40) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f"bytes {start}-{end}/{r['bytes']}":
                raise ValueError('Exact wheel HTTPS range required')
            dst=parts/f'{i:02d}.bin'
            with dst.open('xb') as out:
                while True:
                    if time.time()>=deadline:raise TimeoutError('Finite wheel deadline')
                    b=response.read(1048576)
                    if not b:break
                    out.write(b)
            if dst.stat().st_size!=end-start+1:raise ValueError('Wheel range size differs')
        print(json.dumps({'range_complete':r['name'],'range':i}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(segment,jobs))
    for r in records:
        dst=cache/r['name']
        if not dst.exists():
            with dst.open('xb') as out:
                for p in sorted((cache/(r['name']+'.ranges')).glob('*.bin')):
                    with p.open('rb') as src:shutil.copyfileobj(src,out,4194304)
        if dst.stat().st_size!=r['bytes'] or sha(dst)!=r['sha256']:raise ValueError('Whole official wheel SHA differs')
    return [str(cache/r['name']) for r in records]

def run(command,deadline,**kw):
    subprocess.run(command,check=True,timeout=max(1,min(3600,deadline-time.time())),**kw)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    if not 0<a.deadline-time.time()<=7200:raise ValueError('Finite setup <=two hours')
    if sys.version_info[:2]!=(3,12) or REPO!=BASE/'repo':raise ValueError('Exact rental Python/repo only')
    if (REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/rental_fullscope_20261029_01').exists():
        raise ValueError('Do not change an active/sealed measurement environment')
    for pid in (701,710):
        if Path(f'/proc/{pid}').exists():raise ValueError('Original installer still active')
    receipt=BASE/'python_setup_receipt03.json'
    if receipt.exists():raise FileExistsError('Preserve completed setup receipt')
    with concurrent.futures.ThreadPoolExecutor(4) as pool:records=list(pool.map(wheel_record,PINS.items()))
    meta=get_json('https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.7.4.post1')
    f=next(f for f in meta['assets'] if f['id']==224784686)
    if (f['name'],f['size'])!=(FLASH,187826523):raise ValueError('Original official Flash wheel identity')
    flash=dict(package='flash-attn',version='2.7.4.post1',name=FLASH,bytes=187826523,
               sha256=FLASH_SHA,url=f['browser_download_url'],official_json=meta['url'])
    write(BASE/'python_wheel_manifest03.json',records+[flash])
    wheels=download_all(records+[flash],a.deadline)
    common=BASE/'venvs/common';common.parent.mkdir(exist_ok=True)
    if common.exists():raise FileExistsError('No blind install over prior environment')
    venv.EnvBuilder(with_pip=True).create(common)
    env=dict(os.environ,TMPDIR=str(BASE/'tmp'),PIP_DISABLE_PIP_VERSION_CHECK='1')
    py=str(common/'bin/python')
    run([py,'-m','pip','install','--no-deps',*wheels[:-1]],a.deadline,env=env)
    # GPU/backend versions above are frozen. These non-backend utilities are
    # resolved once, then freeze and wheel/source identity are preserved.
    utilities=['opencv-python==4.11.0.86','accelerate==1.12.0','huggingface-hub==0.36.0',
               'sympy==1.13.1','setuptools','packaging','ninja','scipy','psutil','Pillow',
               'tiktoken','einops','timm==1.0.22','nvidia-ml-py','sentencepiece','attrdict','easydict']
    run([py,'-m','pip','install','--retries','0','--timeout','20','--no-cache-dir',
         '-i','https://mirrors.aliyun.com/pypi/simple',*utilities],a.deadline,env=env)
    common_site=common/'lib/python3.12/site-packages'
    for model,tf in [('kimi','4.48.2'),('deepseek','4.38.2')]:
        folder=BASE/'venvs'/model
        if folder.exists():raise FileExistsError('Do not overwrite model environment')
        venv.EnvBuilder(with_pip=True).create(folder)
        (folder/'lib/python3.12/site-packages/shared_original_torch.pth').write_text(str(common_site)+'\n')
        mp=str(folder/'bin/python')
        run([mp,'-m','pip','install','--retries','0','--timeout','20','--no-cache-dir',
             '-i','https://mirrors.aliyun.com/pypi/simple','transformers=='+tf],a.deadline,env=env)
        if model=='kimi':run([mp,'-m','pip','install','--no-deps',wheels[-1]],a.deadline,env=env)
        else:run([mp,'-m','pip','install','--retries','0','--timeout','20','--no-cache-dir',
                  '--no-deps','-i','https://mirrors.aliyun.com/pypi/simple','xformers==0.0.29.post2'],a.deadline,env=env)
        run([mp,'-m','pip','freeze','--all'],a.deadline,stdout=(BASE/(model+'_pip_freeze03.txt')).open('x'))
    write(receipt,{'status':'packages_installed_not_formal_GPU_proof','completed':time.time(),
                   'wheels':records+[flash],'original_failed_install_preserved':True,
                   'no_CUDA_or_model_import':True})

if __name__=='__main__':main()
