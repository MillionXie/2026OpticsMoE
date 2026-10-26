"""Isolated tokenizer-only RTX4090 experiment; never loads neural model weights."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,os,pathlib,shutil,statistics,subprocess,sys,tarfile,time,urllib.request
P=pathlib.Path
HERE=P(__file__).resolve().parent;REPO=P(os.environ.get('TOKENIZER_PROBE_REPO',str(HERE.parents[3]))).resolve()
GPU='GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d'
BLOCKED='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
DEPS=P('/DATA/DATA1/guest3/.cache/tokenizer_gpu_probe_20261026')

def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def save(p,d):
    with P(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def run(cmd,**kw):
    p=subprocess.run([str(x) for x in cmd],text=True,capture_output=True,timeout=kw.pop('timeout',180),**kw)
    if p.returncode:raise RuntimeError(json.dumps({'command':list(map(str,cmd)),'returncode':p.returncode,'stdout':p.stdout[-8000:],'stderr':p.stderr[-8000:]}))
    return p.stdout
def request(url):
    with urllib.request.urlopen(url,timeout=25) as f:return f.read()
def processes():
    text=run(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid,process_name,gpu_uuid,used_memory','--format=csv,noheader,nounits'],timeout=10)
    return [x.strip().split(',') for x in text.splitlines() if x.strip()]
def check(own=False):
    foreign=[x for x in processes() if x[2].strip()==GPU and (not own or int(x[0])!=os.getpid())]
    if foreign:raise RuntimeError('Non-A100 card acquired by another process: '+json.dumps(foreign))
    text=run(['nvidia-smi','--id='+GPU,'--query-gpu=name,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],timeout=10)
    if not own and float(text.split(',')[2])>128:raise RuntimeError('Card memory is not idle: '+text)
    return text.strip()
def download(url,target,digest=None):
    target=P(target)
    if not target.exists():
        part=target.with_suffix(target.suffix+'.part');assert not part.exists()
        with urllib.request.urlopen(url,timeout=30) as f,part.open('xb') as out:
            shutil.copyfileobj(f,out)
        if digest and sha(part)!=digest:raise ValueError('Downloaded SHA mismatch: '+url)
        part.rename(target)
    if digest and sha(target)!=digest:raise ValueError('Cached dependency SHA mismatch')
    return {'url':url,'bytes':target.stat().st_size,'sha256':sha(target)}
def extract(archive,destination):
    destination=P(destination)
    if destination.exists():return
    destination.mkdir()
    with tarfile.open(archive) as t:
        for x in t.getmembers():
            if not (destination/x.name).resolve().is_relative_to(destination.resolve()):raise ValueError('Archive path escape')
        t.extractall(destination,filter='data')
def setup(root):
    assert not (root/'setup.json').exists()
    DEPS.mkdir(parents=True,exist_ok=True);downloads=DEPS/'downloads';downloads.mkdir(exist_ok=True)
    receipts=[];cuda=DEPS/'cuda';cuda.mkdir(exist_ok=True)
    manifest=json.loads(request('https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json'))
    for name in ['cuda_nvcc','cuda_cudart','cuda_cccl','cuda_nvtx']:
        if name not in manifest:continue
        entry=manifest[name]['linux-x86_64'];archive=downloads/(name+'.tar.xz')
        receipts.append(download('https://developer.download.nvidia.com/compute/cuda/redist/'+entry['relative_path'],archive,entry['sha256']))
        dest=DEPS/'components'/name;dest.parent.mkdir(exist_ok=True);extract(archive,dest)
        folders=list(dest.iterdir());assert len(folders)==1;component=folders[0]
        # Overlay only new dependency symlinks, never existing host or environment files.
        for f in component.rglob('*'):
            rel=f.relative_to(component);target=cuda/rel
            if f.is_dir():target.mkdir(parents=True,exist_ok=True)
            elif not target.exists():target.symlink_to(f)
    env=dict(os.environ,PYTHONPATH=str(DEPS/'python'),CUDA_HOME=str(cuda),CUDACXX=str(cuda/'bin/nvcc'),
             CUDAToolkit_ROOT=str(cuda),CMAKE_BUILD_PARALLEL_LEVEL='2')
    env['PATH']=str(cuda/'bin')+':'+str(DEPS/'python/bin')+':'+env['PATH']
    pydeps=DEPS/'python'
    if not (pydeps/'pybind11').exists():
        run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input','--target',pydeps,'--no-deps',
             'cmake==3.30.9','ninja==1.11.1.4','pybind11==2.13.6'],timeout=180)
    cmake=pydeps/'cmake/data/bin/cmake';ninja=pydeps/'ninja/data/bin/ninja'
    utf_archive=downloads/'utf8proc-v2.10.0.tar.gz'
    receipts.append(download('https://codeload.github.com/JuliaStrings/utf8proc/tar.gz/refs/tags/v2.10.0',utf_archive))
    extract(utf_archive,DEPS/'utf8proc_source');utf_src=next((DEPS/'utf8proc_source').iterdir())
    utf_build=DEPS/'utf8proc_build'
    run([cmake,'-S',utf_src,'-B',utf_build,'-G','Ninja','-DCMAKE_MAKE_PROGRAM='+str(ninja),'-DCMAKE_BUILD_TYPE=Release',
         '-DBUILD_SHARED_LIBS=ON','-DUTF8PROC_ENABLE_TESTING=OFF'],env=env)
    run([cmake,'--build',utf_build,'-j','2'],env=env)
    library=DEPS/'GraphTok'
    if not library.exists():run(['git','clone','--depth','1','https://github.com/Msiavashi/GraphTok.git',library],timeout=120)
    commit=run(['git','rev-parse','HEAD'],cwd=library).strip()
    build=DEPS/'graphtok_build';pybind=pydeps/'pybind11/share/cmake/pybind11'
    nvtx=next(iter((cuda/'lib').glob('libnvToolsExt.so*')),None)
    if nvtx is None:nvtx=next(iter((cuda/'lib64').glob('libnvToolsExt.so*')),None)
    if nvtx is None:
        candidate=P('/usr/local/cuda-11.8/lib64/libnvToolsExt.so')
        if candidate.exists():nvtx=candidate
    cmd=[cmake,'-S',library,'-B',build,'-G','Ninja','-DCMAKE_MAKE_PROGRAM='+str(ninja),'-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_CUDA_COMPILER='+str(cuda/'bin/nvcc'),'-DCUDAToolkit_ROOT='+str(cuda),'-DCMAKE_CUDA_ARCHITECTURES=89',
         '-DGBPE_VOCABS=QWEN25','-DGBPE_GIGATOKEN=OFF','-DGBPE_BUILD_TESTS=OFF','-DGBPE_BUILD_PYTHON=ON',
         '-DPython3_EXECUTABLE='+sys.executable,'-Dpybind11_DIR='+str(pybind),'-DUTF8PROC_INCLUDE_DIR='+str(utf_src),
         '-DUTF8PROC_LIB='+str(utf_build/'libutf8proc.so')]
    if nvtx:cmd+=['-DGBPE_NVTOOLSEXT_LIBRARY='+str(nvtx)]
    config=run(cmd,env=env);buildlog=run([cmake,'--build',build,'--target','gpu_bpe_tokenizer_py','-j','2'],env=env,timeout=300)
    (root/'build.log').write_text(config+'\n'+buildlog)
    extensions=list(build.glob('_gpu_bpe_tokenizer*.so'));assert len(extensions)==1
    save(root/'setup.json',{'status':'complete','third_party_commit':commit,'extension':str(extensions[0]),
         'extension_sha256':sha(extensions[0]),'cuda_compiler':run([cuda/'bin/nvcc','--version'],env=env),
         'download_receipts':receipts,'existing_environment_unchanged':True,'cpu_build_jobs':2})

def bench(root):
    import threading
    assert not (root/'result.json').exists()
    setup_data=json.loads((root/'setup.json').read_text());sys.path.insert(0,str(DEPS/'graphtok_build'))
    import _gpu_bpe_tokenizer as gbpe
    import torch,numpy as np
    from transformers import AutoTokenizer
    for _ in range(3):check();time.sleep(1)
    torch.set_num_threads(2)
    if hasattr(os,'sched_setaffinity'):
        available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(available[-2:]))
    binding_path=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/smoke/lgvq_frontend_identity_recovery_20261024_01/front_parameter_binding_02.json'
    binding=json.loads(binding_path.read_text());model=P(binding['model_candidate']);vocab=model/'tokenizer.json'
    stop=threading.Event();errors=[];observations=[]
    def monitor():
        while not stop.is_set():
            try:observations.append({'unix':time.time(),'gpu':check(own=True)})
            except BaseException as e:errors.append(repr(e));return
            stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    gpu_name=torch.cuda.get_device_name(0);assert '4090' in gpu_name
    hf=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    started=time.perf_counter();gpu=gbpe.Tokenizer(str(vocab),host_max_bytes=0,max_input_chars=65536);torch.cuda.synchronize()
    initialization_s=time.perf_counter()-started
    assert gpu.host_max_bytes==0
    prompts={x['target']:x['rendered_chat_template'] for x in binding['matched_prompt_checks']}
    fixtures=[prompts['temporal'],prompts['spatial'],'你好，测试视频的时间一致性。','Cafe\u0301\n12345','',"We'll rate it: Good.\r\n"]
    def cpu(texts,device=False):
        v=hf(texts,padding=True,return_tensors='pt',add_special_tokens=False)
        out={k:v[k] for k in ('input_ids','attention_mask')}
        return {k:t.to('cuda') for k,t in out.items()} if device else out
    def gpu_call(texts,device=False):
        if device:
            flat,offsets=gpu.encode_batch_to_device(texts)
            lengths=np.diff(offsets);maxlen=int(lengths.max(initial=0));rows=[];masks=[]
            if np.all(lengths==maxlen):
                ids=flat.reshape(len(texts),maxlen).to(torch.int64)
                return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
            for i,n in enumerate(lengths):
                n=int(n);ids=flat[int(offsets[i]):int(offsets[i+1])].to(torch.int64)
                pad=torch.full((maxlen-n,),hf.pad_token_id,dtype=torch.int64,device='cuda')
                mask=torch.cat((torch.zeros(maxlen-n,dtype=torch.int64,device='cuda'),torch.ones(n,dtype=torch.int64,device='cuda')))
                rows.append(torch.cat((pad,ids)) if hf.padding_side=='left' else torch.cat((ids,pad)))
                masks.append(mask if hf.padding_side=='left' else mask.flip(0))
            return {'input_ids':torch.stack(rows),'attention_mask':torch.stack(masks)}
        flat,offsets=gpu.encode_batch(texts);lengths=np.diff(offsets)
        if len(lengths) and np.all(lengths==lengths[0]):
            ids=flat.astype(np.int64).reshape(len(texts),int(lengths[0]))
            return {'input_ids':torch.from_numpy(ids),'attention_mask':torch.from_numpy(np.ones_like(ids))}
        rows=[flat[int(offsets[i]):int(offsets[i+1])].tolist() for i in range(len(texts))]
        maxlen=max(map(len,rows),default=0);ids=[];masks=[]
        for row in rows:
            n=len(row);pad=[hf.pad_token_id]*(maxlen-n)
            ids.append(pad+row if hf.padding_side=='left' else row+pad)
            masks.append([0]*(maxlen-n)+[1]*n if hf.padding_side=='left' else [1]*n+[0]*(maxlen-n))
        return {'input_ids':torch.tensor(ids,dtype=torch.int64),'attention_mask':torch.tensor(masks,dtype=torch.int64)}
    # Same tokenizer asset and a small, relevant exactness check, no neural inference.
    for device in (False,True):
        for text in fixtures:
            a=cpu([text],device);b=gpu_call([text],device);torch.cuda.synchronize()
            assert all(torch.equal(a[k],b[k]) for k in a),(text,device)
    sample_ids=hf([prompts['temporal']],add_special_tokens=False)['input_ids'][0]
    for b in (16,256):
        texts=[prompts['temporal']]*b;a=cpu(texts);z=gpu_call(texts);assert all(torch.equal(a[k],z[k]) for k in a)
    samples=[];results=[]
    cases=[('actual_prompt_B1',[prompts['temporal']],200),('same_prompt_B16',[prompts['temporal']]*16,60),
           ('same_prompt_B256',[prompts['temporal']]*256,15),
           ('synthetic_long_B1',[prompts['temporal']*256],15)]
    try:
        for case,texts,count in cases:
            for device in (False,True):
                ref=cpu(texts,device);actual=gpu_call(texts,device);torch.cuda.synchronize()
                assert all(torch.equal(ref[k],actual[k]) for k in ref),(case,device)
                measurements={}
                for label,fn in [('CPU_AutoTokenizer',cpu),('GPU_GraphTok',gpu_call)]:
                    cold=time.perf_counter();fn(texts,device);torch.cuda.synchronize();first_ms=(time.perf_counter()-cold)*1000
                    for _ in range(5):fn(texts,device)
                    torch.cuda.synchronize();durations=[]
                    for repeat in range(3):
                        for i in range(count):
                            if errors:raise RuntimeError(errors[0])
                            torch.cuda.synchronize();start=time.perf_counter_ns();fn(texts,device)
                            if device or label=='GPU_GraphTok':torch.cuda.synchronize()
                            elapsed=(time.perf_counter_ns()-start)/1e6;durations.append(elapsed)
                            samples.append({'case':case,'output':'GPU_int64_tensors' if device else 'CPU_int64_tensors','method':label,'repeat':repeat+1,'iteration':i+1,'ms':elapsed})
                    measurements[label]={'mean_ms':statistics.mean(durations),'median_ms':statistics.median(durations),'first_observed_ms':first_ms,'timed_calls':len(durations)}
                results.append({'case':case,'batch_size':len(texts),'bytes_per_text':len(texts[0].encode()),'output':'GPU_int64_tensors' if device else 'CPU_int64_tensors',
                    **measurements,'CPU_over_GPU_time_ratio':measurements['CPU_AutoTokenizer']['mean_ms']/measurements['GPU_GraphTok']['mean_ms']})
        check(own=True)
    finally:stop.set();thread.join(timeout=12)
    if errors:raise RuntimeError(errors[0])
    with (root/'timing_samples.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=samples[0].keys());w.writeheader();w.writerows(samples)
    save(root/'result.json',{'status':'complete','gpu_uuid':GPU,'gpu_name':gpu_name,'CPU_model':run(['lscpu']),
         'tokenizer_model_path':str(model),'tokenizer_sha256':sha(vocab),'binding_sha256':sha(binding_path),'prompt_token_ids':sample_ids,
         'original_prompt_tokens':len(sample_ids),'normalization':'original Qwen NFC remains on CPU in GraphTok; GPU pretokenization+BPE+assembly',
         'GPU_fallback_disabled':True,'extra_neural_forwards':0,'initialization_s':initialization_s,'warmup_calls_per_case_method':5,
         'scope':'same raw strings to equal padded int64 input_ids and attention_mask; includes required transfers and synchronization',
         'synthetic_long_case_is_not_original_LGVQ':True,'CPU_affinity':sorted(os.sched_getaffinity(0)),'system_load':os.getloadavg(),
         'versions':{x:importlib.metadata.version(x) for x in ('torch','transformers','tokenizers','numpy')},
         'third_party_commit':setup_data['third_party_commit'],'source_sha256':sha(__file__),'source_commit':os.environ.get('TOKENIZER_PROBE_SOURCE_COMMIT'),
         'repository_HEAD':run(['git','rev-parse','HEAD'],cwd=REPO).strip(),
         'device_observations':observations,'results':results})
    print(json.dumps({'status':'complete','results':results}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('setup','bench'),required=True);p.add_argument('--output',type=P,required=True);a=p.parse_args();root=a.output.resolve()
    assert root.is_relative_to(REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation')
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',RAYON_NUM_THREADS='2',TOKENIZERS_PARALLELISM='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    os.environ.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None);os.environ['CUDA_MPS_PIPE_DIRECTORY']=str(DEPS/'no_mps')
    root.mkdir(parents=True,exist_ok=True)
    try:
        print(json.dumps({'phase':a.phase,'gpu':GPU,'idle':check()}),flush=True)
        (setup if a.phase=='setup' else bench)(root)
    except BaseException as e:
        save(root/(a.phase+'_failure.json'),{'error':repr(e),'unix':time.time(),'source_sha256':sha(__file__)});raise
if __name__=='__main__':main()
