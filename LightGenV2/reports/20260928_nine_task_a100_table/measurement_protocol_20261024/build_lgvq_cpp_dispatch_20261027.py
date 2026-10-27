"""Explicit CPU-only host C++ build, no model load/forward or CUDA query."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

HERE=Path(__file__).resolve().parent


def build():
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only build cannot inherit CUDA initialization')
    source=HERE/'lgvq_a100_cpp_dispatch_20261027.cpp'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    destination=HERE/'cpp_dispatch_build_20261027'
    destination.mkdir(exist_ok=False)
    binary=destination/'lgvq_native_dispatch_20261027.so'
    cxx=shutil.which('c++')
    if not cxx:raise RuntimeError('Existing host C++ compiler required; do not install/change environment automatically')
    torch_root=Path(torch.__file__).resolve().parent
    cuda_root=Path('/usr/local/cuda')
    if not (cuda_root/'include/cuda_runtime_api.h').exists():raise RuntimeError('Existing CUDA runtime headers required')
    includes=[torch_root/'include',torch_root/'include/torch/csrc/api/include',
              Path(sysconfig.get_paths()['include']),cuda_root/'include']
    command=[cxx,'-O3','-std=c++17','-shared','-fPIC',
        '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    for name in ('COMPILER_TYPE','STDLIB','BUILD_ABI'):
        command.append('-DPYBIND11_'+name+'="'+getattr(torch._C,'_PYBIND11_'+name)+'"')
    command.extend('-I'+str(p) for p in includes)
    command.extend([str(source),'-L'+str(torch_root/'lib'),'-L'+str(cuda_root/'lib64'),
        '-Wl,-rpath,'+str(torch_root/'lib'),'-Wl,-rpath,'+str(cuda_root/'lib64'),
        '-ltorch_cuda','-lc10_cuda','-lcudart','-ltorch_python','-ltorch_cpu','-ltorch','-lc10','-o',str(binary)])
    result=subprocess.run(command,text=True,capture_output=True,timeout=180)
    with (destination/'compiler.log').open('x') as stream:stream.write(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Host build failed; preserve compiler.log: '+str(result.returncode))
    spec=importlib.util.spec_from_file_location('lgvq_native_dispatch_20261027',binary)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import time
    a=time.perf_counter();cpp=module.now_seconds();b=time.perf_counter()
    if not a<=cpp<=b:raise ValueError('C++ and Python clocks differ; no shift allowed')
    if torch.cuda.is_initialized() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Source changed or build created a CUDA context')
    receipt=dict(source_sha256=digest,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        compiler_returncode=result.returncode,CUDA_context_created_during_build=False,
        compiler=cxx,command=command,torch_version=torch.__version__,clock_bracket=[a,b],cpp_clock_observation=cpp,
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (destination/'build_identity.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    return receipt


if __name__=='__main__':
    result=build()
    print(json.dumps({k:v for k,v in result.items() if k!='command'},ensure_ascii=True))
