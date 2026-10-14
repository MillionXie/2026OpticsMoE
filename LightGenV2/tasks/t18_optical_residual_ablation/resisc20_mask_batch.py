"""Batched optics with scalar-equivalent fixed detector reductions, bounded probe only."""
import argparse,os,sys,json
from pathlib import Path
import torch
import resisc20_batch_probe as probe
# Keep the exact B1 reduction used by the running model for each image. Batched
# FFT, phase modulation and per-image OEO remain in the frozen graph unchanged.
ORIGINAL_EINSUM=torch.einsum

def scalar_mask_einsum(equation,*args,**kwargs):
 if equation=='bhw,khw->bk' and len(args)==2 and args[0].ndim==3 and len(args[0])>1:
  return torch.cat([ORIGINAL_EINSUM(equation,args[0][k:k+1],args[1],**kwargs) for k in range(len(args[0]))])
 return ORIGINAL_EINSUM(equation,*args,**kwargs)

def install():
 assert torch.einsum in (ORIGINAL_EINSUM,scalar_mask_einsum)
 torch.einsum=scalar_mask_einsum

def sources():
 src=probe.s.sources()
 for name in ('resisc20_batch_probe.py','resisc20_mask_batch.py'):
  p=probe.HERE/name;src[p.relative_to(probe.ROOT).as_posix()]=probe.s.sha(p)
 return src
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert not a.out.exists();install();probe.sources=sources
 original_save=probe.s.save
 def save(path,value,exclusive=False):
  if Path(path).name=='checks.json':
   value.update(execution_change='Batch optics, scalar-equivalent fixed router/detector mask reductions per image',original_micro1_unchanged=True,physical_graph_unchanged=True,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision())
  return original_save(path,value,exclusive=exclusive)
 probe.s.save=save
 try:print(json.dumps(probe.run(a)))
 except BaseException as e:
  if a.out.exists():original_save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
  raise
