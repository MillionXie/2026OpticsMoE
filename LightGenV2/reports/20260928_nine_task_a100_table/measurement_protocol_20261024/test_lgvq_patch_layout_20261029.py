"""CPU exact native patch/cast equality, no model or CUDA initialization."""
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import lgvq_compute_scope_20261029 as scope
from lgvq_patch_layout_20261029 import strided_patch

def main():
    import torch
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    _,native,receipt=scope.processor_adapter(Qwen2VLImageProcessorFast._preprocess,lambda *a:None)
    torch.set_num_threads(1)
    for side in (32,448):
        values=torch.arange(4*3*side*side,dtype=torch.float32).reshape(4,3,side,side)
        values=values.sin()*3.731-1.135
        expected=native(values,side,side,2,16,2).reshape(-1,1536).to(torch.bfloat16)
        actual=strided_patch(values,side,side,2,16,2)
        if not torch.equal(actual.view(torch.int16),expected.view(torch.int16)):raise ValueError('Patch/cast exactness failed')
    if torch.cuda.is_initialized():raise ValueError('CPU test initialized CUDA')
    print(json.dumps(dict(status='CPU_exact_native_patch_then_cast_bytes_equal',cases=2,
        no_model_forward=True,CUDA_initialized=False,processor=receipt)))

if __name__=='__main__':main()
