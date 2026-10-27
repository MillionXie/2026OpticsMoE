"""Cache the SAME Qwen's native vision-merger tokens, not a second encoder.

Used only for frozen-compute training acceleration. Raw inference captures this
output in the full Qwen-VL forward and still runs all 28 language layers.
"""
import argparse,gc,json,os,time
from pathlib import Path

import torch

from .qwen_vl_turbo_cache import save_with_digest
from .qwen_vl_turbo_data import sha256,write_json
from .qwen_vl_turbo_run import git_identity,tensor_image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('qwen','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--cache-run',type=Path,help='Completed reviewed cache; no model checkpoint needed')
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise ValueError('Unauthorized GPU')
    if (args.output/'report.json').exists():raise ValueError('Cache already complete; reuse instead of rerunning')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20,0);device=torch.device('cuda:0')
    if bool(args.parent)==bool(args.cache_run):raise ValueError('Provide exactly one parent or cache-run')
    cache=(args.cache_run or args.parent)/'frozen_features.pt'
    if args.cache_run:
        ready=json.loads((args.cache_run/'cache_complete.json').read_text())
        if not ready['complete']:raise ValueError('Reviewed cache incomplete')
        source_digest=ready['sha256']
    else:
        parent=torch.load(args.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        source_digest=parent['cache_sha256']
    if sha256(cache)!=source_digest:raise ValueError('Parent frozen input cache changed')
    data=torch.load(cache,map_location='cpu',weights_only=False,mmap=True)
    identity={'git':git_identity(),'script_sha256':sha256(Path(__file__)),'qwen':str(args.qwen),
       'source_cache_sha256':source_digest,'gpu_uuid':args.gpu_uuid,'started_unix':time.time(),
       'scope':'Native Qwen vision output for exact cached reference RGB; no target image input; no optimization'}
    write_json(args.output/'identity.json',identity)
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    full=Qwen3VLForConditionalGeneration.from_pretrained(args.qwen,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True)
    vision=full.model.visual
    del full;gc.collect()
    vision=vision.to(device).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(args.qwen,local_files_only=True,min_pixels=65536,max_pixels=65536)
    features=[]
    with torch.inference_mode():
        for start in range(0,len(data['rows']),args.batch_size):
            ids=list(range(start,min(start+args.batch_size,len(data['rows']))))
            images=[tensor_image(data['reference'][i]) for i in ids]
            inputs=processor.image_processor(images=images,return_tensors='pt').to(device)
            if inputs['image_grid_thw'].tolist()!=[[1,16,16]]*len(ids):raise ValueError('Unexpected native vision image grid')
            output,_=vision(inputs['pixel_values'].to(vision.dtype),grid_thw=inputs['image_grid_thw'])
            if output.shape!=(64*len(ids),2048):raise ValueError('Native visual merger tensor shape differs')
            features.extend(output.detach().bfloat16().cpu().split(64))
            if (start//args.batch_size)%10==0:
                print('native vision',len(features),'/',len(data['rows']),flush=True)
                write_json(args.output/'progress.json',{'done':len(features),'total':len(data['rows'])})
    saved={'source_cache_sha256':source_digest,'sample_ids':[r['sample_id'] for r in data['rows']],
       'vision':features,'grid':[8,8],'source':'Qwen3-VL native visual merger before language layers',
       'identity':identity}
    digest=save_with_digest(saved,args.output/'native_vision.pt')
    write_json(args.output/'report.json',{'complete':True,'samples':len(features),'sha256':digest,
       'file':'native_vision.pt','source_cache_sha256':source_digest,
       'vision_parameters':sum(p.numel() for p in vision.parameters()),
       'note':'This cache does not replace full Qwen in deployment; native vision and fused language states come from one full Qwen forward at inference.'})


if __name__=='__main__':main()
