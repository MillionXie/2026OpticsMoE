"""Raw reference/text inference for spatial Qwen; no dataset/GT/mask access."""
import argparse,gc,hashlib,json,os
from pathlib import Path

import torch
from PIL import Image

from .qwen_vl_spatial import build_spatial_head
from .qwen_vl_turbo_data import sha256,write_json
from .qwen_vl_turbo_model import encode_reference,pack_hidden


@torch.no_grad()
def raw_hidden(qwen,processor,image,prompt,layout,device):
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=processor(text=[text],images=[image],return_tensors='pt')
    positions=torch.where(inputs['input_ids'][0]==layout['image_token_id'])[0].tolist()
    if positions!=layout['positions'] or inputs['image_grid_thw'].tolist()!=[layout['unmerged_grid']]:
        raise ValueError('Raw image token layout differs from the verified spatial grid')
    return encode_reference(qwen,processor,image,prompt,device)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--prompt')
    parser.add_argument('--manifest',type=Path,help='Input-only JSON list of image/prompt/sample_id; no GT fields')
    parser.add_argument('--ablate',action='store_true',help='Also gray the Qwen image and permute matched-group text')
    parser.add_argument('--reference-reconstruction',action='store_true',help='Export TRAIN-learned Qwen-to-reference latent auxiliary readout as a separately labelled diagnostic')
    for name in ('checkpoint','qwen','sd','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,default=1042)
    args=parser.parse_args()
    if args.manifest:
        records=json.loads(args.manifest.read_text(encoding='utf-8'))
        allowed={'image','prompt','sample_id','category','mode'}
        if any(set(r)-allowed for r in records):raise ValueError('Inference manifest may contain only input/identity fields')
    else:
        if args.image is None or args.prompt is None:raise ValueError('Provide RGB/text or an input-only manifest')
        records=[{'image':str(args.image),'prompt':args.prompt,'sample_id':'image_0000'}]
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state.get('schema')!='qwen_spatial_v1':raise ValueError('Not a spatial Qwen checkpoint')
    cfg=state['config'];layout=state['visual_layout']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid']:raise ValueError('Unauthorized GPU UUID')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'],0)
    device=torch.device('cuda:0')
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    qwen=Qwen3VLForConditionalGeneration.from_pretrained(args.qwen,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to(device).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(args.qwen,local_files_only=True,min_pixels=65536,max_pixels=65536)
    groups={}
    for i,r in enumerate(records):groups.setdefault((r.get('category'),r.get('mode')),[]).append(i)
    partners={i:group[(j+1)%len(group)] for group in groups.values() for j,i in enumerate(group)}
    features=[]
    for i,r in enumerate(records):
        with Image.open(r['image']) as handle:image=handle.convert('RGB')
        if image.size!=(256,256):raise ValueError('Current protocol requires native 256x256 reference RGB')
        variants=[('normal',image,r['prompt'])]
        if args.ablate:
            variants.extend([('gray_vision',Image.new('RGB',image.size,(127,127,127)),r['prompt']),
                             ('shuffled_text',image,records[partners[i]]['prompt'])])
        for kind,rgb,prompt in variants:
            hidden=raw_hidden(qwen,processor,rgb,prompt,layout,device)
            features.append((i,kind,prompt,hidden))
        print('raw Qwen',i+1,'/',len(records),flush=True)
    del qwen,processor;gc.collect();torch.cuda.empty_cache()
    model=build_spatial_head(args.sd,state,cfg,layout['positions'],device,
        drop_vae_encoder=not args.reference_reconstruction).eval().requires_grad_(False)
    if args.reference_reconstruction:
        model.vae.encoder=None;model.vae.quant_conv=None
    args.output.mkdir(parents=True,exist_ok=True);reports=[]
    for i,kind,prompt,hidden in features:
        h,valid=pack_hidden([hidden],device)
        noise=torch.randn(1,4,32,32,device=device,generator=torch.Generator(device=device).manual_seed(args.seed))
        auxiliary=None
        with torch.inference_mode():
            if args.reference_reconstruction and kind=='normal':
                _,prediction,reference=model.forward_with_aux(h,valid,noise)
                auxiliary=model.vae.decode(reference.to(model.vae.dtype)/model.vae.config.scaling_factor).sample.float()
            else:_,prediction=model(h,valid,noise)
        array=((prediction[0].cpu().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy()
        directory=args.output/kind;directory.mkdir(exist_ok=True)
        filename=directory/f'sample_{i:04d}.png';Image.fromarray(array).save(filename)
        if auxiliary is not None:
            path=args.output/'qwen_reference_reconstruction';path.mkdir(exist_ok=True)
            aux_array=((auxiliary[0].cpu().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy()
            Image.fromarray(aux_array).save(path/f'sample_{i:04d}.png')
        reports.append({'sample_id':records[i]['sample_id'],'variant':kind,'prompt':prompt,
          'reference_sha256':sha256(Path(records[i]['image'])),'seed':args.seed,
          'qwen_hidden_sha256':hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
          'qwen_hidden_shape':list(hidden.shape),'output':str(filename),'output_sha256':sha256(filename),
          'text_changed':prompt!=records[i]['prompt']})
    write_json(args.output/'report.json',{'complete':True,'records':reports,
       'checkpoint_sha256':sha256(args.checkpoint),'training_source_commit':state['git']['published_source_commit'],
       'model_inputs':'Raw RGB + text + Gaussian noise; all image information passes full Qwen-VL',
       'reference_vae_encoder':False,'target_or_mask_input':False,'training_auxiliary_discarded':not args.reference_reconstruction,
       'reference_reconstruction_note':'Optional auxiliary decoder diagnostic from Qwen features; not the edited result and not used by ordinary inference',
       'ablation_note':'Gray-image ablation recomputes the FULL Qwen fused representation while retaining the original instruction. Text permutation is within optional category/mode groups.'})


if __name__=='__main__':main()
