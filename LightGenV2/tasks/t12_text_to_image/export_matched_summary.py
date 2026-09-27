"""Fixed-weight TEST export; per-image [0,1] metrics and native lossless PNGs.

No training, masks, image retrieval, or reference-pixel pasting in inference.
Historical Qwen baseline is explicitly a transfer diagnostic, not matched training.
"""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from .audited_unified import architecture_report
from .sealed_editor import build_sealed
from .qwen_mini_small import PromptEmbeddingLookup
from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from .lab_shs8um.train_channel_robust import ssim_per_image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def per_image_metrics(pred, target):
    pred = pred.float().clamp(-1, 1).add(1).mul(.5)
    target = target.float().clamp(-1, 1).add(1).mul(.5)
    mse = (pred-target).square().mean((1,2,3))
    mae = (pred-target).abs().mean((1,2,3))
    psnr = -10*torch.log10(mse.clamp_min(1e-12))
    ssim = ssim_per_image(pred, target)
    return [{k: float(v[i]) for k, v in zip(('mse_0_1','mae_0_1','psnr_db','ssim'), (mse,mae,psnr,ssim))}
            for i in range(len(pred))]


def save_png(value, path):
    array = value.detach().float().clamp(-1,1).add(1).mul(127.5).round().byte().permute(1,2,0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return sha(path)


def noise_batch(count, device, start, seed):
    return torch.cat([torch.randn((1,4,32,32), device=device,
                     generator=torch.Generator(device=device).manual_seed(seed+start+i)) for i in range(count)])


def summarize(rows, prefix):
    result = {}
    for group in ('overall','background','object','joint'):
        selected = [r for r in rows if group == 'overall' or r['mode'] == group]
        result[group] = {'count':len(selected), **{k: float(np.mean([r[prefix+'_'+k] for r in selected]))
                            for k in ('mse_0_1','mae_0_1','psnr_db','ssim')}}
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('assets','qwen','large','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seed',type=int,default=1042)
    args=p.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    args.output.mkdir(parents=True,exist_ok=True)
    assets=args.assets
    data=assets/'datasets/abo_cleanrender_lamp_table_pillow_256_v1'
    instruction=assets/'datasets/abo_unified_expanded_instructions_qwen2_v2.pt'
    embedding=assets/'datasets/abo_unified_expanded_qwen_embeddings_v2.pt'
    dataset=ExpandedUnifiedProductEditDataset(data,'test',256,instruction)
    metadata={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'argv':sys.argv,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
              'samples':len(dataset),'split':'TEST','resolution':256,'seed':args.seed,
              'metric_protocol':'RGB [0,1], clamp before metrics; per-image PSNR then mean; SSIM Gaussian 11 sigma1.5 valid',
              'png_protocol':'round((clamp(RGB,-1,1)+1)*127.5); native256 lossless; metrics before PNG quantization',
              'test_manifest_sha256':sha(data/'test.jsonl'),
              'instruction_cache_sha256':sha(instruction),'embedding_cache_sha256':sha(embedding),
              'large_checkpoint_sha256':sha(args.large),'baseline_caveat':'Historical lamp-background-only Qwen28 checkpoint; new category/object/joint transfer diagnostic, not matched-training comparison'}
    model=build_sealed(torch.load(args.large,map_location='cpu',weights_only=False)).to(device).eval()
    metadata['large_architecture']=architecture_report(model)
    lookup=PromptEmbeddingLookup(embedding)
    rows=[]
    for bi,batch in enumerate(DataLoader(dataset,batch_size=args.batch_size,num_workers=0)):
        start=len(rows)
        ref,gt=batch['reference'].to(device),batch['target'].to(device)
        emb,mask,_=lookup.batch(list(batch['prompt']),device)
        pred=model(ref,emb.float(),mask,noise_batch(len(ref),device,start,args.seed))
        metrics=per_image_metrics(pred,gt)
        for i in range(len(ref)):
            index=start+i; sid=f'test_{index:05d}'
            row={'test_index':index,'sample_id':sid,'source_id':batch['sample_id'][i],
                 'category':batch['category'][i],'mode':batch['mode'][i],'prompt':batch['prompt'][i]}
            row.update({'large_sim_'+k:v for k,v in metrics[i].items()})
            for name,value in (('reference',ref[i]),('target',gt[i]),('large_sim',pred[i])):
                relative=f'images/{name}/{sid}.png'
                row[name+'_image']=relative
                row[name+'_sha256']=save_png(value,args.output/relative)
            rows.append(row)
        if bi%50==0: print('large',len(rows),'/',len(dataset),flush=True)
    (args.output/'large_metrics.json').write_text(json.dumps(summarize(rows,'large_sim'),indent=2))
    del model,lookup,emb,mask,pred,ref,gt
    gc.collect();torch.cuda.empty_cache()
    from diffusers import AutoencoderKL,EulerDiscreteScheduler,UNet2DConditionModel
    from .half_qwen import load_half_qwen_text_encoder
    from .feature_cache import _qwen_prompts
    from .electronic_turbo_infer import _load_adapter
    from .product_repair_model import expand_reference_conditioning,one_step_edit
    baseline=assets/'runs/abo_scene_replace_28layer_electronic_baseline_v1/best_model.pt'
    payload=torch.load(baseline,map_location='cpu',weights_only=False,mmap=True)
    qwen,processor,qreport=load_half_qwen_text_encoder(args.qwen,device,keep_layers=28)
    adapter,_=_load_adapter(assets/'runs/677deac6/qwen_sd_turbo_one_step_text_aug_seed42/best_adapter.pt',device)
    adapter.load_state_dict(payload['adapter']);adapter.eval()
    unet=UNet2DConditionModel.from_pretrained(assets/'models/bk-sdm-v2-tiny',subfolder='unet',variant='fp16',torch_dtype=torch.float32,local_files_only=True)
    expand_reference_conditioning(unet)
    unet.load_state_dict(payload['unet']);unet=unet.to(device).eval()
    vae=AutoencoderKL.from_pretrained(assets/'models/sd-turbo-fp16',subfolder='vae',variant='fp16',torch_dtype=torch.float16,local_files_only=True).to(device).eval()
    scheduler=EulerDiscreteScheduler.from_pretrained(assets/'models/sd-turbo-fp16',subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(1,device=device)
    metadata['baseline_checkpoint_sha256']=sha(baseline)
    metadata['baseline_qwen_report']=qreport
    metadata['baseline_counted_parameters']=qreport['counted_text_encoder_parameters']+sum(v.numel() for m in (adapter,unet,vae) for v in m.parameters())
    conditions={}
    for bi,batch in enumerate(DataLoader(dataset,batch_size=args.batch_size,num_workers=0)):
        start=bi*args.batch_size
        ref,gt=batch['reference'].to(device),batch['target'].to(device)
        with torch.autocast('cuda',dtype=torch.float16):
            for prompt in batch['prompt']:
                if prompt not in conditions:
                    tokens={k:v[:,-64:].to(device) for k,v in _qwen_prompts(processor,[prompt]).items()}
                    hidden=qwen(**tokens,use_cache=False,return_dict=True).last_hidden_state.float()
                    valid=tokens['attention_mask'].to(hidden.dtype).unsqueeze(-1)
                    conditions[prompt]=(hidden*valid).sum(1)/valid.sum(1).clamp_min(1)
            pooled=torch.cat([conditions[v] for v in batch['prompt']])
            latent=vae.encode(ref.half()).latent_dist.mode()*vae.config.scaling_factor
            edited=one_step_edit(unet,noise_batch(len(ref),device,start,args.seed),latent.float(),adapter.condition(pooled.float()),scheduler.sigmas[0],
                                 residual_scale=float(payload['training_config']['residual_scale']),noise_scale=float(payload['training_config']['noise_scale']))
            pred=vae.decode(edited.half()/vae.config.scaling_factor,return_dict=False)[0].float()
        for i,metric in enumerate(per_image_metrics(pred,gt)):
            row=rows[start+i]
            row.update({'qwen_baseline_'+k:v for k,v in metric.items()})
            relative=f"images/qwen_baseline/{row['sample_id']}.png"
            row['qwen_baseline_image']=relative
            row['qwen_baseline_sha256']=save_png(pred[i],args.output/relative)
        if bi%50==0: print('baseline',start+len(ref),'/',len(dataset),flush=True)
    metadata['metrics']={'large_sim':summarize(rows,'large_sim'),'qwen_baseline':summarize(rows,'qwen_baseline')}
    metadata['precision']={'large':'FP32','baseline':'FP16 autocast; Qwen BF16 pretrained weights'}
    metadata['status']='complete'
    (args.output/'report.json').write_text(json.dumps(metadata,indent=2))
    (args.output/'sample_metrics.json').write_text(json.dumps(rows,indent=2))
    with zipfile.ZipFile(args.output/'paper_bundle.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
        for path in sorted(args.output.rglob('*.png')):archive.write(path,path.relative_to(args.output).as_posix())
        for name in ('report.json','sample_metrics.json'):archive.write(args.output/name,name)
    print(json.dumps(metadata['metrics']),flush=True)


if __name__=='__main__': main()
