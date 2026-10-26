"""TRAIN-only generator/conditioning probe, never an inference benchmark.

Compare native CLIP target-caption conditions with learned Qwen conditions on
the same 18 TRAIN diagnostic pairs. Target captions must never be mistaken for
available test-time information. No optimization or VAL/TEST selection occurs.
"""
import argparse
import os
from pathlib import Path

import torch

from .qwen_vl_turbo_data import pilot_subset,sha256,write_json
from .qwen_vl_turbo_model import NativeTurboHead,QueryResampler
from .qwen_vl_turbo_run import evaluate,git_identity
from .qwen_vl_turbo_training import configure_unet_training


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--sd',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=state['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid']:
        raise ValueError('Select the explicitly authorized GPU UUID')
    if sha256(args.cache)!=state['cache_sha256']:
        raise ValueError('Checkpoint/cache identity mismatch')
    payload=torch.load(args.cache,map_location='cpu',weights_only=False)
    names={r['sample_id'] for r in pilot_subset(payload['rows'],2) if r['split']=='train'}
    ids=[i for i,r in enumerate(payload['rows']) if r['sample_id'] in names]
    if len(ids)!=18 or any(payload['rows'][i]['split']!='train' for i in ids):
        raise ValueError('Expected the fixed 18 TRAIN diagnostics only')
    for row in payload['rows']:
        if 'shared_noise_seed' in cfg:row['noise_seed']=int(cfg['shared_noise_seed'])
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'identity.json',{'scope':'TRAIN-only teacher-interface diagnostic, NOT deployed Qwen performance',
        'checkpoint_sha256':sha256(args.checkpoint),'cache_sha256':state['cache_sha256'],
        'git':git_identity(),'diagnostic_source_sha256':sha256(Path(__file__)),
        'sample_ids':[payload['rows'][i]['sample_id'] for i in ids],
        'no_updates':True,'no_val_or_test_forward':True})
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'],0)
    device=torch.device('cuda:0')
    from diffusers import AutoencoderKL,EulerDiscreteScheduler,UNet2DConditionModel
    from peft import LoraConfig
    import lpips
    adapter=QueryResampler(hidden=cfg['adapter_hidden'],layers=cfg['adapter_layers']).to(device)
    adapter.load_state_dict(state['adapter'])
    unet=UNet2DConditionModel.from_pretrained(args.sd,subfolder='unet',variant='fp16',
        torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    unet.add_adapter(LoraConfig(r=cfg['unet_lora_rank'],lora_alpha=cfg['unet_lora_rank'],
        target_modules=['to_q','to_k','to_v','to_out.0'],init_lora_weights=True))
    configure_unet_training(unet,cfg['unet_training'])
    vae=AutoencoderKL.from_pretrained(args.sd,subfolder='vae',variant='fp16',
        torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    vae.encoder=None;vae.quant_conv=None
    scheduler=EulerDiscreteScheduler.from_pretrained(args.sd,subfolder='scheduler',local_files_only=True)
    model=NativeTurboHead(unet,vae,scheduler,adapter).eval()
    metric=lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    public=evaluate(model,payload,ids,device,args.output,'train_public_native_caption',
        native_teacher=True,export_png=True,metric_network=metric)
    names={n for n,p in unet.named_parameters() if p.requires_grad}
    if names!=set(state['unet_trainable']):raise ValueError('Trainable scope mismatch')
    with torch.no_grad():
        for name,param in unet.named_parameters():
            if name in names:param.data=state['unet_trainable'][name].to(device)
    model.eval().requires_grad_(False)
    adapted=evaluate(model,payload,ids,device,args.output,'train_adapted_native_caption',
        native_teacher=True,export_png=True,metric_network=metric)
    qwen=evaluate(model,payload,ids,device,args.output,'train_adapted_qwen',
        export_png=True,metric_network=metric)
    result={'complete':True,'scope':'18 TRAIN pairs, target-caption teacher probes; NOT VAL/TEST or deployed-model comparisons',
        'public_unet_native_caption':{k:v for k,v in public.items() if k!='images'},
        'adapted_unet_native_caption':{k:v for k,v in adapted.items() if k!='images'},
        'adapted_unet_qwen_condition':{k:v for k,v in qwen.items() if k!='images'}}
    write_json(args.output/'report.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
