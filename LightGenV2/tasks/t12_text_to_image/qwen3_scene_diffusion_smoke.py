"""Four real paired diffusion updates on six reviewed TRAIN pairs, not a result.

GT is VAE-encoded ONLY for the usual noisy-TRAIN-latent/epsilon loss. Reference
RGB and text enter Qwen alone. At deployment the generator starts from noise;
there is no target or reference VAE input path. 512 is a backward smoke size,
not a replacement for the 1024 generation/evaluation protocol.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image
import torch
from .qwen3_scene_conditioning import QwenSDXLConditions,fresh_qwen,raw_features
from .white_reference_scene_probe import GPU,digest,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','alignment','qwen','model-path','adapter-path','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();commit=os.environ.get('T12_SOURCE_COMMIT','')
    if a.output.exists() or len(commit)!=40 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('New run/published source/GPU6 only')
    report=json.loads((a.data/'report.json').read_text());rows=json.loads((a.data/'pairs.json').read_text())
    assert report['complete'] and not report['DEV_or_TEST_used'] and len(rows)==6 and all(r['split']=='train' for r in rows)
    aligned=json.loads((a.alignment/'alignment_report.json').read_text());assert aligned['complete']
    if digest(a.alignment/'best_checkpoint.pt')!=aligned['best_sha256']:raise ValueError('Alignment identity changed')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.7,0)
    write(a.output/'execution.json',{'pid':os.getpid(),'argv':sys.argv,'source_commit':commit,'source_sha256':digest(Path(__file__)),
      'pairs_sha256':digest(a.data/'pairs.json'),'alignment_sha256':aligned['best_sha256'],'backward_smoke_resolution':512})
    try:
        from diffusers import StableDiffusionXLPipeline,DDPMScheduler
        from peft import LoraConfig
        if not torch.cuda.is_bf16_supported():raise ValueError('Authorized A100 BF16 training required')
        dtype=torch.bfloat16
        qwen,processor=fresh_qwen(a.qwen);features=[]
        for row in rows:
            assert digest(a.data/row['reference'])==row['reference_sha256'] and digest(a.data/row['target'])==row['target_sha256']
            with Image.open(a.data/row['reference']) as im:h,v,ids=raw_features(qwen,processor,im.convert('RGB'),row['prompt'])
            features.append((h.float(),v.float()))
        del qwen,processor;gc.collect();torch.cuda.empty_cache()
        pipe=StableDiffusionXLPipeline.from_pretrained(a.model_path,variant='fp16',torch_dtype=dtype,
          text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None,image_encoder=None,local_files_only=True).to('cuda')
        pipe.load_ip_adapter(str(a.adapter_path),subfolder='sdxl_models',weight_name='ip-adapter-plus_sdxl_vit-h.safetensors',image_encoder_folder=None,local_files_only=True)
        pipe.set_ip_adapter_scale(.25)
        for c in pipe.components.values():
            if isinstance(c,torch.nn.Module):c.eval().requires_grad_(False)
        pipe.vae.float();latents=[]
        for row in rows:
            with Image.open(a.data/row['target']) as im:rgb=np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)).copy()
            target=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).cuda().float()/127.5-1
            with torch.no_grad():z=pipe.vae.encode(target).latent_dist.mode()*pipe.vae.config.scaling_factor
            latents.append(z.to(dtype))
        pipe.vae.cpu();gc.collect();torch.cuda.empty_cache()
        unet=pipe.unet;unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v','to_out.0']),adapter_name='scene_smoke')
        unet.enable_gradient_checkpointing();unet.train()
        assert all(('lora_' in n) for n,p in unet.named_parameters() if p.requires_grad)
        head=QwenSDXLConditions().cuda().train();state=torch.load(a.alignment/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        head.load_state_dict(state['state_dict'],strict=True)
        lora=[p for p in unet.parameters() if p.requires_grad];opt=torch.optim.AdamW([{'params':head.parameters(),'lr':1e-4},{'params':lora,'lr':1e-5}],weight_decay=.01)
        scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon'
        results=[]
        for step,timestep in enumerate([128,384,640,896]):
            row=rows[step];h,v=features[step];cond=head(h.unsqueeze(0).cuda(),v.unsqueeze(0).cuda())
            generator=torch.Generator(device='cuda').manual_seed((row['seed']+step)%(2**31));noise=torch.randn(latents[step].shape,generator=generator,device='cuda',dtype=dtype)
            t=torch.tensor([timestep],device='cuda',dtype=torch.long);noisy=scheduler.add_noise(latents[step],noise,t)
            kwargs={'text_embeds':cond['pooled'].to(dtype),'time_ids':torch.tensor([[512,512,0,0,512,512]],device='cuda',dtype=dtype),
                    'image_embeds':[cond['image'].to(dtype).unsqueeze(1)]}
            prediction=unet(noisy,t,encoder_hidden_states=cond['text'].to(dtype),added_cond_kwargs=kwargs).sample
            loss=(prediction.float()-noise.float()).square().mean();opt.zero_grad();loss.backward()
            hgrad=sum(p.grad.float().abs().sum().item() for p in head.parameters() if p.grad is not None)
            lgrad=sum(p.grad.float().abs().sum().item() for p in lora if p.grad is not None)
            all_trainable=list(head.parameters())+lora
            write(a.output/'gradient_health.json',{'step':step+1,'loss_finite':bool(torch.isfinite(loss)),
              'head_gradient_finite':bool(np.isfinite(hgrad)),'lora_gradient_finite':bool(np.isfinite(lgrad)),
              'head_text_gradient':sum(p.grad.float().abs().sum().item() for p in head.text.parameters() if p.grad is not None),
              'head_image_gradient':sum(p.grad.float().abs().sum().item() for p in head.image.parameters() if p.grad is not None),
              'nonfinite_parameter_gradients':[n for n,p in list(head.named_parameters())+list(unet.named_parameters()) if p.grad is not None and not torch.isfinite(p.grad).all()]})
            assert torch.isfinite(loss) and hgrad>0 and lgrad>0 and all(torch.isfinite(p.grad).all() for p in all_trainable if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(all_trainable,1.);opt.step()
            results.append({'step':step+1,'TRAIN_sample_id':row['sample_id'],'timestep':timestep,'epsilon_MSE':loss.item(),'head_absolute_gradient':hgrad,'UNet_LoRA_absolute_gradient':lgrad})
            write(a.output/'progress.json',{'stage':'actual_paired_epsilon_loss_backward','steps':results})
        assert all(not p.requires_grad for p in pipe.vae.parameters())
        torch.save({'schema':'qwen3_scene_paired_diffusion_smoke_v1','head':head.state_dict(),'unet_lora':{n:p.detach().cpu() for n,p in unet.named_parameters() if 'lora_' in n},
          'source_commit':commit,'step':4,'not_for_baseline_or_evaluation':True},a.output/'last_checkpoint.pt')
        write(a.output/'report.json',{'complete':True,'actual_backward_steps':results,'TRAIN_pairs_available':6,'TRAIN_pairs_updated':4,
          'Qwen_full_raw_forward_per_reference_prompt':True,'Qwen_frozen':True,'VAE_frozen':True,'UNet_base_frozen':True,
          'head_parameters':sum(p.numel() for p in head.parameters()),'UNet_LoRA_parameters':sum(p.numel() for p in lora),
          'loss':'Standard noisy TRAIN target latent -> epsilon MSE; not RGB/latent deterministic target regression.',
          'GT_at_inference':False,'reference_VAE_encoder_branch':False,'CLIP_encoders_loaded':False,'DEV_or_TEST_used':False,
          'quality_or_speed_established':False,'training_precision':'BF16 base, FP32 head/LoRA optimizer; FP32 VAE target encoding',
          'smoke_resolution':512,'formal_generation_resolution_unchanged':1024,
          'checkpoint_sha256':digest(a.output/'last_checkpoint.pt'),'max_allocated_GPU_bytes':torch.cuda.max_memory_allocated()})
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
