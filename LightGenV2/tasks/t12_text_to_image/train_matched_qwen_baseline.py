"""Warm-start the unchanged Qwen28+electronic decoder on all three editing modes.

Frozen Qwen features may be cached for training, never replaced by the small head.
TRAIN fits weights, full VAL chooses best, TEST is not opened during training.
"""
import argparse,gc,json,subprocess,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from .export_matched_summary import sha,per_image_metrics
from .half_qwen import load_half_qwen_text_encoder
from .feature_cache import _qwen_prompts
from .electronic_turbo_infer import _load_adapter
from .product_repair_model import expand_reference_conditioning,one_step_edit
from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset


class CachedPairs(Dataset):
    def __init__(self,payload,conditions):self.payload=payload;self.conditions=conditions
    def __len__(self):return len(self.payload['reference'])
    def __getitem__(self,i):
        return {'reference':self.payload['reference'][i].float(),'target':self.payload['target'][i].float(),
                'condition':self.conditions[self.payload['prompts'][i]][0].float(),'index':i}


def main():
    p=argparse.ArgumentParser()
    for name in ('assets','qwen','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--accumulation',type=int,default=2);p.add_argument('--learning-rate',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=927);p.add_argument('--validation-every',type=int,default=1000)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=True);device=torch.device('cuda:0');assets=args.assets
    cache=assets/'datasets/abo_unified_expanded_latents_qwenmini_true256_v2'
    pairs={split:torch.load(cache/(split+'.pt'),map_location='cpu',mmap=True,weights_only=False) for split in ('train','val')}
    assert len(pairs['train']['reference'])==20736 and len(pairs['val']['reference'])==2304
    data=assets/'datasets/abo_cleanrender_lamp_table_pillow_256_v1'
    instructions=assets/'datasets/abo_unified_expanded_instructions_qwen2_v2.pt'
    datasets={split:ExpandedUnifiedProductEditDataset(data,split,256,instructions) for split in ('train','val')}
    for split in datasets:
        assert len(datasets[split])==len(pairs[split]['reference'])
        for i in range(len(datasets[split])):
            # Identities depend only on source manifest and offset, not image loading.
            source=datasets[split].sources[i//12];mode='background' if i%12<4 else 'object' if i%12<8 else 'joint'
            assert pairs[split]['sample_ids'][i]==f"{source['sample_id']}:{mode}:{i%12}"
    conditions_path=args.output/'qwen28_frozen_conditions.pt'
    qwen,processor,qreport=load_half_qwen_text_encoder(args.qwen,device,keep_layers=28)
    if conditions_path.exists():
        cached=torch.load(conditions_path,map_location='cpu',weights_only=False)
        assert cached['keep_layers']==28
        conditions=cached['conditions']
    else:
        conditions={}
        prompts=sorted({prompt for split in pairs.values() for prompt in split['prompts']})
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for i,prompt in enumerate(prompts):
                tokens={k:v[:,-64:].to(device) for k,v in _qwen_prompts(processor,[prompt]).items()}
                hidden=qwen(**tokens,use_cache=False,return_dict=True).last_hidden_state.float()
                valid=tokens['attention_mask'].to(hidden.dtype).unsqueeze(-1)
                conditions[prompt]=((hidden*valid).sum(1)/valid.sum(1).clamp_min(1)).cpu()
                if i%40==0:print('Qwen28 conditions',i,'/',len(prompts),flush=True)
        torch.save({'keep_layers':28,'qwen_path':str(args.qwen),'qwen_report':qreport,'conditions':conditions},conditions_path)
    del qwen,processor;gc.collect();torch.cuda.empty_cache()
    from diffusers import AutoencoderKL,UNet2DConditionModel
    source=assets/'runs/abo_scene_replace_28layer_electronic_baseline_v1/best_model.pt'
    original=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    adapter,_=_load_adapter(assets/'runs/677deac6/qwen_sd_turbo_one_step_text_aug_seed42/best_adapter.pt',device)
    adapter.load_state_dict(original['adapter']);adapter.train().requires_grad_(True)
    unet=UNet2DConditionModel.from_pretrained(assets/'models/bk-sdm-v2-tiny',subfolder='unet',variant='fp16',torch_dtype=torch.float32,local_files_only=True)
    expand_reference_conditioning(unet);unet.load_state_dict(original['unet']);unet=unet.to(device)
    unet.enable_gradient_checkpointing();unet.train().requires_grad_(True)
    vae=AutoencoderKL.from_pretrained(assets/'models/sd-turbo-fp16',subfolder='vae',variant='fp16',torch_dtype=torch.float16,local_files_only=True).to(device).eval().requires_grad_(False)
    # Existing latent cache is reusable only if it matches current un-eroded GT.
    cache_audit={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for split in datasets:
            errors=[]
            indices=[0,4,8,len(datasets[split])//2,len(datasets[split])-1]
            for i in indices:
                row=datasets[split][i];assert row['prompt']==pairs[split]['prompts'][i]
                for key in ('reference','target'):
                    latent=vae.encode(row[key][None].cuda().half()).latent_dist.mode()*vae.config.scaling_factor
                    errors.append(float((latent.float().cpu()-pairs[split][key][i:i+1].float()).square().mean()))
            cache_audit[split]={'indices':indices,'max_latent_mse':max(errors),'cache_sha256':sha(cache/(split+'.pt'))}
            if max(errors)>2e-4:raise ValueError(f'Latent cache does not match current dataset: {split}: {max(errors)}')
    cfg=dict(original['training_config']);residual=float(cfg['residual_scale']);noise_scale=float(cfg['noise_scale'])
    counted=qreport['counted_text_encoder_parameters']+sum(v.numel() for m in (adapter,unet,vae) for v in m.parameters())
    protocol={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'argv':sys.argv,
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'seed':args.seed,'train_count':20736,'val_count':2304,
              'architecture':'Unchanged full Qwen28 + electronic BK-SDM tiny UNet + SD-Turbo VAE + original conditioning adapter',
              'qwen':qreport,'counted_parameters':counted,'fixed_condition_buffer_values':10171648,
              'source_sha256':sha(source),'cache_audit':cache_audit,
              'data_sha256':{split:sha(data/(split+'.jsonl')) for split in ('train','val')},
              'instruction_cache_sha256':sha(instructions),'frozen_condition_cache_sha256':sha(conditions_path),
              'training':'Qwen and VAE frozen; all UNet and original adapter trainable; TRAIN only; full VAL mean per-image PSNR selection; no TEST used',
              'loss':'latent MSE + 0.05 latent spatial-gradient L1; every eighth batch adds 0.1 full256 RGB MSE',
              'status':'training'}
    def write(name,obj):(args.output/name).write_text(json.dumps(obj,indent=2))
    write('protocol.json',protocol)
    def prediction(ref,condition,noise):return one_step_edit(unet,noise,ref,adapter.condition(condition),torch.tensor(1.,device=device),residual_scale=residual,noise_scale=noise_scale)
    val_loader=DataLoader(CachedPairs(pairs['val'],conditions),batch_size=8,num_workers=0)
    @torch.no_grad()
    def validate():
        unet.eval();adapter.eval();totals={'count':0,'psnr_db':0.,'ssim':0.,'mse_0_1':0.,'mae_0_1':0.}
        for batch in val_loader:
            ref,condition=batch['reference'].cuda(),batch['condition'].cuda()
            noise=torch.cat([torch.randn((1,4,32,32),device=device,generator=torch.Generator(device=device).manual_seed(1042+int(i))) for i in batch['index']])
            with torch.autocast('cuda',dtype=torch.float16):
                edited=prediction(ref,condition,noise);rgb=vae.decode(edited.half()/vae.config.scaling_factor,return_dict=False)[0].float()
            target=torch.stack([datasets['val'][int(i)]['target'] for i in batch['index']]).cuda()
            for metric in per_image_metrics(rgb,target):
                totals['count']+=1
                for k in metric:totals[k]+=metric[k]
        for k in list(totals):
            if k!='count':totals[k]/=totals['count']
        unet.train();adapter.train();return totals
    baseline_val=validate();best_psnr=baseline_val['psnr_db'];best_step=0;history=[];step=0
    def save(name):
        saved={k:v for k,v in original.items() if k not in ('unet','adapter')}
        saved.update(unet={k:v.detach().cpu() for k,v in unet.state_dict().items()},adapter={k:v.detach().cpu() for k,v in adapter.state_dict().items()},
                     matched_training={'protocol':protocol,'step':step,'best_val_psnr':best_psnr,'best_step':best_step})
        torch.save(saved,args.output/name)
    save('best_checkpoint.pt')
    optimizer=torch.optim.AdamW([{'params':unet.parameters(),'lr':args.learning_rate},{'params':adapter.parameters(),'lr':args.learning_rate}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');began=time.monotonic();optimizer.zero_grad(set_to_none=True)
    loader=DataLoader(CachedPairs(pairs['train'],conditions),batch_size=args.batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(args.seed))
    def check(epoch,loss):
        nonlocal best_psnr,best_step
        val=validate();entry={'epoch':epoch,'step':step,'loss':loss,'val':val,'elapsed_seconds':time.monotonic()-began};history.append(entry)
        if val['psnr_db']>best_psnr:best_psnr=val['psnr_db'];best_step=step;save('best_checkpoint.pt')
        save('last_checkpoint.pt');write('history.json',history);write('progress.json',{'status':'training','epoch':epoch,'step':step,'best_step':best_step,'best_val_psnr':best_psnr,'last_val':val});print(json.dumps(entry),flush=True)
    write('baseline_val.json',baseline_val)
    for epoch in range(1,args.epochs+1):
        for bi,batch in enumerate(loader):
            ref,target,condition=batch['reference'].cuda(),batch['target'].cuda(),batch['condition'].cuda();noise=torch.randn_like(ref)
            with torch.autocast('cuda',dtype=torch.float16):
                edited=prediction(ref,condition,noise)
                loss=F.mse_loss(edited.float(),target.float())
                for axis in (-1,-2):loss=loss+.05*F.l1_loss(torch.diff(edited.float(),dim=axis),torch.diff(target.float(),dim=axis))
                if bi%8==0:
                    rgb=vae.decode(edited.half()/vae.config.scaling_factor,return_dict=False)[0]
                    gt=torch.stack([datasets['train'][int(i)]['target'] for i in batch['index']]).cuda()
                    loss=loss+.1*F.mse_loss(rgb.float(),gt.float())
            if not torch.isfinite(loss):raise ValueError('Nonfinite TRAIN loss')
            scaler.scale(loss/args.accumulation).backward();step+=1
            if (bi+1)%args.accumulation==0 or bi+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([v for m in (unet,adapter) for v in m.parameters()],1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            if step%100==0:
                print(json.dumps({'epoch':epoch,'step':step,'loss':float(loss),'elapsed_seconds':time.monotonic()-began}),flush=True)
                write('progress.json',{'status':'training','epoch':epoch,'step':step,'loss':float(loss),'best_step':best_step,'best_val_psnr':best_psnr})
            if step%args.validation_every==0:check(epoch,float(loss))
        check(epoch,float(loss))
    write('report.json',{'status':'complete','protocol':protocol,'best_step':best_step,'best_val_psnr':best_psnr,'baseline_val':baseline_val,
                        'best_checkpoint_sha256':sha(args.output/'best_checkpoint.pt'),'elapsed_seconds':time.monotonic()-began,'test_evaluation':'pending separate fixed-weight export'})
    print('Training complete',best_psnr,flush=True)


if __name__=='__main__':main()
