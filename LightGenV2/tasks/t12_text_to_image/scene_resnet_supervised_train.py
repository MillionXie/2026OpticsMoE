"""Image epsilon supervision only; frozen RN18 and exact E68 decoder.

Random readouts, no Qwen features, condition targets, anchors or inherited
frontend parameters. Original reviewed TRAIN target images are unchanged.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
from PIL import Image
import torch
from .scene_instruction_data import sha,write
from .scene_resnet_frontend import (ResNetSceneFrontend,ResNetConditionReadout,rgb_tensor,tokenize_text,
    experiment_profile,PAIRS,REQUESTS)
from .scene_resnet_frontend_run import parent,checked_data

PROFILE='qwen70_30_supervised'
REQUESTED_EPOCHS=(0,1,2,5,10)


def requested_training_epochs(sweep=False):
    return 10 if sweep else 50


def partial_training_updates(value,sweep=False):
    if value is not None and (sweep or type(value)is not int or not 1<=value<78):
        raise ValueError('Explicit sub-epoch budget1..77; cannot mix with epoch sweep')
    return value


def tensor_state_sha(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(list(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def fresh_readout():
    # Never call initialize_readouts and never accept a parent head argument.
    torch.manual_seed(42)
    return ResNetConditionReadout()


def image_epsilon_loss(prediction,noise,weights):
    if prediction.shape!=noise.shape or prediction.ndim!=4 or weights.shape!=(len(noise),):
        raise ValueError('Per-image epsilon and reviewed target weights required')
    if not torch.all((weights==.5)|(weights==1.)):
        raise ValueError('Keep original reviewed half/full weights')
    per_image=(prediction.float()-noise.float().detach()).square().flatten(1).mean(1)
    return (per_image*weights).mean()


def cache(a):
    from diffusers import AutoencoderKL
    from ..t19_resnet20_baselines.frozen_resnet18 import assert_encoder_unchanged
    rows,_=checked_data(a)
    from .white_reference_scene_probe import model_manifest
    expected=json.loads(a.decoder_assets.read_text())
    if model_manifest(a.model_path)!=expected['SDXL']:raise ValueError('Same original Qwen VAE required')
    model=ResNetSceneFrontend().cuda().eval();initial=model.encoder.pretrained_provenance['encoder_state_sha256']
    visuals={};records=[]
    for row in rows:
        row=dict(row,input_key=hashlib.sha256((row['reference_sha256']+'\0'+row['prompt']).encode()).hexdigest())
        records.append(row)
        if row['reference_sha256']not in visuals:
            with Image.open(a.data/row['reference'])as im:rgb=rgb_tensor(im.convert('RGB'))[None].cuda()
            with torch.no_grad():visuals[row['reference_sha256']]=model.visual_tokens(rgb)[0].cpu().clone()
    from .scene_resnet_frontend import validate_training_rows
    validate_training_rows(records)
    frozen=assert_encoder_unchanged(model.encoder,initial);official=model.encoder.pretrained_provenance
    del model;gc.collect();torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(a.model_path/'vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval().requires_grad_(False)
    latents={}
    for i,row in enumerate(records):
        path=a.data/row['target']
        if sha(path)!=row['target_sha256']:raise ValueError('Changed TRAIN image')
        if row['target_sha256']not in latents:
            with Image.open(path)as im:
                if im.size!=(512,512):raise ValueError('Original512 TRAIN targets required')
                image=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1)[None].cuda().float()/127.5-1
            with torch.no_grad():latent=vae.encode(image).latent_dist.mode()*vae.config.scaling_factor
            if not torch.isfinite(latent).all():raise ValueError('Finite original VAE target latents required')
            latents[row['target_sha256']]=latent[0].cpu().bfloat16().clone()
        write(a.output/'progress.json',dict(stage='reviewed_TRAIN_image_latents_only',done=i+1,total=624))
    payload=dict(schema=experiment_profile(PROFILE)['schema'],records=records,visuals=visuals,latents=latents,
                 pair_sha256=PAIRS,official_visual=official)
    torch.save(payload,a.output/'image_cache.pt')
    write(a.output/'report.json',dict(complete=True,cache_sha256=sha(a.output/'image_cache.pt'),TRAIN_contexts=624,TRAIN_views=76,
        Qwen_features_or_conditions_loaded=False,readout_inheritance=False,condition_distillation=False,
        target_source='Unchanged reviewed paired TRAIN images, generated historically by frozen official OmniGen',
        original_VAE_mode_latents=True,frozen_encoder_audit=frozen,official_visual=official,TEST_targets_loaded=False,
        source_commit=os.environ['T12_SOURCE_COMMIT']))


def frozen_decoder(a,state):
    from peft import LoraConfig
    from .qwen3_scene_paired_train import pipeline
    from .scene_fast_tail import fuse_task_adapter
    from .scene_depth_probe import prune_unet
    from .white_reference_scene_probe import model_manifest
    assets=json.loads(a.decoder_assets.read_text())
    if model_manifest(a.model_path)!=assets['SDXL']or model_manifest(a.adapter_path)!=assets['IPAdapter']:
        raise ValueError('Exact original Qwen decoder assets required')
    pipe=pipeline(a);pipe.vae.cpu()
    shapes={n:tuple(p.shape)for n,p in pipe.unet.named_parameters()}
    pipe.unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v','to_out.0']),adapter_name='scene_pair')
    named=dict(pipe.unet.named_parameters())
    if set(state['unet_lora'])!={n for n in named if 'lora_'in n}:raise ValueError('Matched fixed decoder task-LoRA required')
    with torch.no_grad():
        for n,v in state['unet_lora'].items():named[n].copy_(v)
    pipe.unet.requires_grad_(False);fuse_task_adapter(pipe,shapes)
    graph=prune_unet(pipe.unet,10)
    if graph['actual_blocks']!=70 or graph['removed_parameters']!=0:raise ValueError('All original70 blocks required')
    # Diffusers gates activation checkpointing on train mode. SDXL has no active
    # dropout/BN here; all weights remain frozen. RN18 stays strictly eval.
    if any(isinstance(m,torch.nn.modules.dropout._DropoutNd)and m.p for m in pipe.unet.modules()):
        raise ValueError('No decoder dropout changes allowed')
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm)for m in pipe.unet.modules()):
        raise ValueError('No mutable decoder BN allowed')
    pipe.unet.train();pipe.unet.enable_gradient_checkpointing()
    if any(p.requires_grad for p in pipe.unet.parameters()):raise ValueError('Decoder must remain frozen')
    return pipe,graph


def fit(a):
    from diffusers import DDPMScheduler
    from .scene_resnet_frontend import validate_training_rows
    state=parent(a);profile=experiment_profile(PROFILE)
    cr=json.loads((a.cache/'report.json').read_text())
    if not cr['complete']or cr['condition_distillation']or cr['readout_inheritance']or sha(a.cache/'image_cache.pt')!=cr['cache_sha256']:
        raise ValueError('Image-only independently encoded TRAIN cache required')
    data=torch.load(a.cache/'image_cache.pt',map_location='cpu',weights_only=False,mmap=True)
    if set(data)!={'schema','records','visuals','latents','pair_sha256','official_visual'}or data['schema']!=profile['schema']or data['pair_sha256']!=PAIRS:
        raise ValueError('Qwen feature/condition targets forbidden in this profile')
    rows=data['records'];validate_training_rows(rows)
    pipe,graph=frozen_decoder(a,state);unet=pipe.unet
    versions={n:p._version for n,p in unet.named_parameters()}
    scheduler=DDPMScheduler.from_config(pipe.scheduler.config)
    if scheduler.config.prediction_type!='epsilon':raise ValueError('Standard epsilon scheduler required')
    time_ids=torch.tensor([[1024,1024,0,0,1024,1024]],device='cuda',dtype=torch.bfloat16)
    def fresh():
        model=fresh_readout().cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        if set(map(id,model.parameters()))&set(map(id,unet.parameters())):raise ValueError('Disjoint frozen decoder and optimizer required')
        return model,opt
    model,opt=fresh();initial=tensor_state_sha(model)
    def update(indices,seed):
        selected=[rows[i]for i in indices];size=len(selected)
        latent=torch.stack([data['latents'][r['target_sha256']]for r in selected]).cuda()
        rng=torch.Generator(device='cuda').manual_seed(seed)
        noise=torch.randn(latent.shape,generator=rng,device='cuda',dtype=torch.bfloat16)
        timestep=torch.randint(0,scheduler.config.num_train_timesteps,(size,),generator=rng,device='cuda')
        noisy=scheduler.add_noise(latent,noise,timestep)
        opt.zero_grad();loss_total=0.
        for offset in range(0,size,2):
            group=selected[offset:offset+2];n=len(group)
            visual=torch.stack([data['visuals'][r['reference_sha256']]for r in group]).cuda().float()
            tokens=tokenize_text([r['prompt']for r in group]).cuda()
            cond=model(visual,tokens)
            prediction=unet(noisy[offset:offset+n],timestep[offset:offset+n],encoder_hidden_states=cond['text'].bfloat16(),
                added_cond_kwargs={'text_embeds':cond['pooled'].bfloat16(),'time_ids':time_ids.expand(n,-1),
                                  'image_embeds':[cond['image'].bfloat16().unsqueeze(1)]}).sample
            weight=torch.tensor([r['supervision_weight']for r in group],device='cuda')
            loss=image_epsilon_loss(prediction,noise[offset:offset+n],weight)*n/size
            if not torch.isfinite(loss):raise ValueError('Finite image epsilon loss required')
            loss.backward();loss_total+=float(loss.detach())
        gradients={name:sum(float(p.grad.abs().sum())for k,p in model.named_parameters()if k.startswith(name)and p.grad is not None)
                   for name in ['text_encoder','project','text_projection','text.','image.','pooled.']}
        if not all(v>0 for v in gradients.values())or not all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None):
            raise ValueError('Finite nonzero image-supervision gradient in all frontend components required')
        if any(p.grad is not None or p.requires_grad or p._version!=versions[n]for n,p in unet.named_parameters()):
            raise ValueError('Frozen decoder mutated or acquired gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        return loss_total,gradients
    smoke=[]
    for step in range(4):
        began=time.monotonic();loss,grads=update(list(range(step*8,step*8+8)),1042+step)
        smoke.append(dict(step=step+1,image_epsilon_loss=loss,gradients=grads,seconds=time.monotonic()-began))
    del model,opt;gc.collect();torch.cuda.empty_cache();model,opt=fresh()
    if tensor_state_sha(model)!=initial or opt.state:raise ValueError('Exact fresh random model and optimizer reset required')
    write(a.output/'smoke_report.json',dict(complete=True,steps=smoke,random_initialization_sha256=initial,
        readout_inheritance=False,condition_distillation=False,weights_and_optimizer_reset=True,
        decoder_gradients_absent=True,decoder_parameter_versions_unchanged=True,graph=graph,
        effective_batch=8,microbatch=2,formal_updates_completed=0))
    sweep=getattr(a,'epoch_sweep',False);partial=partial_training_updates(getattr(a,'partial_updates',None),sweep)
    epochs=1 if partial else requested_training_epochs(sweep);saved={}
    def save_requested(epoch,loss):
        checkpoint=dict(schema=profile['schema'],readout=model.state_dict(),epoch=epoch,step=epoch*78,
            generator_sha256=profile['generator_sha256'],pair_sha256=PAIRS,cache_sha256=cr['cache_sha256'],
            official_visual=data['official_visual'],TRAIN_image_epsilon_MSE=loss,source_commit=os.environ['T12_SOURCE_COMMIT'],
            condition_distillation=False,readout_inheritance=False,random_initialization_sha256=initial,
            requested_epoch_sweep=True,human_requested_epochs=list(REQUESTED_EPOCHS))
        path=a.output/('epoch'+str(epoch)+'_checkpoint.pt');torch.save(checkpoint,path)
        saved[str(epoch)]=dict(file=path.name,sha256=sha(path),epoch=epoch,updates=epoch*78)
        write(a.output/'requested_checkpoints.json',dict(complete=epoch==10,checkpoints=saved,selection_uses_TEST=False))
    if sweep:save_requested(0,None)
    history=[];best=float('inf');updates=0;rng=random.Random(42);schedule=hashlib.sha256();began=time.monotonic()
    for epoch in range(1,epochs+1):
        order=list(range(624));rng.shuffle(order);values=[]
        for start in range(0,624,8):
            indices=order[start:start+8];seed=rng.randrange(2**31)
            schedule.update(json.dumps([indices,seed]).encode());loss,_=update(indices,seed);updates+=1;values.append(loss)
            write(a.output/'progress.json',dict(stage='image_supervised_partial_updates'if partial else'image_supervised_requested_epoch_replay'if sweep else'image_supervised_frozen_decoder_50epochs',epoch=epoch,update=updates,total_updates=partial or epochs*78,
                epoch_update=len(values),epoch_updates=78,TRAIN_epsilon_recent=sum(values[-10:])/len(values[-10:]),seconds=time.monotonic()-began))
            if partial and updates==partial:break
        mean=sum(values)/len(values);history.append(dict(epoch=0 if partial else epoch,epoch_fraction=updates/78,updates=updates,TRAIN_image_epsilon_MSE=mean,seconds=time.monotonic()-began))
        write(a.output/'history.json',history)
        checkpoint=dict(schema=profile['schema'],readout=model.state_dict(),optimizer=opt.state_dict(),epoch=0 if partial else epoch,step=updates,
            generator_sha256=profile['generator_sha256'],pair_sha256=PAIRS,cache_sha256=cr['cache_sha256'],official_visual=data['official_visual'],
            TRAIN_image_epsilon_MSE=mean,source_commit=os.environ['T12_SOURCE_COMMIT'],training_objective='reviewed_TRAIN_image_epsilon_only',
            condition_distillation=False,readout_inheritance=False,random_initialization_sha256=initial,python_rng_state=rng.getstate(),
            explicit_partial_updates=partial,completed_epochs=0 if partial else epoch)
        torch.save(checkpoint,a.output/'last_checkpoint.pt')
        if mean<best:best=mean;torch.save(checkpoint,a.output/'best_checkpoint.pt')
        if sweep and epoch in REQUESTED_EPOCHS:save_requested(epoch,mean)
    if any(p._version!=versions[n]for n,p in unet.named_parameters()):raise ValueError('Decoder changed')
    write(a.output/'report.json',dict(complete=True,schema=profile['schema'],epochs=0 if partial else epochs,updates=updates,TRAIN_contexts=624,TRAIN_views=76,
        explicit_partial_updates=partial,epoch_fraction=updates/78,partial_TRAIN_examples=partial*8 if partial else None,
        seconds=time.monotonic()-began,best_sha256=sha(a.output/'best_checkpoint.pt'),last_sha256=sha(a.output/'last_checkpoint.pt'),
        generator_sha256=profile['generator_sha256'],generator_weights_modified=False,condition_distillation=False,readout_inheritance=False,
        Qwen_features_or_conditions_loaded=False,ResNet_encoder_and_BN_frozen=True,UNet_LoRA_and_VAE_frozen=True,
        trainable_frontend_parameters=sum(p.numel()for p in model.parameters()),frozen_visual_parameters=11176512,
        random_initialization_sha256=initial,TRAIN_schedule_sha256=schedule.hexdigest(),batch_size=8,microbatch=2,
        TEST_used_for_gradient=False,formal_checkpoint='partial_last'if partial else'requested_epochs_0_1_2_5_10'if sweep else'last50',
        requested_epochs=list(REQUESTED_EPOCHS)if sweep else None,requested_checkpoints=saved,
        source_commit=os.environ['T12_SOURCE_COMMIT']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['cache','fit'],required=True)
    for key in ['parent','data','output','model-path','adapter-path','decoder-assets']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--epoch-sweep',action='store_true',help='Explicit user-requested0/1/2/5/10 snapshots, same unchanged training recipe')
    p.add_argument('--partial-updates',type=int,help='Explicit fixed sub-epoch budget1..77, not a complete epoch')
    a=p.parse_args();partial_training_updates(a.partial_updates,a.epoch_sweep);a.profile=PROFILE
    from .scene_execution_gpu import configured_gpu
    gpu=configured_gpu()
    if a.output.exists()or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('Fresh run, published source and explicit GPU UUID required')
    fraction=float(os.environ.get('T12_CUDA_MEMORY_FRACTION','.8'))
    if not 0<fraction<=.8:raise ValueError('Bounded explicit CUDA memory fraction required')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(fraction,0)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__),GPU_UUID=gpu,GPU_name=torch.cuda.get_device_name(0)))
    try:globals()[a.stage](a)
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
