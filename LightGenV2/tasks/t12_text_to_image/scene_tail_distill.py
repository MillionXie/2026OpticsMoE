"""TRAIN-only guided Euler12 -> conditional Euler6 trajectory distillation.

This is a candidate tail optimization, not an assertion of preserved quality.
Teacher uses full electronic Qwen conditions; the student uses the bound LG
conditions. No new deployed parameters, reference encoder or image feedback.
Related method: https://arxiv.org/abs/2210.03142 ; Euler step documentation:
https://huggingface.co/docs/diffusers/api/schedulers/euler
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import torch
from .qwen3_scene_conditioning import QwenSDXLConditions
from .qwen3_scene_paired_train import pipeline
from .scene_generation_data import request_seed
from .white_reference_scene_probe import GPU,digest,write


def collapsed_epsilon(start,end,sigma_start,sigma_end):
    if not float(sigma_start)>float(sigma_end)>=0:raise ValueError('Require descending Euler interval')
    return (end.float()-start.float())/(float(sigma_end)-float(sigma_start))


def complete_state(folder):
    report=json.loads((folder/'report.json').read_text());assert report['complete']
    assert digest(folder/'best_checkpoint.pt')==report['best_sha256']
    state=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['schema']=='qwen3_scene_paired_v1' and state['variant']=='qwen' and state['step']>0
    assert state['resolution']==512 and not state.get('sampling_contract')
    return state,report


def load_cache(folder):
    report=json.loads((folder/'report.json').read_text());assert report['complete']
    assert digest(folder/'paired_cache.pt')==report['cache_sha256']
    return torch.load(folder/'paired_cache.pt',map_location='cpu',weights_only=False,mmap=True),report


def load_generator(a,state,trainable=False):
    from peft import LoraConfig
    pipe=pipeline(a);pipe.vae.cpu();pipe.unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',
        target_modules=['to_q','to_k','to_v','to_out.0']),adapter_name='scene_pair')
    named=dict(pipe.unet.named_parameters());assert set(state['unet_lora'])=={n for n in named if 'lora_' in n}
    with torch.no_grad():
        for n,value in state['unet_lora'].items():named[n].copy_(value)
    pipe.unet.requires_grad_(False)
    if trainable:
        for n,p in pipe.unet.named_parameters():
            if 'lora_' in n:p.requires_grad_(True);p.data=p.data.float()
        pipe.unet.enable_gradient_checkpointing();pipe.unet.train()
    head=QwenSDXLConditions().cuda().eval().requires_grad_(False);head.load_state_dict(state['head'],strict=True)
    return pipe,head


def predict(unet,x,t,cond,state,sigma,guided):
    if guided:
        x=torch.cat([x,x]);text=torch.cat([state['negative_text'].cuda(),cond['text']]);pool=torch.cat([state['negative_pooled'].cuda(),cond['pooled']])
        images=torch.cat([state['negative_image'].cuda(),cond['image']]).unsqueeze(1)
    else:text,pool,images=cond['text'],cond['pooled'],cond['image'].unsqueeze(1)
    side=state.get('SDXL_conditioning_size',512)
    ids=torch.tensor([[side,side,0,0,side,side]],device='cuda',dtype=torch.bfloat16).repeat(len(x),1)
    out=unet((x.float()/math.sqrt(float(sigma)**2+1)).bfloat16(),t,
        encoder_hidden_states=text.bfloat16(),added_cond_kwargs={'text_embeds':pool.bfloat16(),
        'time_ids':ids,'image_embeds':[images.bfloat16()]}).sample
    if guided:
        u,c=out.chunk(2);return u+7.5*(c-u)
    return out


def trajectories(a):
    teacher,identity=complete_state(a.teacher);payload,cache_report=load_cache(a.teacher_cache)
    assert not teacher.get('optical_frontend') and not payload.get('optical_frontend')
    assert teacher['pair_sha256']==payload['pair_sha256']
    pipe,head=load_generator(a,teacher)
    assert type(pipe.scheduler).__name__=='EulerDiscreteScheduler' and pipe.scheduler.config.prediction_type=='epsilon'
    contexts={r['input_key']:r for r in payload['records'] if r['split']=='train'}
    dev={r['input_key'] for r in payload['records'] if r['split']=='dev'};assert not set(contexts)&dev
    records=[];sigmas=None;timesteps=None
    for i,(key,row) in enumerate(contexts.items()):
        pipe.scheduler.set_timesteps(12,device='cuda');sigmas=pipe.scheduler.sigmas.tolist();timesteps=pipe.scheduler.timesteps.cpu().tolist()
        generator=torch.Generator(device='cuda').manual_seed(request_seed(row['reference_sha256'],137))
        x=torch.randn((1,4,64,64),generator=generator,device='cuda',dtype=torch.bfloat16)*pipe.scheduler.init_noise_sigma
        f=payload['features'][key]
        with torch.inference_mode():
            cond=head(f['hidden'].unsqueeze(0).cuda().float(),f['visual'].unsqueeze(0).cuda().float())
            for k in range(6):
                start=x.clone()
                for j in [2*k,2*k+1]:
                    # Calling scale_model_input initializes the scheduler index;
                    # predict applies the same scale to its CFG batch explicitly.
                    pipe.scheduler.scale_model_input(x,pipe.scheduler.timesteps[j])
                    epsilon=predict(pipe.unet,x,pipe.scheduler.timesteps[j],cond,teacher,sigmas[j],True)
                    x=pipe.scheduler.step(epsilon,pipe.scheduler.timesteps[j],x).prev_sample
                target=collapsed_epsilon(start,x,sigmas[2*k],sigmas[2*k+2])
                assert torch.isfinite(target).all()
                records.append({'split':'train','input_key':key,'interval':k,'x':start[0].cpu(),
                    'target_epsilon':target[0].cpu(),'sigma':sigmas[2*k],'timestep':timesteps[2*k]})
        write(a.output/'progress.json',{'stage':'TRAIN_electronic12_trajectory_targets','contexts':i+1,'total':len(contexts),'intervals':len(records)})
    grid=timesteps[::2]
    check=type(pipe.scheduler).from_config(pipe.scheduler.config);check.set_timesteps(timesteps=grid,device='cuda')
    assert torch.allclose(check.sigmas.cpu(),torch.tensor(sigmas[::2]),atol=1e-5,rtol=1e-5)
    saved={'records':records,'pair_sha256':payload['pair_sha256'],'teacher_sha256':identity['best_sha256'],
        'teacher_cache_sha256':cache_report['cache_sha256'],'teacher_steps':12,'timesteps':grid,'sigmas':sigmas[::2]}
    torch.save(saved,a.output/'TRAIN_trajectories.pt')
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],
        'trajectory_sha256':digest(a.output/'TRAIN_trajectories.pt'),'TRAIN_contexts':len(contexts),'TRAIN_intervals':len(records),
        'teacher_sha256':identity['best_sha256'],'teacher_steps':12,'teacher_CFG':7.5,'student_steps':6,'student_CFG':1.,
        'synthetic_scene_GT_used_for_trajectory_targets':False,'DEV_or_TEST_targets_created':False,
        'teacher_is_full_electronic_Qwen':True,'quality_or_speed_accepted':False})


def train(a):
    state,parent=complete_state(a.student);payload,cache_report=load_cache(a.student_cache)
    assert payload.get('optical_frontend') and state['optical_frontend']['checkpoint_sha256']==payload['optical_frontend']['checkpoint_sha256']
    ready=json.loads((a.trajectories/'report.json').read_text());assert ready['complete']
    assert digest(a.trajectories/'TRAIN_trajectories.pt')==ready['trajectory_sha256']
    targets=torch.load(a.trajectories/'TRAIN_trajectories.pt',map_location='cpu',weights_only=False)
    assert targets['pair_sha256']==state['pair_sha256']==payload['pair_sha256']
    train_keys={r['input_key'] for r in payload['records'] if r['split']=='train'}
    assert all(r['split']=='train' and r['input_key'] in train_keys for r in targets['records'])
    pipe,head=load_generator(a,state,True);pipe.set_ip_adapter_scale(a.ip_scale)
    lora=[p for p in pipe.unet.parameters() if p.requires_grad];assert lora
    # Re-loading from ordinary serialization removes inference-tensor status.
    conds={}
    with torch.no_grad():
        for key in train_keys:
            f=payload['features'][key];out=head(f['hidden'].unsqueeze(0).cuda().float(),f['visual'].unsqueeze(0).cuda().float())
            conds[key]={n:v.detach().clone() for n,v in out.items()}
    def loss_for(row):
        assert row['split']=='train'
        x=row['x'].unsqueeze(0).cuda();wanted=row['target_epsilon'].unsqueeze(0).cuda()
        pred=predict(pipe.unet,x,torch.tensor(row['timestep'],device='cuda'),conds[row['input_key']],state,row['sigma'],False)
        return (pred.float()-wanted).square().mean()/wanted.square().mean().clamp_min(1.)
    def optimizer():return torch.optim.AdamW(lora,lr=a.lr,weight_decay=.01)
    initial=[p.detach().clone() for p in lora];opt=optimizer();smoke=[]
    for step,row in enumerate(targets['records'][:4]):
        opt.zero_grad();loss=loss_for(row);loss.backward();norm=torch.nn.utils.clip_grad_norm_(lora,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss) and float(norm)>0;opt.step();smoke.append({'step':step+1,'loss':loss.item(),'gradient_norm':float(norm)})
    write(a.output/'smoke_report.json',{'complete':True,'real_backward_steps':smoke,'all_TRAIN':True,'reset_before_training':True})
    with torch.no_grad():
        for p,old in zip(lora,initial):p.copy_(old)
    del initial,opt;opt=optimizer();rng=random.Random(1042);rows=list(targets['records']);history=[];updates=0;best=math.inf
    if a.stage=='smoke':epochs=0
    else:epochs=a.epochs
    for epoch in range(1,epochs+1):
        rng.shuffle(rows);running=[];began=time.monotonic()
        for row in rows:
            opt.zero_grad();loss=loss_for(row);loss.backward();torch.nn.utils.clip_grad_norm_(lora,1.,error_if_nonfinite=True)
            assert torch.isfinite(loss);opt.step();updates+=1;running.append(loss.item())
            for group in opt.param_groups:group['lr']=a.lr*(.1+.9*.5*(1+math.cos(math.pi*updates/(len(rows)*epochs))))
            if updates%20==0:write(a.output/'progress.json',{'stage':'TRAIN_guided12_to_conditional6_distillation','epoch':epoch,'update':updates,'total':len(rows)*epochs,'recent_loss':sum(running[-20:])/len(running[-20:])})
        score=sum(running)/len(running);history.append({'epoch':epoch,'TRAIN_loss':score,'seconds':time.monotonic()-began});write(a.output/'history.json',history)
        result=dict(state,step=updates,epoch=epoch,source_commit=os.environ['T12_SOURCE_COMMIT'],
            unet_lora={n:p.detach().cpu() for n,p in pipe.unet.named_parameters() if 'lora_' in n},
            sampling_contract={'kind':'guided_Euler12_to_conditional6_v1','steps':6,'CFG':1.,'timesteps':targets['timesteps'],'IP_scale':a.ip_scale},
            trajectory_sha256=ready['trajectory_sha256'])
        torch.save(result,a.output/'last_checkpoint.pt')
        if score<best:best=score;torch.save(result,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],'stage':a.stage,
        'best_sha256':digest(a.output/'best_checkpoint.pt') if epochs else None,'actual_paired_updates':updates,
        'TRAIN_intervals':len(rows),'TRAIN_contexts':len(train_keys),'epochs':epochs,'trainable_UNet_LoRA_parameters':sum(p.numel() for p in lora),
        'head_and_Qwen_and_optical_and_VAE_frozen':True,'extra_deployment_parameters':0,'trajectory_sha256':ready['trajectory_sha256'],
        'parent_checkpoint_sha256':parent['best_sha256'],'student_cache_sha256':cache_report['cache_sha256'],
        'best_selection':'TRAIN trajectory loss only; DEV generation evaluation still required','quality_or_speed_accepted':False,'TEST_opened':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['trajectories','smoke','train'],required=True)
    for name in ['teacher','teacher-cache','student','student-cache','trajectories','model-path','adapter-path','output']:
        p.add_argument('--'+name,type=Path,required=name in ['model-path','adapter-path','output'])
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--ip-scale',type=float,default=.25)
    a=p.parse_args();assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);torch.manual_seed(1042)
    write(a.output/'execution.json',{'argv':sys.argv,'source_commit':os.environ['T12_SOURCE_COMMIT'],'source_sha256':digest(Path(__file__)),'pid':os.getpid()})
    try:trajectories(a) if a.stage=='trajectories' else train(a)
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
