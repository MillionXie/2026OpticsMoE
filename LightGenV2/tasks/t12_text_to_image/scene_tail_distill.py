"""TRAIN-only progressive guided Euler trajectory distillation.

This is a candidate tail optimization, not an assertion of preserved quality.
Teacher uses full electronic Qwen conditions; the student uses the bound LG
conditions. No new deployed parameters, reference encoder or image feedback.
Related method: https://arxiv.org/abs/2210.03142 ; Euler step documentation:
https://huggingface.co/docs/diffusers/api/schedulers/euler
"""
import argparse
import copy
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


def interval_edges(teacher_steps,student_steps):
    if not 1<=student_steps<teacher_steps:raise ValueError('Require a smaller positive student budget')
    return [i*teacher_steps//student_steps for i in range(student_steps+1)]


def complete_state(folder):
    report=json.loads((folder/'report.json').read_text());assert report['complete']
    assert digest(folder/'best_checkpoint.pt')==report['best_sha256']
    state=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['schema']=='qwen3_scene_paired_v1' and state['variant']=='qwen' and state['step']>0
    assert state['resolution']==512
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
        if not getattr(a,'no_gradient_checkpointing',False):pipe.unet.enable_gradient_checkpointing()
        pipe.unet.train()
    head=QwenSDXLConditions(joint_image_context=state.get('joint_image_context',False)).cuda().eval().requires_grad_(False);head.load_state_dict(state['head'],strict=True)
    if trainable and a.adapt_head:head.train().requires_grad_(True)
    return pipe,head


def predict(unet,x,t,cond,state,sigma,guided):
    batch=len(x);device=x.device
    scale=((torch.as_tensor(sigma,device=device,dtype=torch.float32)**2+1)**.5)
    if scale.ndim and scale.numel()!=batch:raise ValueError('One sigma per original sample required')
    if guided:
        x=torch.cat([x,x]);text=torch.cat([state['negative_text'].to(device).expand(batch,-1,-1),cond['text']])
        pool=torch.cat([state['negative_pooled'].to(device).expand(batch,-1),cond['pooled']])
        images=torch.cat([state['negative_image'].to(device).expand(batch,-1,-1),cond['image']]).unsqueeze(1)
        if scale.ndim:scale=scale.repeat(2)
        t=torch.as_tensor(t,device=device)
        if t.ndim and t.numel()==batch:t=t.repeat(2)
    else:text,pool,images=cond['text'],cond['pooled'],cond['image'].unsqueeze(1)
    side=state.get('SDXL_conditioning_size',512)
    ids=torch.tensor([[side,side,0,0,side,side]],device=device,dtype=torch.bfloat16).repeat(len(x),1)
    # Match EulerDiscreteScheduler.scale_model_input dtype and rounding.
    if scale.ndim:scale=scale[:,None,None,None]
    scaled=x/scale
    out=unet(scaled.bfloat16(),t,
        encoder_hidden_states=text.bfloat16(),added_cond_kwargs={'text_embeds':pool.bfloat16(),
        'time_ids':ids,'image_embeds':[images.bfloat16()]}).sample
    if guided:
        u,c=out.chunk(2);return u+7.5*(c-u)
    return out


def student_sampling_contract(targets,guidance,ip_scale):
    if guidance not in [1.,7.5]:raise ValueError('Only conditional1 or preserved externalCFG7.5 is supported')
    if not 0<=ip_scale<=1:raise ValueError('Invalid reference strength')
    grid=targets['timesteps'];steps=targets['student_steps']
    if len(grid)!=steps or not all(a>b for a,b in zip(grid,grid[1:])):raise ValueError('Invalid student grid')
    kind='progressive_guided_Euler_v2' if guidance==1. else 'progressive_guided_Euler_CFGpreserved_v1'
    return dict(kind=kind,steps=steps,teacher_steps=targets['teacher_steps'],CFG=guidance,timesteps=grid,IP_scale=ip_scale)


def trajectories(a):
    teacher,identity=complete_state(a.teacher);payload,cache_report=load_cache(a.teacher_cache)
    assert (teacher.get('optical_frontend') or {}).get('checkpoint_sha256')==(payload.get('optical_frontend') or {}).get('checkpoint_sha256')
    assert teacher['pair_sha256']==payload['pair_sha256']
    approved=None
    if teacher.get('optical_frontend') or teacher.get('sampling_contract'):
        if a.teacher_quality_approval is None:raise ValueError('A distilled/LG teacher needs completed DEV quality approval')
        approved=json.loads(a.teacher_quality_approval.read_text())
        assert approved['approved_for_distillation_teacher'] and approved['all_32_reviewed']
        assert approved['checkpoint_sha256']==identity['best_sha256'] and not approved['TEST_used_for_selection']
    pipe,head=load_generator(a,teacher)
    assert type(pipe.scheduler).__name__=='EulerDiscreteScheduler' and pipe.scheduler.config.prediction_type=='epsilon'
    contexts={r['input_key']:r for r in payload['records'] if r['split']=='train'}
    dev={r['input_key'] for r in payload['records'] if r['split']=='dev'};assert not set(contexts)&dev
    if payload.get('TRAIN_trajectory_keys') is not None:
        keys=payload['TRAIN_trajectory_keys']
        if not keys or len(set(keys))!=len(keys) or not set(keys)<=set(contexts) or set(keys)&dev:
            raise ValueError('Trajectory subset must contain unique actual TRAIN inputs only')
        contexts={k:contexts[k] for k in keys}
    records=[];sigmas=None;timesteps=None
    contract=teacher.get('sampling_contract')
    teacher_steps=contract['steps'] if contract else a.teacher_steps
    student_steps=a.student_steps or teacher_steps//2
    edges=interval_edges(teacher_steps,student_steps)
    if (contract or {}).get('CFG',7.5) not in [1.,7.5]:raise ValueError('Unsupported teacher guidance')
    guided=(contract or {}).get('CFG',7.5)>1
    pipe.set_ip_adapter_scale((contract or {}).get('IP_scale',.25))
    for i,(key,row) in enumerate(contexts.items()):
        if contract:pipe.scheduler.set_timesteps(timesteps=contract['timesteps'],device='cuda')
        else:pipe.scheduler.set_timesteps(teacher_steps,device='cuda')
        sigmas=pipe.scheduler.sigmas.tolist();timesteps=pipe.scheduler.timesteps.cpu().tolist()
        generator=torch.Generator(device='cuda').manual_seed(request_seed(row['reference_sha256'],a.noise_index))
        x=torch.randn((1,4,64,64),generator=generator,device='cuda',dtype=torch.bfloat16)*pipe.scheduler.init_noise_sigma
        f=payload['features'][key]
        with torch.inference_mode():
            cond=head(f['hidden'].unsqueeze(0).cuda().float(),f['visual'].unsqueeze(0).cuda().float())
            for k in range(student_steps):
                start=x.clone()
                for j in range(edges[k],edges[k+1]):
                    # Calling scale_model_input initializes the scheduler index;
                    # predict applies the same scale to its CFG batch explicitly.
                    pipe.scheduler.scale_model_input(x,pipe.scheduler.timesteps[j])
                    epsilon=predict(pipe.unet,x,pipe.scheduler.timesteps[j],cond,teacher,sigmas[j],guided)
                    x=pipe.scheduler.step(epsilon,pipe.scheduler.timesteps[j],x).prev_sample
                target=collapsed_epsilon(start,x,sigmas[edges[k]],sigmas[edges[k+1]])
                assert torch.isfinite(target).all()
                records.append({'split':'train','input_key':key,'interval':k,'x':start[0].cpu(),
                    'target_epsilon':target[0].cpu(),'sigma':sigmas[edges[k]],'timestep':timesteps[edges[k]]})
        write(a.output/'progress.json',{'stage':'TRAIN_teacher_trajectory_targets','teacher_steps':teacher_steps,'student_steps':student_steps,'contexts':i+1,'total':len(contexts),'intervals':len(records)})
    grid=[timesteps[i] for i in edges[:-1]]
    check=type(pipe.scheduler).from_config(pipe.scheduler.config);check.set_timesteps(timesteps=grid,device='cuda')
    assert torch.allclose(check.sigmas.cpu(),torch.tensor([sigmas[i] for i in edges]),atol=1e-5,rtol=1e-5)
    saved={'records':records,'pair_sha256':payload['pair_sha256'],'teacher_sha256':identity['best_sha256'],
        'teacher_cache_sha256':cache_report['cache_sha256'],'teacher_steps':teacher_steps,'student_steps':student_steps,
        'timesteps':grid,'sigmas':[sigmas[i] for i in edges],'noise_index':a.noise_index}
    torch.save(saved,a.output/'TRAIN_trajectories.pt')
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],
        'trajectory_sha256':digest(a.output/'TRAIN_trajectories.pt'),'TRAIN_contexts':len(contexts),'TRAIN_intervals':len(records),
        'teacher_sha256':identity['best_sha256'],'teacher_steps':teacher_steps,'teacher_CFG':7.5 if guided else 1.,'student_steps':student_steps,'student_CFG':1.,
        'synthetic_scene_GT_used_for_trajectory_targets':False,'DEV_or_TEST_targets_created':False,
        'teacher_is_full_electronic_Qwen':not bool(teacher.get('optical_frontend')),'noise_index':a.noise_index,
        'teacher_quality_approval_sha256':digest(a.teacher_quality_approval) if approved else None,'quality_or_speed_accepted':False})


def train(a):
    state,parent=complete_state(a.student);payload,cache_report=load_cache(a.student_cache)
    assert (state.get('optical_frontend') or {}).get('checkpoint_sha256')==(payload.get('optical_frontend') or {}).get('checkpoint_sha256')
    ready=json.loads((a.trajectories/'report.json').read_text());assert ready['complete']
    assert digest(a.trajectories/'TRAIN_trajectories.pt')==ready['trajectory_sha256']
    targets=torch.load(a.trajectories/'TRAIN_trajectories.pt',map_location='cpu',weights_only=False)
    assert targets['pair_sha256']==state['pair_sha256']==payload['pair_sha256']
    train_keys={r['input_key'] for r in payload['records'] if r['split']=='train'}
    assert all(r['split']=='train' and r['input_key'] in train_keys for r in targets['records'])
    pipe,head=load_generator(a,state,True);pipe.set_ip_adapter_scale(a.ip_scale)
    guidance=getattr(a,'student_guidance',1.);sampling=student_sampling_contract(targets,guidance,a.ip_scale)
    # The frozen electronic control can use the same optimization budget. Its
    # initial head supplies TRAIN-only condition regularization; it is never
    # added to deployment. LG keeps the already audited electronic targets.
    teacher_head=copy.deepcopy(head).eval().requires_grad_(False) if a.adapt_head and not payload.get('TRAIN_electronic_conditions') else None
    def condition_regularizer(cond,row):
        assert row['split']=='train' and row['input_key'] in train_keys
        from .qwen3_scene_paired_train import electronic_condition_loss
        if teacher_head is None:
            return electronic_condition_loss(cond,payload,row)
        f=payload['features'][row['input_key']]
        with torch.no_grad():wanted=teacher_head(f['hidden'].unsqueeze(0).cuda().float(),f['visual'].unsqueeze(0).cuda().float())
        local={'TRAIN_electronic_conditions':{row['input_key']:{n:v[0].bfloat16().detach() for n,v in wanted.items()}}}
        return electronic_condition_loss(cond,local,row)
    lora=[p for p in pipe.unet.parameters() if p.requires_grad];assert lora
    # Re-loading from ordinary serialization removes inference-tensor status.
    conds={}
    if not a.adapt_head:
        with torch.no_grad():
            for key in train_keys:
                f=payload['features'][key];out=head(f['hidden'].unsqueeze(0).cuda().float(),f['visual'].unsqueeze(0).cuda().float())
                conds[key]={n:v.detach().clone() for n,v in out.items()}
    def loss_for(row):
        assert row['split']=='train'
        x=row['x'].unsqueeze(0).cuda();wanted=row['target_epsilon'].unsqueeze(0).cuda()
        if a.adapt_head:
            f=payload['features'][row['input_key']];cond=head(f['hidden'].unsqueeze(0).cuda().float(),f['visual'].unsqueeze(0).cuda().float())
        else:cond=conds[row['input_key']]
        pred=predict(pipe.unet,x,torch.tensor(row['timestep'],device='cuda'),cond,state,row['sigma'],guidance>1)
        main=(pred.float()-wanted).square().mean()/wanted.square().mean().clamp_min(1.)
        if a.adapt_head:
            return main+.05*condition_regularizer(cond,dict(row,split='train')),main
        return main,main
    def loss_batch(rows):
        if len(rows)==1:return loss_for(rows[0])
        from .scene_tail_batches import pack_features,interval_loss
        assert all(r['split']=='train' for r in rows)
        if a.adapt_head:
            h,v,padding=pack_features([payload['features'][r['input_key']] for r in rows],'cuda');cond=head(h,v,padding)
        else:cond={name:torch.cat([conds[r['input_key']][name] for r in rows]) for name in ['text','image','pooled']}
        x=torch.stack([r['x'] for r in rows]).cuda();wanted=torch.stack([r['target_epsilon'] for r in rows]).cuda()
        predicted=predict(pipe.unet,x,torch.tensor([r['timestep'] for r in rows],device='cuda'),cond,state,[r['sigma'] for r in rows],guidance>1)
        main=interval_loss(predicted,wanted)
        if a.adapt_head:
            kd=sum(condition_regularizer({n:v[i:i+1] for n,v in cond.items()},row) for i,row in enumerate(rows))/len(rows)
            return main+.05*kd,main
        return main,main
    trainable=lora+(list(head.parameters()) if a.adapt_head else [])
    def optimizer():
        groups=[{'params':lora,'lr':a.lr}]
        if a.adapt_head:groups.append({'params':head.parameters(),'lr':a.lr*.25})
        return torch.optim.AdamW(groups,weight_decay=.01)
    initial=[p.detach().clone() for p in trainable];opt=optimizer();smoke=[];torch.cuda.reset_peak_memory_stats()
    for step in range(4):
        rows=targets['records'][step*a.batch_size:(step+1)*a.batch_size]
        opt.zero_grad();loss,main=loss_batch(rows);direct={}
        if a.adapt_head:
            gradients=torch.autograd.grad(main,[head.text.out.weight,head.image.out.weight],retain_graph=True)
            direct={n:g.abs().sum().item() for n,g in zip(['text','image'],gradients)};assert all(math.isfinite(v) and v>0 for v in direct.values())
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss) and float(norm)>0;opt.step();smoke.append({'step':step+1,'loss':loss.item(),'gradient_norm':float(norm),'trajectory_only_head_gradient':direct})
    write(a.output/'smoke_report.json',{'complete':True,'real_backward_steps':smoke,'all_TRAIN':True,'reset_before_training':True,
        'batch_size':a.batch_size,'gradient_checkpointing':not a.no_gradient_checkpointing,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    with torch.no_grad():
        for p,old in zip(trainable,initial):p.copy_(old)
    del initial,opt;opt=optimizer();rng=random.Random(1042);rows=list(targets['records']);history=[];updates=0;best=math.inf
    if a.stage=='smoke':epochs=0
    else:epochs=a.epochs
    for epoch in range(1,epochs+1):
        rng.shuffle(rows);running=[];began=time.monotonic()
        sizes=[];total_updates=math.ceil(len(rows)/a.batch_size)*epochs
        for offset in range(0,len(rows),a.batch_size):
            batch=rows[offset:offset+a.batch_size]
            opt.zero_grad();loss,_=loss_batch(batch);loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            assert torch.isfinite(loss);opt.step();updates+=1;running.append(loss.item());sizes.append(len(batch))
            for index,group in enumerate(opt.param_groups):group['lr']=a.lr*(1 if index==0 else .25)*(.1+.9*.5*(1+math.cos(math.pi*updates/total_updates)))
            if updates%20==0:write(a.output/'progress.json',{'stage':'TRAIN_progressive_guided_Euler_distillation','epoch':epoch,'update':updates,'total':total_updates,'batch_size':a.batch_size,'recent_loss':sum(running[-20:])/len(running[-20:])})
        score=sum(x*n for x,n in zip(running,sizes))/sum(sizes);history.append({'epoch':epoch,'TRAIN_loss':score,'seconds':time.monotonic()-began});write(a.output/'history.json',history)
        result=dict(state,step=updates,epoch=epoch,source_commit=os.environ['T12_SOURCE_COMMIT'],
            unet_lora={n:p.detach().cpu() for n,p in pipe.unet.named_parameters() if 'lora_' in n},
            head={n:p.detach().cpu() for n,p in head.state_dict().items()},
            sampling_contract=sampling,
            trajectory_sha256=ready['trajectory_sha256'])
        result.pop('DEV_epsilon_MSE',None)
        torch.save(result,a.output/'last_checkpoint.pt')
        if score<best:best=score;torch.save(result,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],'stage':a.stage,
        'best_sha256':digest(a.output/'best_checkpoint.pt') if epochs else None,'actual_paired_updates':updates,
        'TRAIN_intervals':len(rows),'TRAIN_contexts':len(train_keys),'epochs':epochs,'trainable_UNet_LoRA_parameters':sum(p.numel() for p in lora),
        'trainable_existing_head_parameters':sum(p.numel() for p in head.parameters() if p.requires_grad),
        'training_batch_size':a.batch_size,'gradient_checkpointing':not a.no_gradient_checkpointing,
        'student_CFG':guidance,'effective_epsilon_targets_independent_of_student_CFG':True,
        'trajectory_creation_student_CFG_metadata':ready['student_CFG'],
        'head_frozen':not a.adapt_head,'Qwen_optical_VAE_frozen':True,'extra_deployment_parameters':0,'trajectory_sha256':ready['trajectory_sha256'],
        'parent_checkpoint_sha256':parent['best_sha256'],'student_cache_sha256':cache_report['cache_sha256'],
        'best_selection':'TRAIN trajectory loss only; DEV generation evaluation still required','quality_or_speed_accepted':False,'TEST_opened':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['trajectories','smoke','train'],required=True)
    for name in ['teacher','teacher-cache','student','student-cache','trajectories','model-path','adapter-path','output']:
        p.add_argument('--'+name,type=Path,required=name in ['model-path','adapter-path','output'])
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--ip-scale',type=float,default=.25)
    p.add_argument('--teacher-steps',type=int,choices=[12,30],default=12);p.add_argument('--student-steps',type=int)
    p.add_argument('--noise-index',type=int,default=137);p.add_argument('--adapt-head',action='store_true')
    p.add_argument('--student-guidance',type=float,choices=[1.,7.5],default=1.,help='Keep external CFG7.5 while distilling step intervals, or legacy conditional1')
    p.add_argument('--batch-size',type=int,choices=[1,2],default=1);p.add_argument('--no-gradient-checkpointing',action='store_true')
    p.add_argument('--teacher-quality-approval',type=Path)
    a=p.parse_args();assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);torch.manual_seed(1042)
    write(a.output/'execution.json',{'argv':sys.argv,'source_commit':os.environ['T12_SOURCE_COMMIT'],'source_sha256':digest(Path(__file__)),'pid':os.getpid()})
    try:trajectories(a) if a.stage=='trajectories' else train(a)
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
