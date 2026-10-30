"""Fresh-noise TRAIN local teacher supervision on the student's latent states.

Only existing UNet LoRA is updated. Frozen Qwen features and condition heads
remain the model's original RGB/text interface. Latents never feed back into
Qwen; no scene GT or VAE image encoder is used. The teacher is the accepted full
electronic thirty-step generator, not an unaccepted LG output distribution.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import torch
from .scene_tail_distill import complete_state,load_cache,load_generator,predict,collapsed_epsilon
from .scene_tail_batches import interval_loss
from .scene_generation_data import request_seed
from .white_reference_scene_probe import GPU,digest,write


def teacher_edges(teacher_grid,student_grid):
    if not student_grid or len(set(student_grid))!=len(student_grid) or not all(a>b for a,b in zip(student_grid,student_grid[1:])):
        raise ValueError('Unique descending student grid required')
    try:edges=[teacher_grid.index(t) for t in student_grid]+[len(teacher_grid)]
    except ValueError as exc:raise ValueError('Student grid must be accepted teacher endpoints') from exc
    if edges[0]!=0 or not all(a<b for a,b in zip(edges,edges[1:])):raise ValueError('Student must retain the first teacher endpoint')
    return edges


def local_teacher_endpoint(scheduler,unet,start,cond,state,begin,end):
    # set_begin_index alone does not reset a scheduler already used by the
    # preceding local target. Use the public grid setter to reset its state.
    grid=scheduler.timesteps.detach().cpu().tolist()
    scheduler.set_timesteps(timesteps=grid,device=start.device)
    scheduler.set_begin_index(begin);x=start.detach().clone();sigmas=scheduler.sigmas.tolist()
    with torch.no_grad():
        for j in range(begin,end):
            t=scheduler.timesteps[j];scheduler.scale_model_input(x,t)
            epsilon=predict(unet,x,t,cond,state,sigmas[j],True)
            x=scheduler.step(epsilon,t,x).prev_sample
    return x.detach()


def run(a):
    state,initial_report=complete_state(a.student);teacher,teacher_report=complete_state(a.teacher)
    cache,cache_report=load_cache(a.student_cache);tc,tc_report=load_cache(a.teacher_cache)
    assert not teacher.get('optical_frontend') and not teacher.get('sampling_contract')
    assert state['sampling_contract']['CFG']==7.5 and state['sampling_contract']['IP_scale']==.25
    assert (state.get('optical_frontend') or {}).get('checkpoint_sha256')==(cache.get('optical_frontend') or {}).get('checkpoint_sha256')
    assert state['pair_sha256']==teacher['pair_sha256']==cache['pair_sha256']==tc['pair_sha256']
    train={r['input_key']:r for r in cache['records'] if r['split']=='train'};teacher_keys={r['input_key'] for r in tc['records'] if r['split']=='train'}
    assert set(train)==teacher_keys and not set(train)&{r['input_key'] for r in cache['records'] if r['split']=='dev'}
    student,head=load_generator(a,state,True);tp,th=load_generator(a,teacher,False)
    student.set_ip_adapter_scale(.25);tp.set_ip_adapter_scale(.25)
    parameters=[p for p in student.unet.parameters() if p.requires_grad];assert parameters and not any(p.requires_grad for p in head.parameters())
    head_before={n:v.detach().cpu().clone() for n,v in head.state_dict().items()}
    conds={};teacher_conds={}
    with torch.no_grad():
        for key in train:
            for h,payload,destination in [(head,cache,conds),(th,tc,teacher_conds)]:
                f=payload['features'][key];out=h(f['hidden'][None].cuda().float(),f['visual'][None].cuda().float())
                destination[key]={n:v.detach().clone() for n,v in out.items()}
    tp.scheduler.set_timesteps(30,device='cuda');full_grid=tp.scheduler.timesteps.cpu().tolist();edges=teacher_edges(full_grid,state['sampling_contract']['timesteps']);full_sigmas=tp.scheduler.sigmas.tolist()
    student.scheduler.set_timesteps(timesteps=state['sampling_contract']['timesteps'],device='cuda')
    assert torch.allclose(student.scheduler.sigmas.cpu(),torch.tensor([full_sigmas[i] for i in edges]),atol=1e-5,rtol=1e-5)
    targets_report=json.loads((a.probe_trajectories/'report.json').read_text());assert targets_report['complete'] and targets_report['teacher_sha256']==teacher_report['best_sha256']
    assert digest(a.probe_trajectories/'TRAIN_trajectories.pt')==targets_report['trajectory_sha256']
    bank=torch.load(a.probe_trajectories/'TRAIN_trajectories.pt',map_location='cpu',weights_only=False,mmap=True)
    assert bank['timesteps']==state['sampling_contract']['timesteps'] and bank['pair_sha256']==state['pair_sha256']
    groups={category:sorted(k for k,row in train.items() if row['category']==category) for category in ['lamp','pillow']};assert all(groups.values())
    probe_keys={key for keys in groups.values() for key in keys[:4]}
    probes=[row for row in bank['records'] if row['input_key'] in probe_keys and row['interval'] in [0,(len(edges)-1)//2,len(edges)-2]]
    assert len(probes)==24 and all(row['split']=='train' and row['input_key'] in train for row in probes)
    def loss_for(key,k,noise_index):
        student.scheduler.set_timesteps(timesteps=state['sampling_contract']['timesteps'],device='cuda')
        seed=request_seed(train[key]['reference_sha256'],noise_index);generator=torch.Generator(device='cuda').manual_seed(seed)
        with torch.no_grad():
            x=torch.randn((1,4,64,64),generator=generator,device='cuda',dtype=torch.bfloat16)*student.scheduler.init_noise_sigma
            for j in range(k):
                t=student.scheduler.timesteps[j];student.scheduler.scale_model_input(x,t)
                epsilon=predict(student.unet,x,t,conds[key],state,float(student.scheduler.sigmas[j]),True)
                x=student.scheduler.step(epsilon,t,x).prev_sample
            start=x.detach().clone();end=local_teacher_endpoint(tp.scheduler,tp.unet,start,teacher_conds[key],teacher,edges[k],edges[k+1])
            wanted=collapsed_epsilon(start,end,full_sigmas[edges[k]],full_sigmas[edges[k+1]])
        predicted=predict(student.unet,start,student.scheduler.timesteps[k],conds[key],state,full_sigmas[edges[k]],True)
        return interval_loss(predicted,wanted),seed
    initial=[p.detach().clone() for p in parameters];opt=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=.01);smoke=[];torch.cuda.reset_peak_memory_stats()
    for i in range(4):
        key=groups[['lamp','pillow'][i%2]][0];loss,seed=loss_for(key,i%len(state['sampling_contract']['timesteps']),900000+i)
        opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss) and norm>0;opt.step();assert all(p.grad is None for p in tp.unet.parameters())
        smoke.append(dict(loss=float(loss),gradient_norm=float(norm),fresh_source_noise_seed=seed))
    peak=torch.cuda.max_memory_allocated();assert peak<31*1024**3
    with torch.no_grad():
        for p,old in zip(parameters,initial):p.copy_(old)
    del initial,opt;opt=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=.01)
    write(a.output/'smoke_report.json',dict(complete=True,real_backward_steps=4,history=smoke,teacher_gradients_absent=True,parameters_and_optimizer_reset=True,peak_allocated_bytes=peak))
    rng=random.Random(20261030);history=[];best=math.inf;began=time.monotonic();seeds=set();recent=[]
    for step in range(1,a.steps+1):
        category=['lamp','pillow'][step%2];key=rng.choice(groups[category]);k=rng.randrange(len(edges)-1)
        loss,seed=loss_for(key,k,1000000+step);seeds.add(seed);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss);opt.step();recent.append(float(loss))
        for group in opt.param_groups:group['lr']=a.lr*min(1.,step/50)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        if step%20==0:write(a.output/'progress.json',dict(stage='TRAIN_fresh_noise_student_rollout_local_electronic_teacher',step=step,total=a.steps,recent_loss=sum(recent[-20:])/len(recent[-20:])))
        if step%200==0 or step==a.steps:
            with torch.no_grad():
                values=[float(interval_loss(predict(student.unet,row['x'][None].cuda(),torch.tensor(row['timestep'],device='cuda'),conds[row['input_key']],state,row['sigma'],True),row['target_epsilon'][None].cuda())) for row in probes]
            score=sum(values)/len(values);history.append(dict(step=step,fixed_TRAIN_teacher_state_probe_loss=score,onpolicy_TRAIN_recent_loss=sum(recent[-200:])/len(recent[-200:]),seconds=time.monotonic()-began));write(a.output/'history.json',history)
            assert all(torch.equal(v,head.state_dict()[n].cpu()) for n,v in head_before.items())
            result=dict(state,source_commit=os.environ['T12_SOURCE_COMMIT'],step=step,cache_sha256=cache_report['cache_sha256'],
                unet_lora={n:p.detach().cpu() for n,p in student.unet.named_parameters() if 'lora_' in n},
                onpolicy_teacher_sha256=teacher_report['best_sha256'])
            if result.get('optical_frontend'):
                result['optical_frontend']=dict(result['optical_frontend'],generator_not_updated_after_frontend_recovery=False,TRAIN_generator_tail_recovery=True)
            torch.save(result,a.output/'last_checkpoint.pt')
            if score<best:best=score;torch.save(result,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],steps=a.steps,best_sha256=digest(a.output/'best_checkpoint.pt'),
        parent_student_sha256=initial_report['best_sha256'],teacher_sha256=teacher_report['best_sha256'],teacher_fused=False,teacher_steps=30,
        sampling_contract=state['sampling_contract'],actual_teacher_step_edges=edges,TRAIN_contexts=len(train),fresh_source_noise_seeds=len(seeds),
        student_cache_sha256=cache_report['cache_sha256'],teacher_cache_sha256=tc_report['cache_sha256'],
        fixed_TRAIN_probe_trajectory_sha256=targets_report['trajectory_sha256'],fixed_TRAIN_probe_count=len(probes),
        trainable_existing_UNet_LoRA_parameters=sum(p.numel() for p in parameters),extra_deployment_parameters=0,
        condition_heads_Qwen_optics_VAE_frozen=True,scene_GT_or_image_encoder_used=False,latent_states_not_returned_to_Qwen=True,
        best_selection='Fixed TRAIN teacher-state probe loss; DEV raw generation still required',seconds=time.monotonic()-began,
        DEV_or_TEST_forwarded=False,quality_or_speed_accepted=False,TEST_opened=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['student','teacher','student-cache','teacher-cache','probe-trajectories','model-path','adapter-path','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--gradient-checkpointing',action='store_true');a=p.parse_args();a.no_gradient_checkpointing=not a.gradient_checkpointing;a.adapt_head=False
    assert a.steps>=4 and a.lr>0 and not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);torch.manual_seed(1042)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__))))
    try:run(a)
    except Exception:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
