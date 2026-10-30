"""TRAIN whole-trajectory recovery with gradients through all student steps.

The fixed full electronic teacher and student start from identical fresh noise.
Teacher terminal latents and intermediate endpoints supervise the whole student
rollout; a teacher is not asked to denoise a possibly incorrect student basin.
Only existing task LoRA is trained. Internal latent sampling never returns an
image to Qwen. No scene GT, VAE encoder, DEV or TEST is used during training.
"""
import argparse,copy,json,math,os,random,sys,time,traceback
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from .scene_tail_distill import complete_state,load_cache,load_generator,predict
from .scene_onpolicy_tail import teacher_edges
from .scene_generation_data import request_seed
from .white_reference_scene_probe import GPU,digest,write


def rollout(scheduler,unet,unit_noise,cond,state,grid,gradient=False,capture=None):
    scheduler.set_timesteps(timesteps=grid,device=unit_noise.device)
    x=unit_noise*scheduler.init_noise_sigma;outputs={};sigmas=scheduler.sigmas.tolist()
    for j,t in enumerate(scheduler.timesteps):
        scheduler.scale_model_input(x,t)
        def forward(value,timestep=t,sigma=sigmas[j]):
            return predict(unet,value,timestep,cond,state,sigma,True)
        epsilon=checkpoint(forward,x,use_reentrant=False) if gradient else forward(x)
        x=scheduler.step(epsilon,t,x).prev_sample
        if capture is None or j+1 in capture:outputs[j+1]=x
    return x,outputs


def objective(student,teacher,student_points,teacher_points,edges):
    def error(x,y):
        assert x.shape==y.shape
        return (x.float()-y.float()).square().mean()/y.float().square().mean().clamp_min(1.)
    final=error(student,teacher)
    anchor=torch.stack([error(student_points[j+1],teacher_points[edges[j+1]]) for j in range(len(edges)-1)]).mean()
    return final+.1*anchor,final,anchor


def run(a):
    state,ready=complete_state(a.student);teacher,tr=complete_state(a.teacher)
    cache,cr=load_cache(a.student_cache);tc,tcr=load_cache(a.teacher_cache)
    assert state['sampling_contract']['CFG']==7.5 and state['sampling_contract']['IP_scale']==.25
    assert not teacher.get('optical_frontend') and not teacher.get('sampling_contract')
    assert state['pair_sha256']==teacher['pair_sha256']==cache['pair_sha256']==tc['pair_sha256']
    assert state['optical_frontend']['checkpoint_sha256']==cache['optical_frontend']['checkpoint_sha256']
    train={r['input_key']:r for r in cache['records'] if r['split']=='train'}
    assert set(train)=={r['input_key'] for r in tc['records'] if r['split']=='train'}
    assert not set(train)&{r['input_key'] for r in cache['records'] if r['split']=='dev'}
    student,head=load_generator(a,state,True);tp,th=load_generator(a,teacher,False)
    student.set_ip_adapter_scale(.25);tp.set_ip_adapter_scale(.25)
    reference_unet=None
    if a.fusion_aware:
        reference_unet=copy.deepcopy(student.unet).eval().requires_grad_(False)
        from peft.tuners.lora.layer import Linear
        for layer in reference_unet.modules():
            if isinstance(layer,Linear):layer.merge(safe_merge=True,adapter_names=['scene_pair'])
        from .scene_fusion_aware_tail import install
        state['fusion_aware_training']=install(student.unet)
    parameters=[p for p in student.unet.parameters() if p.requires_grad];assert parameters and not any(p.requires_grad for p in head.parameters())
    head_before={n:v.detach().cpu().clone() for n,v in head.state_dict().items()}
    conds={};teacher_conds={}
    with torch.no_grad():
        for key in train:
            for h,c,d in [(head,cache,conds),(th,tc,teacher_conds)]:
                f=c['features'][key];d[key]={n:v.detach() for n,v in h(f['hidden'][None].cuda().float(),f['visual'][None].cuda().float()).items()}
    tp.scheduler.set_timesteps(30,device='cuda');teacher_grid=tp.scheduler.timesteps.cpu().tolist();grid=state['sampling_contract']['timesteps'];edges=teacher_edges(teacher_grid,grid)
    groups={c:sorted(k for k,r in train.items() if r['category']==c) for c in ['lamp','pillow']};assert all(groups.values())
    if reference_unet is not None:
        key=groups['lamp'][0];x=torch.randn((1,4,64,64),device='cuda',dtype=torch.bfloat16)
        with torch.no_grad():
            actual=predict(student.unet,x,torch.tensor(grid[0],device='cuda'),conds[key],state,float(tp.scheduler.sigmas[0]),True)
            expected=predict(reference_unet,x,torch.tensor(grid[0],device='cuda'),conds[key],state,float(tp.scheduler.sigmas[0]),True)
        assert torch.equal(actual,expected),'Training forward differs from the actual official merged full UNet'
        write(a.output/'fusion_GPU_identity.json',dict(complete=True,full_UNet_CFG_prediction_exact=True,actual_TRAIN_conditions_used=True,teacher_or_TEST_not_modified=True))
        del reference_unet,actual,expected;torch.cuda.empty_cache()
    probe_keys=[k for c in groups for k in groups[c][:4]]
    def noise(key,index):
        seed=request_seed(train[key]['reference_sha256'],index)
        return torch.randn((1,4,64,64),generator=torch.Generator(device='cuda').manual_seed(seed),device='cuda',dtype=torch.bfloat16),seed
    def target(key,z):
        with torch.no_grad():return rollout(tp.scheduler,tp.unet,z,teacher_conds[key],teacher,teacher_grid,capture=set(edges[1:]))
    probes={}
    for key in probe_keys:
        z,seed=noise(key,9100000);end,_=target(key,z);probes[key]=(z,end.detach(),seed)
    def measure():
        with torch.no_grad():
            values=[]
            for key,(z,wanted,_) in probes.items():
                end,_=rollout(student.scheduler,student.unet,z,conds[key],state,grid)
                values.append(float((end.float()-wanted.float()).square().mean()/wanted.float().square().mean().clamp_min(1.)))
            return sum(values)/len(values)
    initial_probe=measure();initial=[p.detach().clone() for p in parameters];opt=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=.01);smoke=[];torch.cuda.reset_peak_memory_stats()
    for i in range(4):
        key=groups[['lamp','pillow'][i%2]][0];z,seed=noise(key,9200000+i);wanted,points=target(key,z)
        actual,own=rollout(student.scheduler,student.unet,z,conds[key],state,grid,gradient=True)
        loss,terminal,anchor=objective(actual,wanted,own,points,edges);opt.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);assert torch.isfinite(loss) and norm>0;opt.step()
        assert all(p.grad is None for p in tp.unet.parameters());smoke.append(dict(loss=float(loss),terminal_loss=float(terminal),gradient_norm=float(norm),source_noise_seed=seed))
    peak=torch.cuda.max_memory_allocated();assert peak<31*1024**3
    with torch.no_grad():
        for p,old in zip(parameters,initial):p.copy_(old)
    del initial,opt,actual,own,wanted,points,loss,terminal,anchor
    opt=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=.01)
    write(a.output/'smoke_report.json',dict(complete=True,actual_full_rollout_backward_steps=4,student_steps=len(grid),history=smoke,teacher_has_no_gradients=True,student_and_optimizer_reset=True,peak_allocated_bytes=peak,initial_fixed_TRAIN_terminal_loss=initial_probe))
    rng=random.Random(20261030);best=math.inf;history=[];began=time.monotonic();seeds=set();recent=[]
    for step in range(1,a.steps+1):
        key=rng.choice(groups[['lamp','pillow'][step%2]]);z,seed=noise(key,9300000+step);seeds.add(seed);wanted,points=target(key,z)
        actual,own=rollout(student.scheduler,student.unet,z,conds[key],state,grid,gradient=True)
        loss,terminal,anchor=objective(actual,wanted,own,points,edges)
        for group in opt.param_groups:group['lr']=a.lr*min(1.,step/20)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);assert torch.isfinite(loss);opt.step();recent.append(float(terminal))
        if step%10==0:write(a.output/'progress.json',dict(stage='TRAIN_full_student_rollout_same_fresh_noise_as_full_teacher',step=step,total=a.steps,recent_terminal_loss=sum(recent[-10:])/len(recent[-10:])))
        if step%50==0 or step==a.steps:
            score=measure();assert all(torch.equal(v,head.state_dict()[n].cpu()) for n,v in head_before.items())
            history.append(dict(step=step,fixed_TRAIN_terminal_loss=score,seconds=time.monotonic()-began));write(a.output/'history.json',history)
            result=dict(state,source_commit=os.environ['T12_SOURCE_COMMIT'],step=step,cache_sha256=cr['cache_sha256'],unet_lora={n:p.detach().cpu() for n,p in student.unet.named_parameters() if 'lora_' in n},full_rollout_teacher_sha256=tr['best_sha256'])
            result['optical_frontend']=dict(result['optical_frontend'],generator_not_updated_after_frontend_recovery=False,TRAIN_generator_tail_recovery=True)
            torch.save(result,a.output/'last_checkpoint.pt')
            if score<best:best=score;torch.save(result,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],steps=a.steps,best_sha256=digest(a.output/'best_checkpoint.pt'),parent_student_sha256=ready['best_sha256'],teacher_sha256=tr['best_sha256'],teacher_steps=30,student_steps=len(grid),sampling_contract=state['sampling_contract'],TRAIN_contexts=len(train),fresh_noise_seeds=len(seeds),fixed_TRAIN_probe_count=len(probes),initial_fixed_TRAIN_terminal_loss=initial_probe,best_TRAIN_terminal_loss=best,student_cache_sha256=cr['cache_sha256'],teacher_cache_sha256=tcr['cache_sha256'],trainable_existing_UNet_LoRA_parameters=sum(p.numel() for p in parameters),extra_deployment_parameters=0,condition_heads_Qwen_optics_VAE_frozen=True,gradient_through_all_student_steps=True,fusion_aware=a.fusion_aware,seconds=time.monotonic()-began,best_selection='Fixed TRAIN whole-trajectory terminal loss only; raw DEV quality still required',scene_GT_or_image_encoder_used=False,generated_images_returned_to_Qwen=False,DEV_or_TEST_forwarded=False,quality_or_speed_accepted=False,TEST_opened=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['student','teacher','student-cache','teacher-cache','model-path','adapter-path','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--fusion-aware',action='store_true');a=p.parse_args();a.no_gradient_checkpointing=True;a.adapt_head=False
    assert a.steps>=4 and a.lr>0 and not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);torch.manual_seed(1042)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__))))
    try:run(a)
    except Exception:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
