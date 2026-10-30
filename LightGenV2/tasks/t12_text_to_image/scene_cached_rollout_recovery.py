"""Recover compact LG using gradients through its complete internal rollout.

Fixed accepted electronic30 TRAIN paths supply exact initial/terminal states.
No teacher GPU regeneration, reference VAE, image feedback or extra deployment
parameter is used. This is an optional refinement, not a quality claim.
"""
import argparse,copy,json,math,os,random,sys,time,traceback
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from .scene_tail_distill import complete_state,load_cache,load_generator,predict
from .scene_depth_probe import prune_unet
from .scene_depth_recovery import expand_lora
from .scene_fusion_aware_tail import install
from .scene_rollout_tail import objective
from .white_reference_scene_probe import GPU,digest,write


def rollout_from_scaled(scheduler,unet,initial,cond,state,grid,gradient=False):
    """Use exact saved scaled BF16 noise; never divide and re-multiply sigma."""
    scheduler.set_timesteps(timesteps=grid,device=initial.device)
    x=initial;points={};sigmas=scheduler.sigmas.tolist()
    for j,t in enumerate(scheduler.timesteps):
        scheduler.scale_model_input(x,t)
        def forward(value,timestep=t,sigma=sigmas[j]):return predict(unet,value,timestep,cond,state,sigma,True)
        epsilon=checkpoint(forward,x,use_reentrant=False)if gradient else forward(x)
        x=scheduler.step(epsilon,t,x).prev_sample;points[j+1]=x
    return x,points


def run(a):
    state,parent=complete_state(a.student);cache,cr=load_cache(a.student_cache)
    descriptor=state.get('depth_compression',{});sampling=state['sampling_contract'];config=descriptor['profile']
    assert descriptor['schema']=='scene_compact_recovered_v1'and sampling['steps']in[6,8]and sampling['CFG']==7.5
    assert parent['compression_training_performed']and state['pair_sha256']==cache['pair_sha256']
    assert state['optical_frontend']['checkpoint_sha256']==cache['optical_frontend']['checkpoint_sha256']
    source=json.loads((a.teacher_paths/'report.json').read_text());path=a.teacher_paths/'TRAIN_full_teacher_paths.pt'
    assert source['complete']and source['teacher_is_full_electronic_Qwen']and not source['DEV_or_TEST_targets_created']
    assert digest(path)==source['full_TRAIN_teacher_paths_sha256']
    teacher=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert teacher['all_TRAIN']and teacher['teacher_sha256']=='662ff90f9a46670b28bf86d61629c59e4fb619ed69a62e4ae52ed8527ea76f76'
    train={r['input_key']:r for r in cache['records']if r['split']=='train'};dev={r['input_key']for r in cache['records']if r['split']=='dev'}
    keys=set(teacher['paths']);assert len(keys)==192 and keys<=set(train)and not keys&dev
    edges=[teacher['timesteps'].index(t)for t in sampling['timesteps']]+[30]
    assert edges[0]==0 and len(edges)==sampling['steps']+1 and all(x<y for x,y in zip(edges,edges[1:]))
    assert all(x.dtype==torch.bfloat16 and x.shape==(31,4,64,64)for x in teacher['paths'].values())
    pipe,head=load_generator(a,state,True);graph=prune_unet(pipe.unet,config['UNet_deep_keep']);pipe.set_ip_adapter_scale(.25)
    reference=copy.deepcopy(pipe.unet).eval().requires_grad_(False)
    from peft.tuners.lora.layer import Linear
    for layer in reference.modules():
        if isinstance(layer,Linear):layer.merge(safe_merge=True,adapter_names=['scene_pair'])
    fusion=install(pipe.unet);parameters=[p for p in pipe.unet.parameters()if p.requires_grad]
    assert parameters and not any(p.requires_grad for p in head.parameters())
    head_before={n:v.detach().cpu().clone()for n,v in head.state_dict().items()};conditions={}
    with torch.no_grad():
        for key in keys:
            f=cache['features'][key];conditions[key]={n:v.detach()for n,v in head(f['hidden'][None].cuda().float(),f['visual'][None].cuda().float()).items()}
    groups={c:sorted(k for k in keys if train[k]['category']==c)for c in ['lamp','pillow']};assert all(groups.values())
    probes=[k for c in groups for k in groups[c][::max(1,len(groups[c])//8)][:8]]
    active={c:[k for k in groups[c]if k not in probes]for c in groups};assert all(active.values())and len(set(probes))==16
    key=groups['lamp'][0];initial=teacher['paths'][key][0:1].cuda();sigma=teacher['sigmas'][0];t=torch.tensor(sampling['timesteps'][0],device='cuda')
    with torch.no_grad():
        actual=predict(pipe.unet,initial,t,conditions[key],state,sigma,True);expected=predict(reference,initial,t,conditions[key],state,sigma,True)
    assert torch.equal(actual,expected),'Merged compact prediction differs'
    write(a.output/'fusion_GPU_identity.json',dict(complete=True,compressed_UNet_prediction_exact=True,graph=graph,fusion=fusion,scaled_saved_noise_used_directly=True))
    del reference,initial,actual,expected;torch.cuda.empty_cache()
    def sample(key,gradient=False):
        cached=teacher['paths'][key];x=cached[0:1].cuda();wanted=cached[-1:].cuda()
        actual,points=rollout_from_scaled(pipe.scheduler,pipe.unet,x,conditions[key],state,sampling['timesteps'],gradient)
        endpoints={index:cached[index:index+1].cuda()for index in edges[1:]}
        return objective(actual,wanted,points,endpoints,edges)
    def measure():
        with torch.no_grad():return sum(float(sample(k)[1])for k in probes)/len(probes)
    before=measure();initial=[p.detach().clone()for p in parameters]
    def optimizer():return torch.optim.AdamW(parameters,lr=a.lr,weight_decay=.01)
    opt=optimizer();smoke=[];torch.cuda.reset_peak_memory_stats()
    for index in range(4):
        key=active[['lamp','pillow'][index%2]][0];opt.zero_grad();loss,terminal,anchor=sample(key,True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);assert torch.isfinite(loss)and norm>0;opt.step()
        smoke.append(dict(loss=float(loss),terminal=float(terminal),anchor=float(anchor),gradient_norm=float(norm)))
    peak=torch.cuda.max_memory_allocated();assert peak<55*1024**3
    with torch.no_grad():
        for p,old in zip(parameters,initial):p.copy_(old)
    del initial,opt,loss,terminal,anchor;opt=optimizer()
    write(a.output/'smoke_report.json',dict(complete=True,real_full_rollout_backward_steps=4,student_and_optimizer_reset=True,
        student_steps=sampling['steps'],peak_allocated_bytes=peak,history=smoke,teacher_GPU_loaded=False,initial_fixed_TRAIN_terminal_loss=before))
    if a.stage=='smoke':
        write(a.output/'report.json',dict(complete=True,stage='smoke',training_performed=False,quality_or_speed_accepted=False));return
    rng=random.Random(20261031);best=math.inf;history=[];began=time.monotonic();recent=[]
    for step in range(1,a.updates+1):
        key=rng.choice(active[['lamp','pillow'][step%2]])
        for group in opt.param_groups:group['lr']=a.lr*min(1.,step/20)*(.1+.9*.5*(1+math.cos(math.pi*step/a.updates)))
        opt.zero_grad();loss,terminal,anchor=sample(key,True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss);opt.step();recent.append(float(terminal))
        if step%10==0:write(a.output/'progress.json',dict(stage='TRAIN_cached_electronic_terminal_full_student_rollout',step=step,total=a.updates,recent_terminal=sum(recent[-10:])/len(recent[-10:])))
        if step%25==0 or step==a.updates:
            score=measure();assert all(torch.equal(v,head.state_dict()[n].cpu())for n,v in head_before.items())
            history.append(dict(step=step,fixed_TRAIN_terminal_loss=score,seconds=time.monotonic()-began));write(a.output/'history.json',history)
            compact={n:p.detach().cpu()for n,p in pipe.unet.named_parameters()if'lora_'in n}
            result=dict(state,source_commit=os.environ['T12_SOURCE_COMMIT'],step=step,
                unet_lora=expand_lora(state['unet_lora'],compact,config['UNet_deep_keep']),
                cached_rollout_teacher_sha256=teacher['teacher_sha256'],cached_rollout_warm_parent_sha256=parent['best_sha256'])
            torch.save(result,a.output/'last_checkpoint.pt')
            if score<best:best=score;torch.save(result,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],stage='train',updates=a.updates,
        best_sha256=digest(a.output/'best_checkpoint.pt'),last_sha256=digest(a.output/'last_checkpoint.pt'),
        depth_compression=descriptor,compression_training_performed=True,sampling_steps=sampling['steps'],
        parent_checkpoint_sha256=parent['best_sha256'],teacher_sha256=teacher['teacher_sha256'],teacher_paths_sha256=digest(path),
        student_cache_sha256=cr['cache_sha256'],TRAIN_contexts=192,gradient_TRAIN_contexts=sum(len(g)for g in active.values()),
        fixed_probe_TRAIN_contexts=len(probes),initial_fixed_TRAIN_terminal_loss=before,best_TRAIN_terminal_loss=best,
        best_selection='Reserved fixed TRAIN contexts, not DEV/TEST or image quality',gradient_through_all_student_steps=True,
        exact_saved_scaled_noise=True,one_noise_per_reference_context_from_saved_teacher=True,teacher_GPU_forward_repeated=False,
        head_Qwen_optics_VAE_frozen=True,extra_deployment_parameters=0,scene_GT_or_reference_VAE_used=False,
        image_feedback_to_Qwen=False,independent_TEST_opened=False,quality_or_speed_accepted=False,seconds=time.monotonic()-began))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['student','student-cache','teacher-paths','model-path','adapter-path','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','train'],required=True);p.add_argument('--updates',type=int,default=300);p.add_argument('--lr',type=float,default=5e-6);a=p.parse_args()
    assert a.updates>=25 and a.lr>0 and not a.output.exists()and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);torch.manual_seed(1042)
    a.no_gradient_checkpointing=True;a.adapt_head=False
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__))))
    try:run(a)
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
