"""Optional whole-rollout recovery of existing readout and compact UNet LoRA.

The F6 Qwen/optical frontend, original VAE, six-step sampler and deployment
parameter count remain fixed. Only the already existing F9 readout is additionally
adapted. Stored full E30 TRAIN trajectories supply supervision; no new teacher
generation, reference VAE branch, image feedback or held-out target is used.
"""
import argparse,copy,json,math,os,random,sys,time,traceback
from pathlib import Path
import torch
from .scene_tail_distill import complete_state,load_cache,load_generator,predict
from .scene_cached_rollout_recovery import rollout_from_scaled
from .scene_depth_probe import prune_unet
from .scene_depth_recovery import expand_lora
from .scene_fusion_aware_tail import install
from .scene_rollout_tail import objective
from .white_reference_scene_probe import GPU,digest,write


def condition_anchor(actual,reference):
    assert set(actual)==set(reference)=={'text','pooled','image'}
    return torch.stack([(actual[k].float()-reference[k].float()).square().mean()/reference[k].float().square().mean().clamp_min(1e-6)for k in actual]).mean()


def joint_rollout(scheduler,unet,head,feature,initial,state,grid,gradient=False):
    conditions=head(feature['hidden'],feature['visual'])
    end,points=rollout_from_scaled(scheduler,unet,initial,conditions,state,grid,gradient)
    return end,points,conditions


def run(a):
    state,parent=complete_state(a.student);cache,cr=load_cache(a.student_cache)
    descriptor=state['depth_compression'];sampling=state['sampling_contract'];config=descriptor['profile']
    assert descriptor['schema']=='scene_compact_recovered_v1'and sampling['steps']==6 and sampling['CFG']==7.5
    assert parent['compression_training_performed']and state['pair_sha256']==cache['pair_sha256']
    assert state['optical_frontend']['checkpoint_sha256']==cache['optical_frontend']['checkpoint_sha256']
    source=json.loads((a.teacher_paths/'report.json').read_text());path=a.teacher_paths/'TRAIN_full_teacher_paths.pt'
    assert source['complete']and source['teacher_is_full_electronic_Qwen']and not source['DEV_or_TEST_targets_created']
    assert digest(path)==source['full_TRAIN_teacher_paths_sha256']
    teacher=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert teacher['all_TRAIN']and teacher['teacher_sha256']=='662ff90f9a46670b28bf86d61629c59e4fb619ed69a62e4ae52ed8527ea76f76'
    train={r['input_key']:r for r in cache['records']if r['split']=='train'};dev={r['input_key']for r in cache['records']if r['split']=='dev'}
    keys=set(teacher['paths']);assert len(keys)==192 and keys<=set(train)and not keys&dev
    assert all(v.dtype==torch.bfloat16 and v.shape==(31,4,64,64)for v in teacher['paths'].values())
    edges=[teacher['timesteps'].index(t)for t in sampling['timesteps']]+[30]
    assert edges[0]==0 and len(edges)==7 and all(x<y for x,y in zip(edges,edges[1:]))
    pipe,head=load_generator(a,state,True);graph=prune_unet(pipe.unet,config['UNet_deep_keep']);pipe.set_ip_adapter_scale(.25)
    reference=copy.deepcopy(pipe.unet).eval().requires_grad_(False)
    from peft.tuners.lora.layer import Linear
    for layer in reference.modules():
        if isinstance(layer,Linear):layer.merge(safe_merge=True,adapter_names=['scene_pair'])
    fusion=install(pipe.unet);lora=[p for p in pipe.unet.parameters()if p.requires_grad];head_parameters=list(head.parameters())
    assert lora and all(p.requires_grad for p in head_parameters)
    features={key:{n:cache['features'][key][n][None].cuda().float()for n in ['hidden','visual']}for key in keys}
    with torch.no_grad():warm={key:{n:v.detach()for n,v in head(**features[key]).items()}for key in keys}
    groups={c:sorted(k for k in keys if train[k]['category']==c)for c in ['lamp','pillow']}
    probes=[k for c in groups for k in groups[c][::max(1,len(groups[c])//8)][:8]]
    active={c:[k for k in groups[c]if k not in probes]for c in groups};assert len(set(probes))==16 and all(active.values())
    key=groups['lamp'][0];initial=teacher['paths'][key][0:1].cuda();t=torch.tensor(sampling['timesteps'][0],device='cuda')
    with torch.no_grad():
        actual=predict(pipe.unet,initial,t,warm[key],state,teacher['sigmas'][0],True)
        expected=predict(reference,initial,t,warm[key],state,teacher['sigmas'][0],True)
    assert torch.equal(actual,expected),'Official safe-merged prediction must match before training'
    write(a.output/'fusion_GPU_identity.json',dict(complete=True,compressed_UNet_prediction_exact=True,graph=graph,fusion=fusion))
    del reference,initial,actual,expected;torch.cuda.empty_cache()
    def sample(key,gradient=False):
        saved=teacher['paths'][key];initial=saved[0:1].cuda();wanted=saved[-1:].cuda()
        actual,points,cond=joint_rollout(pipe.scheduler,pipe.unet,head,features[key],initial,state,sampling['timesteps'],gradient)
        endpoints={index:saved[index:index+1].cuda()for index in edges[1:]};loss,terminal,anchor=objective(actual,wanted,points,endpoints,edges)
        regularizer=condition_anchor(cond,warm[key])
        return loss+.02*regularizer,terminal,regularizer
    def measure():
        with torch.no_grad():return sum(float(sample(key)[1])for key in probes)/len(probes)
    parameters=lora+head_parameters;initial_parameters=[p.detach().clone()for p in parameters]
    def optimizer():return torch.optim.AdamW([dict(params=lora,lr=a.lr),dict(params=head_parameters,lr=a.head_lr)],weight_decay=.01)
    opt=optimizer();before=measure();smoke=[];torch.cuda.reset_peak_memory_stats()
    for index in range(4):
        key=active[['lamp','pillow'][index%2]][0];opt.zero_grad();loss,terminal,anchor=sample(key,True);loss.backward()
        lora_norm=torch.nn.utils.clip_grad_norm_(lora,1.,error_if_nonfinite=True);head_norm=torch.nn.utils.clip_grad_norm_(head_parameters,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss)and lora_norm>0 and head_norm>0;opt.step()
        smoke.append(dict(loss=float(loss),terminal=float(terminal),UNet_LoRA_gradient_norm=float(lora_norm),readout_gradient_norm=float(head_norm)))
    peak=torch.cuda.max_memory_allocated();assert peak<55*1024**3
    with torch.no_grad():
        for p,old in zip(parameters,initial_parameters):p.copy_(old)
    del initial_parameters,opt,loss,terminal,anchor;opt=optimizer()
    write(a.output/'smoke_report.json',dict(complete=True,actual_full_rollout_backward_steps=4,readout_and_LoRA_actual_gradients=smoke,
        student_and_optimizer_reset=True,peak_allocated_bytes=peak,initial_fixed_TRAIN_terminal_loss=before))
    if a.stage=='smoke':write(a.output/'report.json',dict(complete=True,stage='smoke',training_performed=False));return
    best=math.inf;rng=random.Random(20261031);history=[];began=time.monotonic();recent=[]
    for step in range(1,a.updates+1):
        for group,lr in zip(opt.param_groups,[a.lr,a.head_lr]):group['lr']=lr*min(1.,step/20)*(.1+.9*.5*(1+math.cos(math.pi*step/a.updates)))
        key=rng.choice(active[['lamp','pillow'][step%2]]);opt.zero_grad();loss,terminal,anchor=sample(key,True);loss.backward()
        torch.nn.utils.clip_grad_norm_(lora,1.,error_if_nonfinite=True);torch.nn.utils.clip_grad_norm_(head_parameters,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss);opt.step();recent.append(float(terminal))
        if step%10==0:write(a.output/'progress.json',dict(stage='TRAIN_joint_existing_readout_and_six_step_rollout',step=step,total=a.updates,recent_terminal=sum(recent[-10:])/len(recent[-10:])))
        if step%25==0 or step==a.updates:
            score=measure();history.append(dict(step=step,fixed_TRAIN_terminal_loss=score,seconds=time.monotonic()-began));write(a.output/'history.json',history)
            compact={n:p.detach().cpu()for n,p in pipe.unet.named_parameters()if'lora_'in n}
            new=dict(state,source_commit=os.environ['T12_SOURCE_COMMIT'],step=step,
                unet_lora=expand_lora(state['unet_lora'],compact,config['UNet_deep_keep']),head={n:v.detach().cpu()for n,v in head.state_dict().items()},
                existing_readout_joint_rollout_recovery=True,joint_rollout_warm_parent_sha256=parent['best_sha256'])
            torch.save(new,a.output/'last_checkpoint.pt')
            if score<best:best=score;torch.save(new,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],stage='train',updates=a.updates,
        best_sha256=digest(a.output/'best_checkpoint.pt'),last_sha256=digest(a.output/'last_checkpoint.pt'),
        compression_training_performed=True,depth_compression=descriptor,sampling_steps=6,existing_readout_adapted=True,
        parent_checkpoint_sha256=parent['best_sha256'],teacher_sha256=teacher['teacher_sha256'],teacher_paths_sha256=digest(path),
        student_cache_sha256=cr['cache_sha256'],TRAIN_contexts=192,gradient_TRAIN_contexts=sum(len(v)for v in active.values()),fixed_probe_TRAIN_contexts=16,
        initial_fixed_TRAIN_terminal_loss=before,best_TRAIN_terminal_loss=best,best_selection='Fixed reused TRAIN probe, not DEV/TEST or image quality',
        existing_readout_trainable_parameters=sum(p.numel()for p in head_parameters),existing_UNet_LoRA_trainable_parameters=sum(p.numel()for p in lora),
        Qwen_optics_original_VAE_frozen=True,extra_deployment_parameters=0,teacher_GPU_forward_repeated=False,
        reference_VAE_or_image_feedback_used=False,independent_TEST_opened=False,quality_or_speed_accepted=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['student','student-cache','teacher-paths','model-path','adapter-path','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','train'],required=True);p.add_argument('--updates',type=int,default=300)
    p.add_argument('--lr',type=float,default=5e-6);p.add_argument('--head-lr',type=float,default=2.5e-6);a=p.parse_args()
    assert not a.output.exists()and a.updates>=25 and min(a.lr,a.head_lr)>0 and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    assert len(os.environ.get('T12_SOURCE_COMMIT',''))==40;a.output.mkdir(parents=True);a.adapt_head=True;a.no_gradient_checkpointing=True
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);torch.manual_seed(1042)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__))))
    try:run(a)
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
