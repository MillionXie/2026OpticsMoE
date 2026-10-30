"""TRAIN-only reference-prefix adaptation of an existing LG image query head.

Freeze the projected visual features, language queries, pooled head and UNet.
The image head reads reference tokens only, as in the electronic teacher. This
is a feature-supervised candidate, not evidence of generation comparability.
"""
import argparse
import copy
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import torch
import torch.nn.functional as F
from .qwen3_scene_conditioning import QwenSDXLConditions
from .scene_tail_distill import complete_state, load_cache
from .white_reference_scene_probe import GPU, digest, write


def reference_rows(cache):
    train={r['input_key'] for r in cache['records'] if r['split']=='train'}
    dev={r['input_key'] for r in cache['records'] if r['split']=='dev'}
    if not train or train&dev or set(cache['TRAIN_electronic_conditions'])!=train:
        raise ValueError('Exact TRAIN-only teacher conditions required')
    groups={}
    for row in cache['records']:
        if row['split']!='train':continue
        key=row['input_key'];reference=row['reference_sha256']
        if reference in groups:
            prior=groups[reference]['input_key']
            if not torch.equal(cache['features'][prior]['visual'],cache['features'][key]['visual']):
                raise ValueError('Reference-only visual tokens depend on prompt')
            if not torch.equal(cache['TRAIN_electronic_conditions'][prior]['image'],cache['TRAIN_electronic_conditions'][key]['image']):
                raise ValueError('Teacher reference prefix depends on prompt')
        else:groups[reference]=row
    return sorted(groups.values(),key=lambda r:r['reference_sha256'])


def configure(head):
    head.eval().requires_grad_(False);head.joint_image_context=False
    head.image.train().requires_grad_(True)
    return list(head.image.parameters())


def prefix_loss(pred,target):
    if pred.shape!=target.shape:raise ValueError('Reference prefix shape mismatch')
    scale=target.float().square().mean(dim=(-2,-1),keepdim=True).clamp_min(1e-6)
    return ((pred.float()-target.float()).square()/scale).mean()+.1*(1-F.cosine_similarity(pred.float(),target.float(),dim=-1)).mean()


def run(a):
    state,ready=complete_state(a.generator);cache,metadata=load_cache(a.cache)
    assert state['pair_sha256']==cache['pair_sha256'] and state['joint_image_context']
    assert state['optical_frontend']['checkpoint_sha256']==cache['optical_frontend']['checkpoint_sha256']
    assert state['optical_frontend']['pair_expansion']==cache['optical_frontend']['pair_expansion']
    rows=reference_rows(cache);assert {r['category'] for r in rows}=={'lamp','pillow'}
    head=QwenSDXLConditions(joint_image_context=True).cuda();head.load_state_dict(state['head'],strict=True)
    parameters=configure(head);frozen={n:v.detach().cpu().clone() for n,v in head.state_dict().items() if not n.startswith('image.')}
    initial=copy.deepcopy(head.image.state_dict())
    with torch.no_grad():
        projected=torch.cat([head.project(cache['features'][r['input_key']]['visual'].unsqueeze(0).cuda().float()) for r in rows])
        target=torch.stack([cache['TRAIN_electronic_conditions'][r['input_key']]['image'] for r in rows]).cuda().float()
    assert not projected.requires_grad and torch.isfinite(projected).all() and torch.isfinite(target).all()
    def objective(indices):return prefix_loss(head.image(projected[indices])[0],target[indices])
    torch.manual_seed(20261030);opt=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=.01);smoke=[]
    for i in range(4):
        opt.zero_grad();loss=objective(list(range(i,min(i+a.batch_size,len(rows)))));loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss) and norm>0;opt.step();smoke.append(dict(loss=float(loss),gradient_norm=float(norm)))
    assert any(not torch.equal(v,initial[n]) for n,v in head.image.state_dict().items())
    head.image.load_state_dict(initial);del initial,opt
    assert all(torch.equal(v,head.state_dict()[n].cpu()) for n,v in frozen.items())
    write(a.output/'smoke_report.json',dict(complete=True,actual_backward_steps=4,history=smoke,parameters_and_optimizer_reset=True,
        projected_visual_frozen=True,language_project_pooled_frozen=True,UNet_Qwen_optics_VAE_not_loaded_or_updated=True))
    torch.manual_seed(20261030);rng=random.Random(20261030);opt=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=.01)
    best=math.inf;history=[];began=time.monotonic()
    def save(name,step,score):
        current=dict(state,head={n:v.detach().cpu() for n,v in head.state_dict().items()},joint_image_context=False,
            cache_sha256=metadata['cache_sha256'],source_commit=os.environ['T12_SOURCE_COMMIT'],step=step,
            reference_head_adaptation=dict(parent_generator_sha256=ready['best_sha256'],TRAIN_image_prefix_loss=score,
                teacher_sha256=cache['condition_teacher_sha256'],image_queries_only=True))
        torch.save(current,a.output/name)
    for step in range(1,a.steps+1):
        indices=rng.sample(range(len(rows)),min(a.batch_size,len(rows)))
        lr=a.lr*min(1.,step/50)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad();loss=objective(indices);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss);opt.step()
        if step%100==0 or step==a.steps:
            with torch.no_grad():score=sum(float(objective(list(range(i,min(i+8,len(rows))))))*(min(i+8,len(rows))-i) for i in range(0,len(rows),8))/len(rows)
            assert all(torch.equal(v,head.state_dict()[n].cpu()) for n,v in frozen.items())
            history.append(dict(step=step,TRAIN_reference_prefix_loss=score,seconds=time.monotonic()-began));write(a.output/'history.json',history)
            save('last_checkpoint.pt',step,score)
            if score<best:best=score;save('best_checkpoint.pt',step,score)
            write(a.output/'progress.json',dict(stage='TRAIN_reference_only_image_queries',step=step,total=a.steps,best_TRAIN_loss=best))
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],steps=a.steps,
        best_sha256=digest(a.output/'best_checkpoint.pt'),parent_generator_sha256=ready['best_sha256'],cache_sha256=metadata['cache_sha256'],
        condition_teacher_sha256=cache['condition_teacher_sha256'],TRAIN_unique_references=len(rows),TRAIN_input_keys=len(cache['TRAIN_electronic_conditions']),
        best_TRAIN_reference_prefix_loss=best,best_selection='TRAIN feature loss only; not generation or generalization quality',
        seconds=time.monotonic()-began,trainable_existing_image_head_parameters=sum(p.numel() for p in parameters),extra_deployment_parameters=0,
        joint_image_context=False,project_text_pooled_UNet_Qwen_optics_VAE_unchanged=True,sampling_contract_unchanged=True,
        only_TRAIN_visual_features_and_teacher_image_prefix_used=True,scene_GT_or_noise_latents_used=False,DEV_or_TEST_forwarded=False,
        quality_or_speed_accepted=False,TEST_opened=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['generator','cache','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4);a=p.parse_args()
    assert a.steps>=4 and a.batch_size>0 and a.lr>0 and not a.output.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__))))
    try:run(a)
    except Exception:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
