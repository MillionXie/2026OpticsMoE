"""Audited cache / TRAIN fit / raw TEST rendering of the RN18 substitution."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import torch
from PIL import Image
from .scene_instruction_data import sha, write
from .scene_resnet_frontend import (SCHEMA, PARENT, PAIRS, REQUESTS, ResNetSceneFrontend,
    ResNetConditionReadout, rgb_tensor, tokenize_text, condition_loss, validate_training_rows)


def parent(a):
    from .scene_tail_distill import complete_state
    state, report = complete_state(a.parent)
    if report['best_sha256'] != PARENT or state['pair_sha256'] != PAIRS:
        raise ValueError('Exact selected full70 parent required')
    p = state['depth_compression']['profile']
    if p['schema'] != 'scene_expanded_full_UNet8_v1' or p['UNet_deep_keep'] != 10:
        raise ValueError('Keep all70 original UNet blocks')
    if state['sampling_contract']['timesteps'] != [958,859,727,595,463,364,232,100]:
        raise ValueError('Do not change the trained eight-step grid')
    return state


def checked_data(a):
    from .scene_expanded_paired_data import checked_dataset
    rows, test, report = checked_dataset(a.data)
    if report['pairs_sha256'] != PAIRS or report['request_sha256'] != REQUESTS:
        raise ValueError('Existing reviewed data required')
    return rows, test


def cache(a):
    from .scene_tail_distill import load_cache
    from .qwen3_scene_conditioning import QwenSDXLConditions
    from ..t19_resnet20_baselines.frozen_resnet18 import assert_encoder_unchanged
    state = parent(a)
    rows, _ = checked_data(a)
    payload, cr = load_cache(a.teacher_cache)
    validate_training_rows(payload['records'])
    if (payload['pair_sha256'] != PAIRS or not payload.get('optical_frontend')
        or cr['cache_sha256'] != 'bf70b5ca54c5d3d0455eae812b1ed8fc03452fe2666db48c90176a58b9acfcb8'):
        raise ValueError('Exact published bound LG TRAIN cache required')
    actual = {r['sample_id']: r for r in rows}
    for row in payload['records']:
        old = actual[row['sample_id']]
        if any(row[k] != old[k] for k in ['prompt','reference_sha256','target_sha256']):
            raise ValueError('TRAIN input/target identity drift')
    model = ResNetSceneFrontend().cuda().eval()
    initial = model.encoder.pretrained_provenance['encoder_state_sha256']
    teacher = QwenSDXLConditions(joint_image_context=False).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(state['head'], strict=True)
    visuals, targets = {}, {}
    for i, row in enumerate(payload['records']):
        if row['reference_sha256'] not in visuals:
            path = a.data/row['reference']
            if sha(path) != row['reference_sha256']:
                raise ValueError('Changed TRAIN RGB')
            with Image.open(path) as image:
                rgb = rgb_tensor(image.convert('RGB'))[None].cuda()
            with torch.no_grad():
                visuals[row['reference_sha256']] = model.visual_tokens(rgb)[0].cpu().clone()
        f = payload['features'][row['input_key']]
        with torch.no_grad():
            out = teacher(f['hidden'][None].cuda().float(), f['visual'][None].cuda().float())
            targets[row['input_key']] = {k: v[0].cpu().bfloat16().clone() for k,v in out.items()}
        write(a.output/'progress.json', dict(stage='TRAIN_only_RN18_and_frozen_condition_targets',done=i+1,total=624))
    frozen = assert_encoder_unchanged(model.encoder, initial)
    data = dict(schema=SCHEMA, records=payload['records'], visuals=visuals, targets=targets,
                pair_sha256=PAIRS, teacher_sha256=PARENT,
                teacher_cache_sha256=cr['cache_sha256'], official_visual=model.encoder.pretrained_provenance)
    torch.save(data, a.output/'frontend_cache.pt')
    write(a.output/'report.json',dict(complete=True,schema=SCHEMA,n=624,TRAIN_views=76,TEST_target_conditions=0,
        cache_sha256=sha(a.output/'frontend_cache.pt'),Qwen_model_loaded=False,
        teacher_is_cached_LG_features_plus_frozen_condition_head=True,TRAIN_distillation_only=True,
        official_visual=data['official_visual'],frozen_encoder_audit=frozen,source_commit=os.environ['T12_SOURCE_COMMIT']))


def fit(a):
    state = parent(a)
    report = json.loads((a.cache/'report.json').read_text())
    if not report['complete'] or sha(a.cache/'frontend_cache.pt') != report['cache_sha256']:
        raise ValueError('Complete exact frontend cache required')
    data = torch.load(a.cache/'frontend_cache.pt',map_location='cpu',weights_only=False,mmap=True)
    rows=data['records'];validate_training_rows(rows)
    def fresh():
        torch.manual_seed(42)
        model=ResNetConditionReadout().cuda()
        model.initialize_readouts(state['head'])
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        return model,opt
    def batch(indices):
        selected=[rows[i] for i in indices]
        visual=torch.stack([data['visuals'][r['reference_sha256']] for r in selected]).cuda().float()
        text=tokenize_text([r['prompt'] for r in selected]).cuda()
        target={k:torch.stack([data['targets'][r['input_key']][k] for r in selected]).cuda().float()
                for k in ['text','pooled','image']}
        return visual,text,target
    model,opt=fresh();initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};smoke=[]
    for step in range(4):
        v,t,y=batch(list(range(step*2,step*2+2)));loss=condition_loss(model(v,t),y)
        opt.zero_grad();loss.backward()
        grads={name:sum(float(p.grad.abs().sum())for n,p in model.named_parameters()
                        if n.startswith(name)and p.grad is not None)
               for name in ['text_encoder','project','text_projection','text.','image.','pooled.']}
        if not all(value>0 for value in grads.values()) or not all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None):
            raise ValueError('Finite nonzero gradients required in every new frontend component')
        opt.step();smoke.append(dict(step=step+1,loss=float(loss),gradients=grads))
    del model,opt;gc.collect();torch.cuda.empty_cache();model,opt=fresh()
    if not all(torch.equal(v.cpu(),initial[k])for k,v in model.state_dict().items())or opt.state:
        raise ValueError('Real backward smoke must reset model and optimizer')
    write(a.output/'smoke_report.json',dict(complete=True,steps=smoke,weights_reset_exact=True,optimizer_reset=True))
    rng=random.Random(42);history=[];best=float('inf');updates=0;began=time.monotonic()
    for epoch in range(1,a.epochs+1):
        order=list(range(624));rng.shuffle(order);values=[]
        for start in range(0,624,a.batch_size):
            v,t,y=batch(order[start:start+a.batch_size]);loss=condition_loss(model(v,t),y)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            values.append(float(loss.detach()));updates+=1
        mean=sum(values)/len(values)
        item=dict(epoch=epoch,updates=updates,TRAIN_condition_loss=mean,seconds=time.monotonic()-began)
        history.append(item);write(a.output/'history.json',history);write(a.output/'progress.json',dict(item,total_epochs=a.epochs,stage='TRAIN_only_new_frontend_fit_fixed_generator'))
        checkpoint=dict(schema=SCHEMA,readout=model.state_dict(),epoch=epoch,step=updates,TRAIN_condition_loss=mean,
            generator_sha256=PARENT,pair_sha256=PAIRS,cache_sha256=report['cache_sha256'],
            official_visual=data['official_visual'],source_commit=os.environ['T12_SOURCE_COMMIT'])
        torch.save(checkpoint,a.output/'last_checkpoint.pt')
        if mean<best:best=mean;torch.save(checkpoint,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',dict(complete=True,schema=SCHEMA,epochs=a.epochs,updates=updates,TRAIN_contexts=624,
        TRAIN_views=76,batch_size=a.batch_size,seconds=time.monotonic()-began,best_TRAIN_loss=best,
        best_sha256=sha(a.output/'best_checkpoint.pt'),last_sha256=sha(a.output/'last_checkpoint.pt'),
        generator_sha256=PARENT,generator_weights_modified=False,TEST_used_for_gradient=False,
        Qwen_model_loaded=False,distilled_from_preexisting_TRAIN_LG_conditions=True,
        readout_initialization='Inherited original text/image/pooled query readouts; new RN18/GRU projections',
        trainable_frontend_parameters=sum(p.numel()for p in model.parameters()),
        frozen_visual_parameters=11176512,source_commit=os.environ['T12_SOURCE_COMMIT']))


def render(a):
    from .scene_tail_distill import load_generator
    from .scene_depth_probe import prune_unet
    from ..t19_resnet20_baselines.frozen_resnet18 import assert_encoder_unchanged
    state=parent(a);_,rows=checked_data(a)
    report=json.loads((a.frontend/'report.json').read_text())
    file=a.frontend/(a.candidate+'_checkpoint.pt')
    expected=report[a.candidate+'_sha256']
    if not report['complete']or sha(file)!=expected:raise ValueError('Complete trained frontend checkpoint required')
    student=torch.load(file,map_location='cpu',weights_only=False)
    if student['schema']!=SCHEMA or student['generator_sha256']!=PARENT:raise ValueError('Wrong frontend/tail binding')
    model=ResNetSceneFrontend().cuda().eval();model.readout.load_state_dict(student['readout'],strict=True)
    official=student['official_visual']
    if model.encoder.pretrained_provenance['encoder_state_sha256']!=official['encoder_state_sha256']:
        raise ValueError('Changed official visual weights')
    a.fuse_teacher_task_lora=True
    pipe,unused_head=load_generator(a,state,False);del unused_head
    graph=prune_unet(pipe.unet,10);assert graph['actual_blocks']==70 and graph['removed_parameters']==0
    pipe.vae.cuda().eval().requires_grad_(False);pipe.set_ip_adapter_scale(.25)
    records=[]
    for i,row in enumerate(rows):
        src=a.data/row['reference']
        if sha(src)!=row['reference_sha256']:raise ValueError('Changed TEST RGB')
        with Image.open(src)as image:rgb=rgb_tensor(image.convert('RGB'))[None].cuda()
        tokens=tokenize_text([row['prompt']]).cuda()
        generator=torch.Generator(device='cuda').manual_seed(row['seed'])
        noise=torch.randn((1,4,64,64),generator=generator,device='cuda',dtype=torch.bfloat16)
        with torch.inference_mode():
            cond={k:v.bfloat16()for k,v in model(rgb,tokens).items()}
            images=torch.cat([state['negative_image'].cuda().bfloat16(),cond['image']],0).unsqueeze(1)
            image=pipe(width=512,height=512,timesteps=state['sampling_contract']['timesteps'],guidance_scale=7.5,
                latents=noise,generator=generator,original_size=(1024,1024),target_size=(1024,1024),
                prompt_embeds=cond['text'],pooled_prompt_embeds=cond['pooled'],
                negative_prompt_embeds=state['negative_text'].cuda().bfloat16(),
                negative_pooled_prompt_embeds=state['negative_pooled'].cuda().bfloat16(),ip_adapter_image_embeds=[images]).images[0]
        name=row['sample_id']+'.png';image.save(a.output/name)
        reference=a.output/(row['reference_id']+'.png')
        if not reference.exists():shutil.copyfile(src,reference)
        records.append(dict(row,split='test',generated_PNG=name,generated_sha256=sha(a.output/name),
            input_sha256=sha(src),initial_latent_sha256=hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
            checkpoint_sha256=expected,fixed_generator_sha256=PARENT,model='resnet18_byte_gru',resolution=512,sampling_steps=8))
        write(a.output/'images.json',records);write(a.output/'progress.json',dict(stage='raw_ResNet_GRU_TEST500_fixed70',done=i+1,total=500))
    audit=assert_encoder_unchanged(model.encoder,official['encoder_state_sha256'])
    write(a.output/'report.json',dict(complete=True,schema=SCHEMA,scope='test',generated_count=500,resolution=512,
        checkpoint_sha256=expected,fixed_generator_sha256=PARENT,sampling_steps=8,actual_UNet_blocks=70,
        Qwen_parameters=0,optical_frontend_loaded=False,raw_RGB_prompt_recomputed=True,cached_conditions_in_inference=False,
        fixed_generator_weights_modified=False,TEST_used_for_gradient=False,TEST_used_for_model_selection=True,
        independent_generalization_claim=False,graph=graph,frozen_encoder_audit=audit,
        frontend_parameters=sum(p.numel()for p in model.parameters()),source_commit=os.environ['T12_SOURCE_COMMIT']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['cache','fit','render'],required=True)
    for key in ['parent','data','output']:p.add_argument('--'+key,type=Path,required=True)
    for key in ['teacher-cache','cache','frontend','model-path','adapter-path']:p.add_argument('--'+key,type=Path)
    p.add_argument('--candidate',choices=['best','last'],default='last')
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    from .scene_execution_gpu import configured_gpu
    gpu=configured_gpu()
    if a.output.exists()or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('Fresh run, explicit GPU UUID and published source required')
    if a.epochs!=50 or a.batch_size!=8:raise ValueError('Fixed50 epochs/B8 frontend-only recipe')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],
        source_sha256=sha(__file__),GPU_UUID=gpu,GPU_name=torch.cuda.get_device_name(0),Qwen_backbone_loaded=False))
    try:globals()[a.stage](a)
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
