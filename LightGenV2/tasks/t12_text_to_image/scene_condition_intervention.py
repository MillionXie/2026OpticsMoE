"""Frozen TRAIN controls for condition, sampling and existing LoRA effects.

Six interventions on sixteen prespecified TRAIN contexts are diagnostics, not
new trained models, TEST candidates, deployment outputs or latency estimates.
All cases use the same saved scaled noise, original70-block UNet and VAE.
"""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

E_SHA='68e04d5f378e2bc0a495363b45d7d8c68e81c41cf4169329749712d23d8998f6'
LG_SHA='aa2d277bd743d4dd831ec33852629dbf3001e772b55a8ac436a0d2a557a06fd8'
PAIRS='c2e30cbdbd871094df8827c78f4a02bc23305a0e39898ab53c6acf4178835fd0'
CASES=(('E_E_30','E','E',30),('E_LG_30','E','LG',30),
       ('E_E_8','E','E',8),('E_LG_8','E','LG',8),
       ('LG_E_8','LG','E',8),('LG_LG_8','LG','LG',8))


def selected_train_rows(records,path_keys):
    train={}
    for row in records:
        if row['split']!='train':continue
        key=row['input_key']
        if key in train:raise ValueError('Duplicate TRAIN context')
        train[key]=row
    if set(train)!=set(path_keys):raise ValueError('Every saved path must be actual TRAIN')
    groups=defaultdict(list)
    for row in train.values():groups[row['category'],row['reference_id']].append(row)
    chosen=[]
    for category in ['lamp','pillow']:
        references=sorted(r for c,r in groups if c==category and len({x['condition_id']for x in groups[c,r]})>=2)
        if len(references)<4:raise ValueError('Four TRAIN reference views per category required')
        for rid in references[:4]:
            rows=sorted(groups[category,rid],key=lambda r:(int(r['condition_id']),r['input_key']))
            seen=set()
            for row in rows:
                if row['condition_id']in seen:continue
                chosen.append(row);seen.add(row['condition_id'])
                if len(seen)==2:break
    if len(chosen)!=16 or len({r['input_key']for r in chosen})!=16:
        raise ValueError('Sixteen distinct fixed TRAIN contexts required')
    return chosen


def relative_error(actual,wanted):
    import torch
    if actual.shape!=wanted.shape or not torch.isfinite(actual).all()or not torch.isfinite(wanted).all():
        raise ValueError('Same finite latent shapes required')
    return float((actual.float()-wanted.float()).square().mean()/wanted.float().square().mean().clamp_min(1.))


def run(a):
    import torch
    from .scene_tail_distill import complete_state,load_cache,load_generator
    from .scene_cached_rollout_recovery import rollout_from_scaled
    from .scene_depth_probe import prune_unet
    from .qwen3_scene_conditioning import QwenSDXLConditions
    from .scene_generation_data import request_seed
    from .white_reference_scene_probe import digest,write
    electronic,er=complete_state(a.electronic);student,lr=complete_state(a.student)
    if er['best_sha256']!=E_SHA or lr['best_sha256']!=LG_SHA:raise ValueError('Exact completed E68/full70 selected pair required')
    if electronic.get('optical_frontend')or electronic.get('sampling_contract'):raise ValueError('Original electronic30 control required')
    profile=student['depth_compression']['profile'];sampling=student['sampling_contract']
    if profile['schema']!='scene_expanded_full_UNet8_v1'or profile['UNet_deep_keep']!=10 or sampling['steps']!=8 or sampling['CFG']!=7.5:
        raise ValueError('Same complete70/eight-step selected model required')
    if student.get('joint_image_context',False):raise ValueError('Preserve reference-only image branch')
    for name in ['negative_text','negative_pooled','negative_image']:
        if not torch.equal(electronic[name],student[name]):raise ValueError('Negative conditions would confound the intervention')
    if electronic.get('SDXL_conditioning_size')!=1024 or student.get('SDXL_conditioning_size')!=1024:
        raise ValueError('Preserve1024 size conditioning')
    cache,cr=load_cache(a.cache)
    if not(electronic['pair_sha256']==student['pair_sha256']==cache['pair_sha256']==PAIRS):raise ValueError('Changed actual TRAIN dataset')
    if cache.get('condition_teacher_sha256')!=E_SHA or cr.get('condition_teacher_sha256')!=E_SHA:
        raise ValueError('Actual E68 TRAIN conditions required')
    tr=json.loads((a.teacher_paths/'report.json').read_text());path=a.teacher_paths/'TRAIN_full_teacher_paths.pt'
    if not tr['complete']or digest(path)!=tr['full_TRAIN_teacher_paths_sha256']:raise ValueError('Changed TRAIN paths')
    paths=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if not paths['all_TRAIN']or paths['teacher_sha256']!=E_SHA or paths['teacher_steps']!=30 or paths['pair_sha256']!=PAIRS:
        raise ValueError('Bound full E30 TRAIN paths only')
    rows=selected_train_rows(cache['records'],paths['paths'])
    if set(cache['TRAIN_electronic_conditions'])!={r['input_key']for r in cache['records']if r['split']=='train'}:
        raise ValueError('No held-out condition targets allowed')
    for row in rows:
        if digest(a.data/row['reference'])!=row['reference_sha256']:raise ValueError('Changed TRAIN RGB')
    by_reference=defaultdict(list)
    for row in rows:by_reference[row['reference_id']].append(row)
    for pair in by_reference.values():
        if not torch.equal(paths['paths'][pair[0]['input_key']][0],paths['paths'][pair[1]['input_key']][0]):
            raise ValueError('Same reference must keep identical actual starting noise')
    write(a.output/'cohort.json',dict(selection='First four sorted eligible TRAIN reference views/category, first two distinct numeric condition IDs; never selected by score',
        requests=[{k:r[k]for k in ['sample_id','input_key','reference_id','reference_sha256','category','condition_id','prompt']}for r in rows],
        TRAIN_contexts=16,TRAIN_reference_views=8,not_eight_independent_products=True,cases=CASES))
    head=QwenSDXLConditions(joint_image_context=False).cuda().eval().requires_grad_(False);head.load_state_dict(student['head'],strict=True)
    conditions={}
    with torch.inference_mode():
        for row in rows:
            key=row['input_key'];f=cache['features'][key]
            out=head(f['hidden'][None].cuda().float(),f['visual'][None].cuda().float())
            conditions[key]={'LG':{n:v.cpu().bfloat16()for n,v in out.items()},
                'E':{n:v[None].bfloat16()for n,v in cache['TRAIN_electronic_conditions'][key].items()}}
    del head;torch.cuda.empty_cache();reference={};latent_scores=[]
    a.fuse_teacher_task_lora=True
    for model_label,state in [('E',electronic),('LG',student)]:
        pipe,unused_head=load_generator(a,state,False);del unused_head
        graph=prune_unet(pipe.unet,10);assert graph['actual_blocks']==70 and graph['removed_parameters']==0
        pipe.set_ip_adapter_scale(.25);pipe.vae.cuda().eval().requires_grad_(False)
        if getattr(pipe.vae.config,'latents_mean',None)is not None:raise ValueError('Explicit original VAE latent convention required')
        for case,model,condition,steps in CASES:
            if model!=model_label:continue
            folder=a.output/case;folder.mkdir();images=[]
            grid=paths['timesteps']if steps==30 else sampling['timesteps']
            for row in rows:
                key=row['input_key'];initial=paths['paths'][key][0:1].cuda()
                if initial.dtype!=torch.bfloat16 or initial.shape!=(1,4,64,64):raise ValueError('Exact saved scaled512 noise required')
                starting=initial.clone();cond={n:v.cuda().float()for n,v in conditions[key][condition].items()}
                with torch.inference_mode():
                    result,_=rollout_from_scaled(pipe.scheduler,pipe.unet,initial,cond,state,grid,False)
                    assert torch.equal(initial,starting),'Sampler changed the fixed starting noise'
                    if case=='E_E_30':reference[key]=result.cpu().clone()
                    score=relative_error(result.cpu(),reference[key])
                    decoded=pipe.vae.decode((result/pipe.vae.config.scaling_factor).to(pipe.vae.dtype),return_dict=False)[0]
                    image=pipe.image_processor.postprocess(decoded,output_type='pil')[0]
                name=row['sample_id']+'.png';image.save(folder/name)
                src=a.data/row['reference'];dst=folder/(row['reference_id']+'.png')
                if not dst.exists():shutil.copyfile(src,dst)
                fields={k:row[k]for k in ['sample_id','reference_id','reference_sha256','category','condition_id','prompt','native_caption','reward_caption','prompt_profile']}
                noise_sha=__import__('hashlib').sha256(starting.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
                images.append(dict(fields,split='train',seed=request_seed(row['reference_sha256'],paths['noise_index']),generated_PNG=name,
                    generated_sha256=digest(folder/name),input_sha256=digest(src),initial_latent_sha256=noise_sha,
                    initial_latent_hash_is_saved_scaled_noise=True,checkpoint_sha256=E_SHA if model=='E'else LG_SHA,
                    condition_source_checkpoint_sha256=E_SHA if condition=='E'else LG_SHA,diagnostic_intervention_only=True,
                    sampling_steps=steps,CFG=7.5,IP_adapter_scale=.25,resolution=512))
                latent_scores.append(dict(case=case,input_key=key,category=row['category'],relative_terminal_MSE_vs_same_GPU_E30=score))
                write(a.output/'progress.json',dict(stage='fixed_TRAIN_condition_sampling_interventions',case=case,done=len(images),total=16))
            write(folder/'images.json',images)
            write(folder/'report.json',dict(complete=True,scope='TRAIN_condition_intervention',source_commit=os.environ['T12_SOURCE_COMMIT'],
                checkpoint_sha256=E_SHA if model=='E'else LG_SHA,condition_source_checkpoint_sha256=E_SHA if condition=='E'else LG_SHA,
                generated_count=16,training_performed=False,independent_TEST_opened=False,TEST_opened=False,
                prompt_profile='two_sentence_product_scene_expanded_v8',sampling_steps=steps,resolution=512,
                actual_UNet_blocks=70,cached_conditions_control_only=True,raw_Qwen_recomputed=False,not_deployable_candidate=True,
                quality_or_speed_accepted=False,graph=graph))
        del pipe;gc.collect();torch.cuda.empty_cache()
    write(a.output/'latent_diagnostic.json',latent_scores)
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(__file__),
        teacher_sha256=E_SHA,student_sha256=LG_SHA,pairs_sha256=PAIRS,cache_sha256=cr['cache_sha256'],
        TRAIN_paths_sha256=tr['full_TRAIN_teacher_paths_sha256'],TRAIN_contexts=16,TRAIN_reference_views=8,
        generated_count=96,cases=CASES,training_performed=False,TEST_opened=False,checkpoint_files_written=False,
        raw_Qwen_forward=False,cached_conditions_control_only=True,all_cases_same_saved_scaled_noise=True,
        all_cases_original70_UNet_and_original_VAE=True,same_negative_conditions=True,
        no_quality_or_speed_gate_claim=True,not_deployable_candidate=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['electronic','student','cache','teacher-paths','data','model-path','adapter-path','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from .scene_execution_gpu import configured_gpu
    from .white_reference_scene_probe import digest,write
    gpu=configured_gpu()
    if a.output.exists()or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('New diagnostic run, published source and explicit matching GPU UUID required')
    import torch
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],
        source_sha256=digest(__file__),GPU_UUID=gpu,GPU_name=torch.cuda.get_device_name(0),training_performed=False))
    try:run(a)
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
