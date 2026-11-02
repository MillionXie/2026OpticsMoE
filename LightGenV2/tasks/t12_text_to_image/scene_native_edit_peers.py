"""Frozen native image+instruction baselines on the canonical target-free TEST.

Full RGB256 information, original complete prompt and seed, native512 output.
No task fine-tuning, output resizing, masks, target access or prompt rewriting.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from PIL import Image
from .scene_instruction_data import sha,write,SAFE

REQUESTS='f132238c3a59efb1308649909c96ce3669e02c7420d598bb29183536720f9f47'
MODELS={
 'ip2p':dict(repo='timbrooks/instruct-pix2pix',steps=100,guidance=7.5,image_guidance=1.5),
 'flux2_klein4b':dict(repo='black-forest-labs/FLUX.2-klein-4B',steps=4,guidance=1.),
 'magicbrush':dict(repo='osunlp/InstructPix2Pix-MagicBrush',steps=100,guidance=7.5,image_guidance=1.5),
}


def canonical_requests(data):
    request=data/'inference_inputs.json'
    if sha(request)!=REQUESTS:raise ValueError('Exact fixed500 TEST request identity required')
    rows=json.loads(request.read_text())
    if len(rows)!=500 or len({r['sample_id']for r in rows})!=500 or any(set(r)!=set(SAFE)for r in rows):
        raise ValueError('Target-free TEST requests only')
    return rows


def preview_first(rows):
    selected=[]
    for category in ['lamp','pillow']:
        refs=sorted({r['reference_id']for r in rows if r['category']==category})[:2]
        for ref in refs:selected.extend(sorted([r for r in rows if r['reference_id']==ref],key=lambda r:int(r['condition_id']))[:2])
    ids={r['sample_id']for r in selected}
    if len(ids)!=8:raise ValueError('Fixed balanced8 preview required')
    return selected+[r for r in rows if r['sample_id']not in ids]


def verify_assets(folder,report,model):
    if not report['complete']or report['repo']!=MODELS[model]['repo']or len(report['revision'])!=40 or not report['all_files_bound_to_official_revision']:
        raise ValueError('Official frozen model revision manifest required')
    for rel,item in report['files'].items():
        path=folder/rel
        if Path(rel).is_absolute()or '..'in Path(rel).parts or path.stat().st_size!=item['bytes']or sha(path)!=item['sha256']:
            raise ValueError('Model asset identity drift')


def infer(a):
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline,EulerAncestralDiscreteScheduler
    rows=preview_first(canonical_requests(a.data));cfg=MODELS[a.model]
    if a.preview_only:rows=rows[:8]
    assets=json.loads(a.asset_report.read_text());verify_assets(a.model_path,assets,a.model)
    identity=sha(a.asset_report)
    if a.model=='flux2_klein4b':
        from diffusers import Flux2KleinPipeline
        pipe=Flux2KleinPipeline.from_pretrained(a.model_path,torch_dtype=torch.bfloat16,local_files_only=True)
        if torch.cuda.get_device_properties(0).total_memory>=38*1024**3:
            pipe.to('cuda');residency='all_components_cuda_on_40GB'
        else:pipe.enable_model_cpu_offload();residency='model_cpu_offload_on_smaller_GPU'
    elif a.model=='magicbrush':
        if a.ip2p_config is None:raise ValueError('Compatible original IP2P architecture config required')
        pipe=StableDiffusionInstructPix2PixPipeline.from_single_file(a.model_path/'MagicBrush-epoch-52-step-4999.ckpt',
             config=str(a.ip2p_config),torch_dtype=torch.float16,local_files_only=True,safety_checker=None)
        pipe.to('cuda');pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        pipe=StableDiffusionInstructPix2PixPipeline.from_pretrained(a.model_path,torch_dtype=torch.float16,local_files_only=True,safety_checker=None)
        pipe.to('cuda');pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if a.model!='flux2_klein4b':residency='all_components_cuda'
    pipe.set_progress_bar_config(disable=True)
    components={name:m for name,m in pipe.components.items()if isinstance(m,torch.nn.Module)}
    for m in components.values():m.eval().requires_grad_(False)
    counts={name:sum(p.numel()for p in m.parameters())for name,m in components.items()}
    if sum(counts.values())>=10_000_000_000:raise ValueError('User excludes models with tens of billions of parameters')
    if a.model!='flux2_klein4b':
        for row in rows:
            if len(pipe.tokenizer(row['prompt'],truncation=False).input_ids)>pipe.tokenizer.model_max_length:
                raise ValueError('Full two-sentence instruction cannot be truncated')
    records=[];began=time.monotonic()
    write(a.output/'loaded_model.json',dict(complete=True,model=a.model,repo=cfg['repo'],revision=assets['revision'],
          component_parameters=counts,total_parameters=sum(counts.values()),trainable_parameters=0,
          upstream_defaults=cfg,actual_output_resolution=512,reference_information_resolution=256,
          residency=residency,source_packages={n:importlib.metadata.version(n)for n in ['torch','diffusers','transformers','accelerate']}))
    for i,row in enumerate(rows):
        src=a.data/row['reference']
        if sha(src)!=row['reference_sha256']:raise ValueError('Changed TEST input')
        with Image.open(src)as im:reference=im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
        effective=hashlib.sha256(reference.tobytes()).hexdigest()
        generator=torch.Generator(device='cuda').manual_seed(row['seed'])
        with torch.inference_mode():
            kwargs=dict(prompt=row['prompt'],image=reference,width=512,height=512,num_inference_steps=cfg['steps'],guidance_scale=cfg['guidance'],generator=generator)
            if 'image_guidance'in cfg:kwargs['image_guidance_scale']=cfg['image_guidance']
            image=pipe(**kwargs).images[0]
        if image.size!=(512,512):raise ValueError('Native512 output required; no output resizing')
        name=row['sample_id']+'.png';image.save(a.output/name)
        original=a.output/(row['reference_id']+'.png')
        if not original.exists():shutil.copyfile(src,original)
        records.append(dict(row,split='test',generated_PNG=name,generated_sha256=sha(a.output/name),input_sha256=sha(src),
            effective_RGB256_sha256=effective,checkpoint_sha256=identity,weights_manifest_sha256=identity,
            source_revision=assets['revision'],model=a.model,resolution=512,sampling_steps=cfg['steps'],task_finetuned=False))
        write(a.output/'images.json',records);write(a.output/'progress.json',dict(stage='native_frozen_preview8'if a.preview_only else'native_frozen_TEST500',done=i+1,total=len(rows),seconds=time.monotonic()-began))
        if i==7:write(a.output/'preview8.json',dict(complete=True,not_full500=True,balanced_two_categories=True,rows=records))
    if any(p.requires_grad or p.grad is not None for m in components.values()for p in m.parameters()):raise ValueError('Frozen native baseline only')
    write(a.output/'report.json',dict(complete=True,native_pretrained_peer=True,scope='TEST_native_preview8'if a.preview_only else'TEST_native_frozen_model',model=a.model,
        generated_count=len(rows),request_sha256=REQUESTS,training_performed=False,task_finetuned=False,trainable_parameters=0,
        component_parameters=counts,total_parameters=sum(counts.values()),repo=cfg['repo'],source_revision=assets['revision'],
        checkpoint_sha256=identity,checkpoint_identity_kind='official_weights_manifest_SHA256_not_task_PT',
        resolution=512,reference_information_resolution=256,prompt_rewritten=False,full_prompt_preserved=True,
        input_image_used=True,TEST_used_for_gradient=False,ImageReward_used=False,new_speed_claim=False,
        residency=residency,source_commit=os.environ['T12_SOURCE_COMMIT'],seconds=time.monotonic()-began))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    for name in ['data','model-path','asset-report','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ip2p-config',type=Path);p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    from .scene_execution_gpu import configured_gpu
    gpu=configured_gpu()
    if a.output.exists()or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('Fresh run and explicit published source/GPU required')
    import torch
    fraction=float(os.environ.get('T12_CUDA_MEMORY_FRACTION','.8'))
    if not 0<fraction<=.8:raise ValueError('Bounded explicit CUDA memory fraction required')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(fraction,0)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),GPU_UUID=gpu,GPU_name=torch.cuda.get_device_name(0),
        source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__)))
    try:infer(a)
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
