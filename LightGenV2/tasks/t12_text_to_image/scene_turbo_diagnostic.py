"""Official edge-to-image Turbo: eight diagnostic requests, not an RGB baseline.

Only redirect upstream pretrained asset paths to verified local copies. The
upstream model, one-step scheduler, Canny thresholds and inference stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from PIL import Image
from .scene_instruction_data import sha,write
from .scene_native_edit_peers import canonical_requests,preview_first

UPSTREAM='86f54146590ffb4543c8cf85b5a36657da670924'


def verify(a):
    report=json.loads(a.assets.read_text())
    if not report['complete'] or report['upstream_commit']!=UPSTREAM:
        raise ValueError('Pinned official Turbo assets required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.upstream,text=True).strip()!=UPSTREAM:
        raise ValueError('Upstream source revision drift')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=a.upstream,text=True).strip():
        raise ValueError('Do not modify the upstream model implementation')
    for rel,item in report['base_files'].items():
        p=a.base/rel
        if Path(rel).is_absolute() or '..'in Path(rel).parts or sha(p)!=item['sha256']:
            raise ValueError('SD-Turbo base asset drift')
    if sha(a.checkpoint)!=report['edge_checkpoint']['sha256']:
        raise ValueError('Official edge checkpoint drift')
    return report


def infer(a):
    import torch
    from unittest.mock import patch
    from transformers import AutoTokenizer,CLIPTextModel
    from diffusers import AutoencoderKL,UNet2DConditionModel,DDPMScheduler
    from torchvision.transforms.functional import to_tensor,to_pil_image
    assets=verify(a)
    sys.path.insert(0,str(a.upstream/'src'))
    from pix2pix_turbo import Pix2Pix_Turbo
    from image_prep import canny_from_pil
    # Keep the upstream constructor and checkpoint loader. Redirect only its
    # hard-coded Hub identifier, with explicit local FP16 base weight variants.
    from contextlib import ExitStack
    def local_loader(cls):
        original=cls.from_pretrained
        def load(name,*args,**kwargs):
            if name!='stabilityai/sd-turbo':raise ValueError('Unexpected upstream dependency')
            kwargs['local_files_only']=True
            if cls in [CLIPTextModel,AutoencoderKL,UNet2DConditionModel]:kwargs['variant']='fp16'
            return original(str(a.base),*args,**kwargs)
        return load
    with ExitStack()as stack:
        for cls in [AutoTokenizer,CLIPTextModel,AutoencoderKL,UNet2DConditionModel,DDPMScheduler]:
            stack.enter_context(patch.object(cls,'from_pretrained',side_effect=local_loader(cls)))
        model=Pix2Pix_Turbo(pretrained_path=str(a.checkpoint))
    model.set_eval();model.eval().requires_grad_(False).half()
    rows=preview_first(canonical_requests(a.data))[:8];records=[];began=time.monotonic()
    count=sum(p.numel()for p in model.parameters())
    write(a.output/'loaded_model.json',dict(complete=True,total_parameters=count,trainable_parameters=0,
        upstream_commit=UPSTREAM,source_commit=os.environ['T12_SOURCE_COMMIT'],base_assets_sha256=sha(a.assets)))
    for i,row in enumerate(rows):
        src=a.data/row['reference']
        if sha(src)!=row['reference_sha256']:raise ValueError('Changed TEST RGB input')
        with Image.open(src)as im:
            reference=im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
        native=reference.resize((512,512),Image.Resampling.BICUBIC)
        control=canny_from_pil(native,100,200)
        if len(model.tokenizer(row['prompt'],truncation=False).input_ids)>model.tokenizer.model_max_length:
            raise ValueError('The complete instruction must fit')
        torch.manual_seed(row['seed']);torch.cuda.manual_seed_all(row['seed'])
        with torch.inference_mode():out=model(to_tensor(control).unsqueeze(0).cuda().half(),row['prompt'])
        result=to_pil_image(out[0].cpu()*.5+.5)
        if result.size!=(512,512):raise ValueError('Native512 required')
        png=row['sample_id']+'.png';edge=row['sample_id']+'-canny.png'
        result.save(a.output/png);control.save(a.output/edge)
        target=a.output/(row['reference_id']+'.png')
        if not target.exists():target.write_bytes(src.read_bytes())
        records.append(dict(row,split='test',generated_PNG=png,generated_sha256=sha(a.output/png),
            input_sha256=sha(src),effective_RGB256_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
            control_PNG=edge,control_sha256=sha(a.output/edge),model='pix2pix_turbo_edge_diagnostic',
            checkpoint_sha256=assets['edge_checkpoint']['sha256'],sampling_steps=1,task_finetuned=False))
        write(a.output/'images.json',records)
        write(a.output/'progress.json',dict(stage='edge_domain_diagnostic_only',done=i+1,total=8,seconds=time.monotonic()-began))
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Frozen diagnostic only')
    write(a.output/'report.json',dict(complete=True,generated_count=8,scope='TEST_edge_Turbo_diagnostic8',
        formal_RGB_baseline=False,input_domain_mismatch=True,control='Canny_of_shared_RGB256_upsampled512',
        canny_thresholds=[100,200],native_output_resolution=512,upstream_commit=UPSTREAM,
        checkpoint_sha256=assets['edge_checkpoint']['sha256'],assets_manifest_sha256=sha(a.assets),
        full_prompt_preserved=True,task_finetuned=False,TEST_used_for_gradient=False,
        source_commit=os.environ['T12_SOURCE_COMMIT'],ImageReward_computed=False,new_speed_claim=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['data','upstream','base','checkpoint','assets','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    from .scene_execution_gpu import configured_gpu
    gpu=configured_gpu()
    if a.output.exists()or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('Fresh diagnostic run and explicit GPU/source required')
    import torch
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    write(a.output/'execution.json',dict(pid=os.getpid(),argv=sys.argv,GPU_UUID=gpu,GPU_name=torch.cuda.get_device_name(0),
        source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__)))
    try:infer(a)
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
