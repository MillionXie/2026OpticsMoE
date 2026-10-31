"""Faster frozen SDXL+IP TRAIN target pilot and explicit mixed-teacher expansion.

No Qwen or trained paired checkpoint is used by this teacher. Original white
ABO RGB and the full equivalent native caption are its only conditions. The
original input-image encoder belongs to this native teacher, not the Qwen model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from .scene_instruction_data import SAFE,sha
from .scene_paper_prompt_protocol import write
from .scene_paper_expanded_targets import checked_plan
from .scene_expanded_teacher_batch import balanced_order
from .white_reference_scene_probe import GPU,model_manifest

KIND='frozen_original_SDXL_IP30_v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','model-path','adapter-path','config','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['pilot','generate'],required=True);p.add_argument('--reuse-prefix',type=Path);p.add_argument('--pilot-review',type=Path)
    a=p.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:raise ValueError('New source-pinned GPU6 run required')
    report,rows=checked_plan(a.plan);cfg=json.loads(a.config.read_text())
    if a.stage=='generate':
        proof=json.loads(a.pilot_review.read_text())
        if not proof['approved_for_TRAIN_target_generation'] or proof['teacher_kind']!=KIND:raise ValueError('Actual native teacher pilot review required')
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88,0)
    assets=dict(SDXL=model_manifest(a.model_path),IPAdapter=model_manifest(a.adapter_path))
    a.output.mkdir(parents=True);(a.output/'references').mkdir();(a.output/'targets').mkdir()
    write(a.output/'model_assets.json',assets);revision=sha(a.output/'model_assets.json')
    write(a.output/'execution.json',dict(source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__),argv=sys.argv,pid=os.getpid(),config_sha256=sha(a.config)))
    try:
        encoder=CLIPVisionModelWithProjection.from_pretrained(a.adapter_path/'models/image_encoder',torch_dtype=torch.bfloat16,local_files_only=True)
        pipe=StableDiffusionXLPipeline.from_pretrained(a.model_path,variant='fp16',torch_dtype=torch.bfloat16,image_encoder=encoder,local_files_only=True).to('cuda')
        pipe.load_ip_adapter(str(a.adapter_path),subfolder='sdxl_models',weight_name='ip-adapter-plus_sdxl_vit-h.safetensors',image_encoder_folder=None,local_files_only=True)
        pipe.set_ip_adapter_scale(.25);pipe.set_progress_bar_config(disable=True)
        for component in pipe.components.values():
            if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
        chosen=rows
        if a.stage=='pilot':
            chosen=[]
            for category in ['lamp','pillow']:
                refs=sorted({r['reference_id'] for r in rows if r['category']==category})[:2]
                for i,rid in enumerate(refs):
                    chosen.extend(next(r for r in rows if r['reference_id']==rid and r['condition_id']==str(j))for j in [i,2])
        records={};timings=[];index={r['sample_id']:i for i,r in enumerate(rows)};inherited=0
        if a.reuse_prefix:
            old=a.reuse_prefix;marker=json.loads((old/'superseded_generation.json').read_text())
            if not marker['own_generator_stopped'] or not marker['GPU_released'] or sha(old/'candidates.json')!=marker['candidate_manifest_sha256']:
                raise ValueError('Preserve only an explicitly stopped own SHA-bound TRAIN prefix')
            write(a.output/'preserved_OmniGen_model_assets.json',json.loads((old/'model_assets.json').read_text()))
            allowed={(r['reference_sha256'],r['prompt'],r['seed']):r for r in chosen}
            for prior in json.loads((old/'candidates.json').read_text()):
                row=allowed.get((prior['reference_sha256'],prior['prompt'],prior['seed']))
                if row is None:continue
                if prior['split']!='train' or prior['GT_masks_or_targets_as_inputs']is not False or sha(old/prior['target'])!=prior['target_sha256']:
                    raise ValueError('Preserved TRAIN prefix identity changed')
                name='targets/'+row['sample_id']+'.png';shutil.copyfile(old/prior['target'],a.output/name)
                record=dict(prior,**{k:row[k]for k in SAFE});record.update(target=name,teacher_kind='frozen_official_OmniGen50_B1',teacher_generation_steps=50,reused_from_run=str(old))
                records[row['sample_id']]=record;inherited+=1
        for row in chosen:
            dst=a.output/row['reference']
            if not dst.exists():shutil.copyfile(a.plan/row['reference'],dst)
        for row in balanced_order([r for r in chosen if r['sample_id']not in records]):
            generator=torch.Generator(device='cuda').manual_seed(row['seed'])
            noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.bfloat16,generator=generator)
            noise_sha=hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
            with Image.open(a.plan/row['reference'])as im:reference=im.convert('RGB').copy()
            torch.cuda.synchronize();began=time.perf_counter()
            with torch.inference_mode():
                image=pipe(prompt=row['native_caption'],negative_prompt=cfg['negative_prompt'],ip_adapter_image=reference,
                    height=512,width=512,num_inference_steps=30,guidance_scale=7.5,latents=noise,generator=generator,
                    original_size=(1024,1024),target_size=(1024,1024)).images[0]
            torch.cuda.synchronize();timings.append(time.perf_counter()-began)
            name='targets/'+row['sample_id']+'.png';image.save(a.output/name)
            records[row['sample_id']]=dict(row,split='train',target=name,target_sha256=sha(a.output/name),initial_latent_sha256=noise_sha,
                native_prompt=row['native_caption'],teacher_kind=KIND,teacher_revision=revision,teacher_generation_steps=30,
                teacher_inference_batch_size=1,GT_masks_or_targets_as_inputs=False,QA_accepted=False)
            write(a.output/'candidates.json',sorted(records.values(),key=lambda r:index[r['sample_id']]))
            write(a.output/'progress.json',dict(stage='frozen_native_TRAIN_candidates_pending_review',done=len(records),total=len(chosen),
                inherited_OmniGen50_targets=inherited,last_image_seconds=timings[-1]))
        write(a.output/'report.json',dict(complete=True,schema='scene_paper_expanded_TRAIN_candidates_v1',prompt_profile=report['prompt_profile'],
            source_commit=os.environ['T12_SOURCE_COMMIT'],TRAIN_request_sha256=report['TRAIN_request_sha256'],candidate_manifest_sha256=sha(a.output/'candidates.json'),
            candidate_count=len(records),teacher_kind=KIND,inherited_OmniGen50_targets=inherited,new_native30_targets=len(timings),
            native_generation_seconds=timings,mean_seconds_per_new_image=sum(timings)/max(1,len(timings)),
            mixed_teacher_recipes_explicit=bool(inherited),semantic_review_not_performed=True,eligible_for_training=False,
            TEST_targets_or_predictions_created=False,pilot_only=a.stage=='pilot'))
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
