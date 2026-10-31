"""Fresh SDXL+IP TRAIN feasibility control; no old task adapter, no Qwen claim.

All24 prespecified same-reference scene-first requests. This tests whether the
native generator can render the requested entities before further paired tuning.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from PIL import Image
from .scene_instruction_data import sha,write,SAFE
from .scene_instruction_balanced import SCENE_FIRST_PROFILE
from .scene_reference_framing import frame_reference,FRAMING


def visible_caption(category,code):
    """Same visible entities in descriptive CLIP format; no target-derived facts."""
    if category=='lamp':return 'An indoor photograph of a table lamp standing on a table, with '+{'B':'bookshelves','D':'a bed','W':'a window','S':'a sofa','A':'an armchair'}[code]+' in the background.'
    if category=='pillow':
        if code=='W':return 'An indoor photograph of a cushion on a wooden surface, with a window in the background.'
        return 'An indoor photograph of a cushion resting on '+{'S':'a sofa','A':'an armchair','D':'a bed'}[code]+'.'
    raise ValueError('Approved product category required')


def descriptive_equivalent(row):
    """Native CLIP wording derived only from the public instruction's facts."""
    from .scene_instruction_balanced import scene_first_condition
    code=({'0':'B','1':'D','2':'W'}if row['category']=='lamp'else{'0':'S','1':'A','2':'D'})[row['condition_id']]
    if row['prompt']!=scene_first_condition(row['category'],code)['prompt']:raise ValueError('Only the explicit public-equivalent condition is supported')
    return visible_caption(row['category'],code)


def checked_requests(plan):
    report=json.loads((plan/'report.json').read_text(encoding='utf8'))
    rows=json.loads((plan/'requests.json').read_text(encoding='utf8'))
    if report.get('schema')!='scene_balanced_scene_first_pilot_plan_v1'or not report.get('complete')or sha(plan/'requests.json')!=report['request_sha256']or len(rows)!=24:
        raise ValueError('Fixed completed scene-first TRAIN24 plan required')
    if any(r['split']!='train'or r['prompt_profile']!=SCENE_FIRST_PROFILE for r in rows):raise ValueError('TRAIN-only new prompt profile')
    if len({r['reference_id']for r in rows})!=8:raise ValueError('All8 fixed TRAIN references required')
    for row in rows:
        if sha(plan/row['reference'])!=row['reference_sha256']:raise ValueError('Input changed')
    return [{k:r[k]for k in SAFE}for r in rows]


def generate(a):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection
    from .white_reference_scene_probe import GPU,model_manifest,checked_tokens
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or a.output.exists():raise ValueError('Fresh serial GPU6 run required')
    rows=checked_requests(a.plan);a.output.mkdir();(a.output/'references').mkdir();(a.output/'generated').mkdir()
    write(a.output/'requests.json',rows);write(a.output/'model_assets.json',dict(SDXL=model_manifest(a.model_path),IPAdapter=model_manifest(a.adapter_path)))
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    encoder=CLIPVisionModelWithProjection.from_pretrained(a.adapter_path/'models/image_encoder',torch_dtype=torch.bfloat16,local_files_only=True)
    pipe=StableDiffusionXLPipeline.from_pretrained(a.model_path,variant='fp16',torch_dtype=torch.bfloat16,image_encoder=encoder,local_files_only=True).to('cuda')
    pipe.load_ip_adapter(str(a.adapter_path),subfolder='sdxl_models',weight_name='ip-adapter-plus_sdxl_vit-h.safetensors',image_encoder_folder=None,local_files_only=True)
    pipe.set_ip_adapter_scale(.25);pipe.set_progress_bar_config(disable=True)
    for c in pipe.components.values():
        if isinstance(c,torch.nn.Module):c.eval().requires_grad_(False)
    negative='drawing, painting, illustration, cartoon, sketch, line art, outlines, flat shading, CGI, 3d render, watermark, text'
    records=[]
    for row in rows:
        for tokenizer in [pipe.tokenizer,pipe.tokenizer_2]:checked_tokens(tokenizer,row['prompt'])
        with Image.open(a.plan/row['reference'])as im:reference=im.convert('RGB').copy()
        model_reference,framing=frame_reference(reference,a.reference_framing)
        dst=a.output/row['reference']
        if not dst.exists():shutil.copyfile(a.plan/row['reference'],dst)
        generator=torch.Generator(device='cuda').manual_seed(row['seed']);noise=torch.randn((1,4,64,64),generator=generator,device='cuda',dtype=torch.bfloat16)
        native_prompt=descriptive_equivalent(row)if a.condition_style=='descriptive_equivalent'else row['prompt']
        for tokenizer in [pipe.tokenizer,pipe.tokenizer_2]:checked_tokens(tokenizer,native_prompt)
        with torch.inference_mode():image=pipe(prompt=native_prompt,negative_prompt=negative,ip_adapter_image=model_reference,height=512,width=512,
            original_size=(1024,1024),target_size=(1024,1024),num_inference_steps=30,guidance_scale=7.5,latents=noise,generator=generator).images[0]
        filename='generated/'+row['sample_id']+'.png';image.save(a.output/filename)
        records.append(dict(row,native_condition_prompt=native_prompt,reference_framing=framing,generated_PNG=filename,generated_sha256=sha(a.output/filename),initial_latent_sha256=hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()))
        write(a.output/'images.json',records);write(a.output/'progress.json',dict(stage='fresh_native_TRAIN24_generation',done=len(records),total=24))
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ.get('T12_SOURCE_COMMIT'),schema='scene_fresh_native_TRAIN_feasibility_v1',
        prompt_profile=SCENE_FIRST_PROFILE,request_sha256=sha(a.plan/'requests.json'),requests=24,resolution=512,sampling_steps=30,IP_scale=.25,
        old_task_LoRA_installed=False,Qwen_used=False,TRAIN_only=True,DEV_or_TEST_accessed=False,automatic_instruction_success_certification=False,
        teacher_or_Qwen_baseline_accepted=False,negative_prompt=negative,SDXL_conditioning_size=1024,native_condition_style=a.condition_style,
        native_caption_is_public_semantic_equivalent=True,no_SKU_GT_or_unrequested_attributes_in_caption=True,reference_framing=a.reference_framing))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','model-path','adapter-path','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--condition-style',choices=['public_instruction','descriptive_equivalent'],default='public_instruction')
    p.add_argument('--reference-framing',choices=['original',FRAMING],default='original')
    a=p.parse_args()
    if len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:raise ValueError('Published source identity required')
    generate(a)


if __name__=='__main__':main()
