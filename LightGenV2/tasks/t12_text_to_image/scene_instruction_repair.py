"""TRAIN-only target repair using coarse form descriptions from white inputs.

These are stronger teacher construction prompts, NOT new student/benchmark
inputs. Public student text stays the two-sentence visible-anchor request. All
teacher frames, prompts and failures remain recorded; nothing is auto-accepted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from PIL import Image
from .scene_instruction_data import condition,sha,write,LAMP
from .scene_instruction_balanced import primary_codes,TEACHER_RECIPE
from .scene_generation_data import request_seed

FORM_PROFILE='TRAIN_source_form_support_teacher_v1'
FORMS={'articulated_desk':('lamp','an adjustable desk lamp'),'column_lamp':('lamp','a table lamp with a column base'),
       'rounded_lamp':('lamp','a table lamp with a rounded base'),'ornamental_lamp':('lamp','a table lamp with a sculpted base'),
       'rectangular_cushion':('pillow','a rectangular cushion'),'square_cushion':('pillow','a square cushion')}


def form_teacher_prompt(category,code,hint):
    if hint not in FORMS or FORMS[hint][0]!=category or code not in primary_codes(category):raise ValueError('Input-grounded supported coarse form required')
    noun=FORMS[hint][1]
    if category=='lamp':
        if code=='D':return f'Generate a bedroom photograph of {noun} on a bedside table. A bed is visible in the background.'
        return f'Generate an indoor photograph of {noun} on a table. '+LAMP[code][1]
    return condition(category,code)['prompt'].replace('a cushion',noun,1)


def prepare(plan,specification,output):
    if output.exists():raise ValueError('Fresh repair plan required')
    parent=json.loads((plan/'report.json').read_text(encoding='utf8'));rows=json.loads((plan/'requests.json').read_text(encoding='utf8'))
    if parent['schema']!='scene_balanced_visible_anchor_plan_v1'or sha(plan/'requests.json')!=parent['request_sha256']:raise ValueError('Fixed original TRAIN plan required')
    sources={r['reference_id']:r for r in rows};spec=json.loads(specification.read_text(encoding='utf8'));cases=spec['cases']
    if not spec.get('form_labels_from_white_input_only')or not cases:raise ValueError('Explicit input-only form audit required')
    annotations={};requests=[];seen=set();output.mkdir();(output/'references').mkdir()
    for case in cases:
        source=sources[case['reference_id']];code=case['anchor_code'];hint=case['source_form_hint']
        if source['split']!='train'or case['reference_sha256']!=source['reference_sha256']or (source['reference_id'],code)in seen:raise ValueError('Wrong source/role or duplicate repair group')
        seen.add((source['reference_id'],code));prompt=form_teacher_prompt(source['category'],code,hint)
        if sha(plan/source['reference'])!=source['reference_sha256']:raise ValueError('Source input changed')
        audit=dict(reference_sha256=source['reference_sha256'],source_form_hint=hint)
        if annotations.setdefault(source['reference_id'],audit)!=audit:raise ValueError('Conflicting source form descriptions')
        dst=output/source['reference']
        if not dst.exists():shutil.copyfile(plan/source['reference'],dst)
        for index in [0,1]:
            row=dict(source,**condition(source['category'],code));row.update(sample_id=source['reference_id']+'-form-repair-'+code+'-draw'+str(index),
                seed_index=index,seed=request_seed(source['reference_sha256'],index),prompt=prompt,native_caption=prompt,reward_caption=prompt,
                prompt_profile=FORM_PROFILE,anchor_code=code,source_form_hint=hint,reused_reviewed_label=None,
                public_training_prompt=condition(source['category'],code)['prompt'],teacher_prompt_has_extra_input_form_description=True)
            requests.append(row)
    write(output/'source_form_annotations.json',annotations);write(output/'requests.json',requests)
    write(output/'report.json',dict(complete=True,schema='scene_source_form_repair_plan_v1',source_commit=os.environ.get('T12_SOURCE_COMMIT'),
        source_plan_sha256=parent['request_sha256'],specification_sha256=sha(specification),request_sha256=sha(output/'requests.json'),
        source_form_annotations_sha256=sha(output/'source_form_annotations.json'),requests=len(requests),TRAIN_only=True,
        source_form_labels_enter_teacher_construction_prompt_only=True,student_public_prompt_not_changed=True,DEV_or_TEST_accessed=False,eligible_for_training=False))


def generate(a):
    import torch
    from diffusers import OmniGenPipeline
    from .scene_omnigen_peer import checked_assets,REVISION
    from .white_reference_scene_probe import GPU
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or a.output.exists():raise ValueError('Fresh serial GPU6 run required')
    report=json.loads((a.plan/'report.json').read_text(encoding='utf8'));rows=json.loads((a.plan/'requests.json').read_text(encoding='utf8'));forms=json.loads((a.plan/'source_form_annotations.json').read_text(encoding='utf8'))
    if report['schema']!='scene_source_form_repair_plan_v1'or sha(a.plan/'requests.json')!=report['request_sha256']or sha(a.plan/'source_form_annotations.json')!=report['source_form_annotations_sha256']or len(rows)!=report['requests']:raise ValueError('Changed repair plan')
    for row in rows:
        if row['split']!='train'or row['prompt']!=form_teacher_prompt(row['category'],row['anchor_code'],row['source_form_hint'])or forms[row['reference_id']]!=dict(reference_sha256=row['reference_sha256'],source_form_hint=row['source_form_hint']):raise ValueError('Changed input-only teacher specification')
    a.output.mkdir();(a.output/'references').mkdir();(a.output/'targets').mkdir();write(a.output/'requests.json',rows);shutil.copyfile(a.plan/'source_form_annotations.json',a.output/'source_form_annotations.json')
    write(a.output/'model_assets.json',dict(official_revision=REVISION,files=checked_assets(a.model_path,a.metadata)))
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    pipe=OmniGenPipeline.from_pretrained(a.model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');pipe.set_progress_bar_config(disable=True);pipe.enable_vae_slicing()
    for component in pipe.components.values():
        if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
    records=[]
    for row in rows:
        src=a.plan/row['reference']
        if sha(src)!=row['reference_sha256']:raise ValueError('Input changed')
        with Image.open(src)as im:reference=im.convert('RGB').copy()
        if not(a.output/row['reference']).exists():shutil.copyfile(src,a.output/row['reference'])
        generator=torch.Generator(device='cuda').manual_seed(row['seed']);noise=torch.randn((1,4,64,64),generator=generator,device='cuda',dtype=torch.bfloat16)
        noise_sha=hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
        with torch.inference_mode():image=pipe(prompt='<img><|image_1|></img>\n'+row['prompt'],input_images=[reference],height=512,width=512,num_inference_steps=50,max_input_image_size=256,guidance_scale=2.5,img_guidance_scale=1.6,use_input_image_size_as_output=False,generator=generator,latents=noise).images[0]
        filename='targets/'+row['sample_id']+'.png';image.save(a.output/filename)
        records.append(dict(row,target=filename,target_sha256=sha(a.output/filename),QA_accepted=False,initial_latent_sha256=noise_sha))
        write(a.output/'candidates.json',records);write(a.output/'progress.json',dict(stage='TRAIN_source_form_target_repair',done=len(records),total=len(rows)))
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ.get('T12_SOURCE_COMMIT'),schema='scene_source_form_TRAIN_candidates_v1',prompt_profile=FORM_PROFILE,
        candidate_manifest_sha256=sha(a.output/'candidates.json'),source_form_annotations_sha256=sha(a.output/'source_form_annotations.json'),teacher_recipe=TEACHER_RECIPE,
        requests=len(rows),TRAIN_only=True,DEV_or_TEST_forwarded=False,eligible_for_training=False,student_public_prompt_not_changed=True,
        stronger_teacher_construction_prompt_not_a_benchmark_output=True,all_candidates_require_per_fact_QA=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','generate'],required=True)
    for key in ['plan','specification','model-path','metadata','output']:p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.output is None or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:raise ValueError('Explicit fresh output and published source required')
    if a.stage=='prepare':prepare(a.plan,a.specification,a.output)
    else:generate(a)


if __name__=='__main__':main()
