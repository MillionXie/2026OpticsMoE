"""TRAIN-only same-reference counterfactual scene coverage with checked labels.

Choose one original white view per TRAIN product by visible input area, before
looking at any model output. Every chosen reference receives all three primary
requests at the same two source-derived seeds. Reuse only pixel-verified factual
labels; generate missing examples with the pinned frozen public teacher.
"""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from .scene_instruction_data import PROFILE,condition,sha,write
from .scene_generation_data import request_seed


def primary_codes(category):
    if category=='lamp':return ['B','D','W']
    if category=='pillow':return ['S','A','D']
    raise ValueError('Approved categories only')


def source_area(path):
    with Image.open(path)as im:rgb=np.asarray(im.convert('RGB').resize((256,256)))
    visible=rgb.min(axis=2)<242
    return int(visible.sum())


def choose_sources(rows,root):
    products=defaultdict(dict)
    for row in rows:
        if row['split']!='train':continue
        if sha(root/row['reference'])!=row['reference_sha256']:raise ValueError('Input pixels changed')
        products[row['source_product_id']].setdefault(row['reference_id'],row)
    chosen=[];audit=[]
    for pid,refs in sorted(products.items()):
        scored=[(source_area(root/r['reference']),rid,r)for rid,r in refs.items()]
        scored.sort(key=lambda x:(-x[0],x[1]));winner=scored[0][2];chosen.append(winner)
        audit.append(dict(product_id=pid,chosen_reference_id=winner['reference_id'],input_only_visible_pixel_scores={rid:n for n,rid,_ in scored}))
    return chosen,audit


def make_plan(data,output):
    if output.exists():raise ValueError('Fresh plan required')
    report=json.loads((data/'report.json').read_text(encoding='utf8'))
    if not report['complete'] or report['prompt_profile']!=PROFILE or report['TEST_opened'] or sha(data/'pairs.json')!=report['pairs_sha256']:raise ValueError('Reviewed factual labels required')
    pairs=json.loads((data/'pairs.json').read_text(encoding='utf8'));chosen,audit=choose_sources(pairs,data)
    if len(chosen)!=44 or len({r['reference_sha256']for r in chosen})!=44:raise ValueError('Keep all original44 distinct TRAIN products/inputs')
    existing={(r['reference_sha256'],r['prompt'],r['seed']):r for r in pairs if r['split']=='train'}
    pilots=set()
    for category in ['lamp','pillow']:
        subset=sorted([r for r in chosen if r['category']==category],key=lambda r:r['source_product_id'])
        for index in np.linspace(0,len(subset)-1,4).round().astype(int):pilots.add(subset[index]['reference_id'])
    output.mkdir();(output/'references').mkdir();rows=[]
    for source in chosen:
        dst=output/source['reference'];dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(data/source['reference'],dst)
        for code in primary_codes(source['category']):
            d=condition(source['category'],code)
            for index in [0,1]:
                seed=request_seed(source['reference_sha256'],index);old=existing.get((source['reference_sha256'],d['prompt'],seed))
                row=dict(d,sample_id=source['reference_id']+'-balanced-'+code+'-draw'+str(index),split='train',reference_id=source['reference_id'],
                    reference=source['reference'],reference_sha256=source['reference_sha256'],source_product_id=source['source_product_id'],
                    source_RGBA_sha256=source['source_RGBA_sha256'],seed_index=index,seed=seed,pilot=source['reference_id']in pilots)
                row['reused_reviewed_label']=None if old is None else dict(data_root=str(data),target=old['target'],target_sha256=old['target_sha256'],original_sample_id=old['sample_id'])
                rows.append(row)
    assert len(rows)==264 and sum(r['pilot']for r in rows)==48
    for ref in chosen:
        subset=[r for r in rows if r['reference_id']==ref['reference_id']]
        assert len(subset)==6 and all(len({r['seed']for r in subset if r['seed_index']==index})==1 for index in [0,1])
    write(output/'requests.json',rows);write(output/'input_only_reference_selection.json',audit)
    write(output/'report.json',dict(complete=True,schema='scene_balanced_visible_anchor_plan_v1',source_commit=os.environ.get('T12_SOURCE_COMMIT'),
        prompt_profile=PROFILE,source_pairs_sha256=report['pairs_sha256'],request_sha256=sha(output/'requests.json'),
        TRAIN_products=44,TRAIN_reference_views=44,requests=264,primary_conditions_per_same_reference=3,seeds_per_condition=2,pilot_requests=48,
        reused_reviewed_targets=sum(r['reused_reviewed_label']is not None for r in rows),new_teacher_candidates_needed=sum(r['reused_reviewed_label']is None for r in rows),
        chosen_from_input_pixels_only=True,all44_products_retained=True,DEV_or_TEST_generated=False,eligible_for_training=False,model_generation_performed=False))


TEACHER_RECIPE=dict(steps=50,text_guidance=2.5,image_guidance=1.6,input_size=256,output_size=512,dtype='bfloat16')
SCENE_FIRST_PROFILE='two_sentence_scene_first_anchor_v6'


def scene_first_condition(category,code):
    """Same required entities, expressed as scene first; no reference directives."""
    row=condition(category,code)
    if category=='lamp':
        noun={'B':'bookshelves','D':'a bed','W':'a window'}[code]
        prompt=f'Generate an indoor photograph with {noun} in the background. A table lamp stands on a table.'
    else:
        noun={'S':'a sofa','A':'an armchair','D':'a bed'}[code]
        prompt=f'Generate an indoor photograph showing {noun}. A cushion rests on the '+{'S':'sofa','A':'armchair','D':'bed'}[code]+'.'
    row.update(prompt=prompt,native_caption=prompt,reward_caption=prompt,prompt_profile=SCENE_FIRST_PROFILE)
    return row


def prepare_scene_first(plan,output):
    if output.exists():raise ValueError('Fresh controlled syntax pilot required')
    report=json.loads((plan/'report.json').read_text(encoding='utf8'))
    if report.get('schema')!='scene_balanced_visible_anchor_plan_v1'or sha(plan/'requests.json')!=report['request_sha256']:raise ValueError('Fixed balanced TRAIN plan required')
    original=json.loads((plan/'requests.json').read_text(encoding='utf8'))
    selected=[r for r in original if r['pilot']and r['seed_index']==0]
    if len(selected)!=24 or any(r['split']!='train'for r in selected):raise ValueError('Fixed8 TRAIN references/3 scenes/first source seed required')
    output.mkdir();(output/'references').mkdir();rows=[]
    for source in selected:
        if sha(plan/source['reference'])!=source['reference_sha256']:raise ValueError('Input changed')
        dst=output/source['reference']
        if not dst.exists():shutil.copyfile(plan/source['reference'],dst)
        code=source['sample_id'].split('-balanced-')[1].split('-draw')[0]
        revised=scene_first_condition(source['category'],code)
        row=dict(source,**revised);row.update(sample_id=source['sample_id']+'-sceneFirst',previous_sample_id=source['sample_id'],previous_prompt=source['prompt'],reused_reviewed_label=None)
        rows.append(row)
    write(output/'requests.json',rows)
    write(output/'report.json',dict(complete=True,schema='scene_balanced_scene_first_pilot_plan_v1',prompt_profile=SCENE_FIRST_PROFILE,
        source_commit=os.environ.get('T12_SOURCE_COMMIT'),source_plan_sha256=report['request_sha256'],request_sha256=sha(output/'requests.json'),
        TRAIN_products=8,requests=24,same_references_seeds_and_required_facts=True,only_sentence_order_and_wording_changed=True,
        DEV_or_TEST_generated=False,eligible_for_training=False,model_generation_performed=False))


def checked_prior_candidates(prior,plan_report,rows,assets):
    """Reuse completed TRAIN pilot pixels, including failures, without acceptance."""
    if prior is None:return {}
    report=json.loads((prior/'report.json').read_text(encoding='utf8'))
    records=json.loads((prior/'candidates.json').read_text(encoding='utf8'))
    if not report.get('complete')or not report.get('pilot_only')or report.get('eligible_for_training')or report.get('DEV_or_TEST_forwarded'):
        raise ValueError('Completed unaccepted TRAIN pilot required')
    if report.get('schema')!='scene_balanced_visible_anchor_candidates_v1'or report.get('plan_sha256')!=plan_report['request_sha256']or report.get('candidate_manifest_sha256')!=sha(prior/'candidates.json'):
        raise ValueError('Prior pilot identity changed')
    # The original published pilot predates explicit recipe metadata. Its exact
    # published implementation is the sole supported legacy recipe identity.
    recipe=report.get('teacher_recipe')
    if recipe!=TEACHER_RECIPE and not(recipe is None and report.get('source_commit')=='d2919422b6035d60918f8a944e7e055b0fbda212'):
        raise ValueError('Different or unspecified teacher sampling recipe')
    if json.loads((prior/'model_assets.json').read_text(encoding='utf8'))!=assets:raise ValueError('Different teacher assets')
    expected={r['sample_id']:r for r in rows if r['pilot']}
    if len(records)!=48 or len(expected)!=48 or {r['sample_id']for r in records}!=set(expected):raise ValueError('All fixed48 pilot candidates required')
    result={}
    for row in records:
        base=expected[row['sample_id']]
        if any(row.get(key)!=value for key,value in base.items())or row.get('QA_accepted')is not False:raise ValueError('Changed pilot request or acceptance')
        if sha(prior/row['reference'])!=row['reference_sha256']or sha(prior/row['target'])!=row['target_sha256']:raise ValueError('Changed pilot pixels')
        result[row['sample_id']]=row
    return result


def generate(plan,model_path,metadata,output,pilot,reuse_candidates=None):
    import torch
    from diffusers import OmniGenPipeline
    from .scene_omnigen_peer import checked_assets,REVISION
    from .white_reference_scene_probe import GPU
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or output.exists():raise ValueError('Fresh run on GPU6 required')
    report=json.loads((plan/'report.json').read_text(encoding='utf8'));rows=json.loads((plan/'requests.json').read_text(encoding='utf8'))
    scene_first=report['schema']=='scene_balanced_scene_first_pilot_plan_v1'
    if report['schema']not in ['scene_balanced_visible_anchor_plan_v1','scene_balanced_scene_first_pilot_plan_v1']or not report['complete']or sha(plan/'requests.json')!=report['request_sha256']:raise ValueError('Changed balanced plan')
    if scene_first and(pilot or reuse_candidates is not None):raise ValueError('Scene-first pilot is a fresh24-request controlled comparison')
    rows=[r for r in rows if r['pilot']]if pilot else rows
    if len(rows)!=(24 if scene_first else 48 if pilot else 264)or any(r['split']!='train'for r in rows):raise ValueError('Only prespecified TRAIN requests')
    if pilot and reuse_candidates is not None:raise ValueError('Reuse is only for full264 expansion')
    assets=dict(official_revision=REVISION,files=checked_assets(model_path,metadata))
    previous=checked_prior_candidates(reuse_candidates,report,rows,assets)
    output.mkdir();(output/'references').mkdir();(output/'targets').mkdir();write(output/'requests.json',rows)
    write(output/'model_assets.json',assets)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    pipe=OmniGenPipeline.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True);pipe.enable_vae_slicing()
    for component in pipe.components.values():
        if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
    records=[];generated=0;reused=0;prior_count=0
    for row in rows:
        src=plan/row['reference']
        if sha(src)!=row['reference_sha256']:raise ValueError('Input changed')
        dst=output/row['reference']
        if not dst.exists():shutil.copyfile(src,dst)
        filename='targets/'+row['sample_id']+'.png';reuse=row['reused_reviewed_label'];noise_sha=None
        prior=previous.get(row['sample_id'])
        if prior is not None:
            shutil.copyfile(reuse_candidates/prior['target'],output/filename);prior_count+=1
            noise_sha=prior['initial_latent_sha256']
        elif reuse is not None:
            target=Path(reuse['data_root'])/reuse['target']
            if sha(target)!=reuse['target_sha256']:raise ValueError('Reviewed target changed')
            shutil.copyfile(target,output/filename);reused+=1
        else:
            with Image.open(src)as im:reference=im.convert('RGB').copy()
            generator=torch.Generator(device='cuda').manual_seed(row['seed']);noise=torch.randn((1,4,64,64),generator=generator,device='cuda',dtype=torch.bfloat16)
            noise_sha=hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
            with torch.inference_mode():image=pipe(prompt='<img><|image_1|></img>\n'+row['prompt'],input_images=[reference],height=512,width=512,
                num_inference_steps=50,max_input_image_size=256,guidance_scale=2.5,img_guidance_scale=1.6,use_input_image_size_as_output=False,generator=generator,latents=noise).images[0]
            image.save(output/filename);generated+=1
        records.append(dict(row,target=filename,target_sha256=sha(output/filename),QA_accepted=False,
            initial_latent_sha256=noise_sha,teacher='frozen official OmniGen-v1'if reuse is None else'previously reviewed factual training example',
            copied_prior_candidate=prior is not None,
            RGB_paste_or_target_inputs=False,scene_GT_is_one_valid_training_example=True))
        write(output/'candidates.json',records);write(output/'progress.json',dict(stage='balanced_TRAIN_teacher_candidates',done=len(records),total=len(rows),new_generated=generated,reused_reviewed=reused,reused_prior_candidates=prior_count))
    write(output/'report.json',dict(complete=True,source_commit=os.environ.get('T12_SOURCE_COMMIT'),schema='scene_scene_first_TRAIN_candidates_v1'if scene_first else'scene_balanced_visible_anchor_candidates_v1',prompt_profile=report['prompt_profile'],
        candidate_manifest_sha256=sha(output/'candidates.json'),plan_sha256=report['request_sha256'],pilot_only=pilot,counts=dict(Counter(r['split']for r in rows)),
        resolution=512,new_generated=generated,reused_reviewed=reused,reused_prior_candidates=prior_count,teacher_recipe=TEACHER_RECIPE,
        prior_candidate_manifest_sha256=None if reuse_candidates is None else sha(reuse_candidates/'candidates.json'),
        all_candidates_require_per_fact_QA=True,eligible_for_training=False,
        optical_or_Qwen_training_performed=False,DEV_or_TEST_forwarded=False,reference_VAE_belongs_to_teacher_not_Qwen=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','prepare-scene-first','generate'],required=True)
    for key in ['data','plan','model-path','metadata','output','reuse-candidates']:p.add_argument('--'+key,type=Path)
    p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.output is None or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:raise ValueError('Published source and explicit output required')
    if a.stage=='prepare':make_plan(a.data,a.output)
    elif a.stage=='prepare-scene-first':prepare_scene_first(a.plan,a.output)
    else:generate(a.plan,a.model_path,a.metadata,a.output,a.pilot,a.reuse_candidates)


if __name__=='__main__':main()
