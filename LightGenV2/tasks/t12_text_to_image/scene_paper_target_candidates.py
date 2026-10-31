"""New v7 TRAIN targets from a frozen native peer, pending pixel-bound review.

The plan has no targets. This generator never substitutes DEV images for TRAIN,
recaptions existing outputs, or automatically approves its own new candidates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from .scene_instruction_data import SAFE,sha
from .scene_paper_prompt_protocol import PROFILE,PARENT_PAIRS,condition,write
from .scene_generation_data import request_seed


def checked_train_rows(report,rows,dev):
    if (report.get('schema')!='scene_paper_request_plan_v1' or not report.get('complete')
        or report.get('prompt_profile')!=PROFILE or report.get('source_pairs_sha256')!=PARENT_PAIRS
        or report.get('TEST_opened')is not False or report.get('role_disjoint')is not True
        or report.get('targets_created')is not False or report.get('eligible_for_training')is not False):
        raise ValueError('Only the explicit target-free v7 plan is admitted')
    if len(rows)!=264 or len(dev)!=48 or len({r['reference_id']for r in rows})!=44:
        raise ValueError('Keep the complete44TRAIN references and264requests')
    if {r['reference_sha256']for r in rows}&{r['reference_sha256']for r in dev}:
        raise ValueError('DEV reference leaked into target generation')
    ids=set()
    for r in rows:
        if set(r)!=set(SAFE)or r['sample_id']in ids:
            raise ValueError('Public input whitelist and unique requests required')
        ids.add(r['sample_id']);definition=condition(r['category'],r['condition_id'])
        if any(r[k]!=definition[k]for k in ['prompt','native_caption','reward_caption','prompt_profile']):
            raise ValueError('Do not change the frozen requested facts')
        draw=next((i for i in [0,1]if r['sample_id']==r['reference_id']+'-paper-scene'+r['condition_id']+'-draw'+str(i)),None)
        if draw is None or r['seed']!=request_seed(r['reference_sha256'],draw):
            raise ValueError('Noise must depend only on source RGB and draw index')
    for rid in {r['reference_id']for r in rows}:
        group=[r for r in rows if r['reference_id']==rid]
        if len(group)!=6 or {(r['condition_id'],r['seed'])for r in group}!={(str(i),request_seed(group[0]['reference_sha256'],j))for i in range(3)for j in [0,1]}:
            raise ValueError('All three same-source counterfactuals and two draws required')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','model-path','metadata','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();from .white_reference_scene_probe import GPU
    if a.output.exists()or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:
        raise ValueError('New run, published source and authorized GPU6 required')
    report=json.loads((a.plan/'report.json').read_text());rows=json.loads((a.plan/'train_inputs.json').read_text());dev=json.loads((a.plan/'dev_inputs.json').read_text())
    checked_train_rows(report,rows,dev)
    if sha(a.plan/'train_inputs.json')!=report['TRAIN_request_sha256']or sha(a.plan/'dev_inputs.json')!=report['request_sha256']:
        raise ValueError('Plan manifest changed')
    for r in rows:
        if sha(a.plan/r['reference'])!=r['reference_sha256']:raise ValueError('Reference bytes changed')
    a.output.mkdir(parents=True);(a.output/'references').mkdir();(a.output/'targets').mkdir()
    write(a.output/'execution.json',dict(source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__),argv=sys.argv,pid=os.getpid(),TRAIN_requests_sha256=report['TRAIN_request_sha256']))
    try:
        import torch
        from PIL import Image
        from diffusers import OmniGenPipeline
        from .scene_omnigen_peer import checked_assets,REVISION
        write(a.output/'model_assets.json',checked_assets(a.model_path,a.metadata));write(a.output/'requests.json',rows)
        torch.set_num_threads(4)
        pipe=OmniGenPipeline.from_pretrained(a.model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True);pipe.enable_vae_slicing()
        for component in pipe.components.values():
            if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
        records=[]
        for i,r in enumerate(rows):
            source=a.plan/r['reference'];dst=a.output/r['reference']
            if not dst.exists():shutil.copyfile(source,dst)
            with Image.open(source)as im:reference=im.convert('RGB').copy()
            generator=torch.Generator(device='cuda').manual_seed(r['seed']);noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.bfloat16,generator=generator)
            noise_sha=hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest();prompt='<img><|image_1|></img>\n'+r['prompt']
            with torch.inference_mode():image=pipe(prompt=prompt,input_images=[reference],height=512,width=512,num_inference_steps=50,max_input_image_size=256,guidance_scale=2.5,img_guidance_scale=1.6,use_input_image_size_as_output=False,generator=generator,latents=noise).images[0]
            assert image.size==(512,512);filename='targets/'+r['sample_id']+'.png';image.save(a.output/filename)
            records.append(dict(r,split='train',target=filename,target_sha256=sha(a.output/filename),initial_latent_sha256=noise_sha,native_prompt=prompt,teacher='frozen official OmniGen-v1',teacher_revision=REVISION,QA_accepted=False,GT_masks_or_targets_as_inputs=False))
            write(a.output/'candidates.json',records);write(a.output/'progress.json',dict(stage='new_v7_TRAIN_targets_pending_visual_review',done=i+1,total=len(rows)))
        write(a.output/'report.json',dict(complete=True,schema='scene_paper_TRAIN_candidates_v1',prompt_profile=PROFILE,source_commit=os.environ['T12_SOURCE_COMMIT'],TRAIN_request_sha256=report['TRAIN_request_sha256'],candidate_manifest_sha256=sha(a.output/'candidates.json'),candidate_count=len(records),TRAIN_references=44,DEV_or_TEST_generated=False,training_performed=False,eligible_for_training=False,all_candidates_require_pixel_bound_review=True,old_target_pixels_or_prompts_not_relabelled=True,backgrounds_AI_generated_not_external_photographs=True))
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
