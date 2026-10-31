"""Generate TRAIN-only expanded targets, reusing only exact native-recipe pixels.

TEST500 stays target-free. Generated candidates are not automatically approved
for semantic correctness or gradients. The existing v7 runs are immutable.
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
from .scene_paper_prompt_protocol import write
from .scene_paper_expanded_data import PROFILE,SCHEMA,conditions
from .scene_generation_data import request_seed


def checked_plan(plan):
    report=json.loads((plan/'report.json').read_text())
    rows=json.loads((plan/'train_inputs.json').read_text());test=json.loads((plan/'test_inputs.json').read_text())
    if (report.get('schema')!=SCHEMA or not report['complete'] or report['prompt_profile']!=PROFILE
        or report['validation_set'] or not report['TEST_used_for_model_selection'] or report['targets_created']
        or report['eligible_for_training'] or sha(plan/'train_inputs.json')!=report['TRAIN_request_sha256']
        or sha(plan/'test_inputs.json')!=report['request_sha256'] or len(rows)!=1520 or len(test)!=500):
        raise ValueError('Explicit complete single-seed TRAIN1520/TEST500 plan required')
    if {r['reference_sha256'] for r in rows}&{r['reference_sha256'] for r in test}:
        raise ValueError('TEST inputs leaked into target generation')
    seen=set()
    for row in rows:
        definition=conditions(row['category'])[int(row['condition_id'])]
        if set(row)!=set(SAFE) or row['sample_id'] in seen or row['seed']!=request_seed(row['reference_sha256'],0):
            raise ValueError('One source-only seed and the public whitelist required')
        if any(row[k]!=definition[k] for k in ['prompt','native_caption','reward_caption','prompt_profile']):
            raise ValueError('Prompt changed after plan publication')
        if sha(plan/row['reference'])!=row['reference_sha256']: raise ValueError('Source RGB changed')
        seen.add(row['sample_id'])
    for rid in {r['reference_id'] for r in rows}:
        group=[r for r in rows if r['reference_id']==rid]
        if len(group)!=20 or {r['condition_id'] for r in group}!={str(i) for i in range(20)}:
            raise ValueError('Twenty distinct scenes per TRAIN view required')
    if len({r['reference_id'] for r in rows})!=76:raise ValueError('Keep all76 TRAIN views')
    return report,rows


def reusable_candidates(folder,assets,revision):
    report=json.loads((folder/'report.json').read_text())
    if (not report['complete'] or report.get('schema')!='scene_paper_TRAIN_candidates_v1'
        or report['DEV_or_TEST_generated'] or json.loads((folder/'model_assets.json').read_text())!=assets
        or sha(folder/'candidates.json')!=report['candidate_manifest_sha256']):
        raise ValueError('Completed exact-model TRAIN-only reuse required')
    found={}
    for row in json.loads((folder/'candidates.json').read_text()):
        if (row['teacher_revision']!=revision or row['GT_masks_or_targets_as_inputs'] is not False
            or row['native_prompt']!='<img><|image_1|></img>\n'+row['prompt']
            or sha(folder/row['reference'])!=row['reference_sha256'] or sha(folder/row['target'])!=row['target_sha256']):
            raise ValueError('Candidate provenance or pixels changed')
        key=(row['reference_sha256'],row['prompt'],row['seed'])
        if key in found:raise ValueError('Conflicting targets for identical input')
        found[key]=row
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','model-path','metadata','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reuse',type=Path)
    a=p.parse_args()
    from .white_reference_scene_probe import GPU
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('New run, published source and authorized GPU6 required')
    plan,rows=checked_plan(a.plan)
    import torch
    from PIL import Image
    from diffusers import OmniGenPipeline
    from .scene_omnigen_peer import checked_assets,REVISION
    assets=checked_assets(a.model_path,a.metadata)
    reused=reusable_candidates(a.reuse,assets,REVISION) if a.reuse else {}
    a.output.mkdir(parents=True);(a.output/'targets').mkdir();(a.output/'references').mkdir()
    write(a.output/'execution.json',dict(source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__),argv=sys.argv,pid=os.getpid()))
    write(a.output/'model_assets.json',assets);write(a.output/'requests.json',rows)
    try:
        torch.set_num_threads(4)
        pipe=OmniGenPipeline.from_pretrained(a.model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True);pipe.enable_vae_slicing()
        for component in pipe.components.values():
            if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
        records=[];reuse_count=0
        for index,row in enumerate(rows):
            source=a.plan/row['reference'];destination=a.output/row['reference']
            if not destination.exists():shutil.copyfile(source,destination)
            name='targets/'+row['sample_id']+'.png';key=(row['reference_sha256'],row['prompt'],row['seed'])
            prior=reused.get(key);noise_sha=None
            if prior:
                shutil.copyfile(a.reuse/prior['target'],a.output/name);noise_sha=prior['initial_latent_sha256'];reuse_count+=1
            else:
                with Image.open(source) as im:reference=im.convert('RGB').copy()
                generator=torch.Generator(device='cuda').manual_seed(row['seed'])
                noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.bfloat16,generator=generator)
                noise_sha=hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
                with torch.inference_mode():
                    image=pipe(prompt='<img><|image_1|></img>\n'+row['prompt'],input_images=[reference],height=512,width=512,
                        num_inference_steps=50,max_input_image_size=256,guidance_scale=2.5,img_guidance_scale=1.6,
                        use_input_image_size_as_output=False,generator=generator,latents=noise).images[0]
                assert image.size==(512,512);image.save(a.output/name)
            records.append(dict(row,split='train',target=name,target_sha256=sha(a.output/name),initial_latent_sha256=noise_sha,
                native_prompt='<img><|image_1|></img>\n'+row['prompt'],teacher_revision=REVISION,
                GT_masks_or_targets_as_inputs=False,QA_accepted=False,reused_exact_v7_target=bool(prior),
                source_candidate_sample_id=prior['sample_id'] if prior else None))
            write(a.output/'candidates.json',records)
            write(a.output/'progress.json',dict(stage='TRAIN_only_candidates_pending_review',done=index+1,total=len(rows),reused=reuse_count))
        write(a.output/'report.json',dict(complete=True,schema='scene_paper_expanded_TRAIN_candidates_v1',prompt_profile=PROFILE,
            source_commit=os.environ['T12_SOURCE_COMMIT'],TRAIN_request_sha256=plan['TRAIN_request_sha256'],
            candidate_manifest_sha256=sha(a.output/'candidates.json'),candidate_count=len(records),TRAIN_views=76,TRAIN_products=44,
            one_seed_per_reference=True,reused_exact_v7_targets=reuse_count,TEST_targets_or_predictions_created=False,
            eligible_for_training=False,semantic_review_not_performed=True,backgrounds_AI_generated_not_external_photographs=True))
    except BaseException:
        write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
