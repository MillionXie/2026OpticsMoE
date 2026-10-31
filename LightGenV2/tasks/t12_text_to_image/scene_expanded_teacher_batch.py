"""Same frozen native50-step teacher, batched TRAIN generation with preserved pixels.

Batching can change floating point rounding and therefore output pixels. It is
recorded as a new generation recipe, never described as B1 bit identity or an
LG inference speed measurement. TEST still has one output per fixed request.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from .scene_paper_expanded_targets import checked_plan
from .scene_instruction_data import SAFE,sha
from .scene_paper_prompt_protocol import write


def balanced_order(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['category'],row['condition_id']].append(row)
    for group in groups.values():group.sort(key=lambda r:r['reference_id'])
    result=[]
    condition_ids=sorted({r['condition_id'] for r in rows},key=int)
    for offset in range(len(condition_ids)):
        for source_index in range(max(map(len,groups.values()))):
            condition=condition_ids[(offset+source_index)%len(condition_ids)]
            for category in ['lamp','pillow']:
                group=groups[category,condition]
                if source_index<len(group):result.append(group[source_index])
    if len(result)!=len(rows) or len({r['sample_id'] for r in result})!=len(rows):raise ValueError('Lost or repeated TRAIN request')
    return result


def preserved(folder,assets,revision):
    report=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else None
    if report is None:
        marker=json.loads((folder/'superseded_generation.json').read_text())
        if(marker.get('own_generator_stopped')is not True or marker.get('GPU_released')is not True
           or marker['candidate_manifest_sha256']!=sha(folder/'candidates.json')):
            raise ValueError('Only a stopped, SHA-bound own generation prefix can resume')
    elif not report['complete']:raise ValueError('Incomplete prior generation without explicit stop evidence')
    if json.loads((folder/'model_assets.json').read_text())!=assets:raise ValueError('Teacher assets changed')
    result={}
    for row in json.loads((folder/'candidates.json').read_text()):
        if(row['split']!='train' or row['teacher_revision']!=revision or row['GT_masks_or_targets_as_inputs']is not False
           or row['native_prompt']!='<img><|image_1|></img>\n'+row['prompt']
           or sha(folder/row['reference'])!=row['reference_sha256'] or sha(folder/row['target'])!=row['target_sha256']):
            raise ValueError('Preserved target recipe or pixels changed')
        key=(row['reference_sha256'],row['prompt'],row['seed'])
        if key in result:raise ValueError('Conflicting preserved requests')
        result[key]=row
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','model-path','metadata','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['pilot','generate'],required=True);p.add_argument('--batch-size',type=int,choices=[2,4],default=2)
    p.add_argument('--steps',type=int,choices=[30,50],default=50)
    p.add_argument('--reuse',type=Path,nargs='*',default=[]);p.add_argument('--pilot-review',type=Path)
    a=p.parse_args()
    from .white_reference_scene_probe import GPU
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('New source-pinned own GPU6 run required')
    plan,rows=checked_plan(a.plan)
    if a.stage=='generate':
        review=json.loads(a.pilot_review.read_text())
        if not review['approved_for_TRAIN_target_generation'] or review['batch_size']!=a.batch_size or review['native_steps']!=a.steps:
            raise ValueError('Actual batched TRAIN pilot visual/resource review required')
    import torch
    from PIL import Image
    from diffusers import OmniGenPipeline
    from .scene_omnigen_peer import checked_assets,REVISION
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88,0)
    assets=checked_assets(a.model_path,a.metadata);a.output.mkdir(parents=True)
    (a.output/'targets').mkdir();(a.output/'references').mkdir()
    write(a.output/'execution.json',dict(source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__),argv=sys.argv,pid=os.getpid()))
    write(a.output/'model_assets.json',assets)
    try:
        pipe=OmniGenPipeline.from_pretrained(a.model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True);pipe.enable_vae_slicing()
        for component in pipe.components.values():
            if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
        chosen=rows
        if a.stage=='pilot':
            chosen=[]
            for category in ['lamp','pillow']:
                sources=sorted({r['reference_id'] for r in rows if r['category']==category})
                for index,rid in enumerate(sources[:2]):chosen.append(next(r for r in rows if r['reference_id']==rid and r['condition_id']==str(index)))
        index={r['sample_id']:i for i,r in enumerate(rows)};records={};reused=0
        for folder in a.reuse:
            prior=preserved(folder,assets,REVISION)
            for row in chosen:
                old=prior.get((row['reference_sha256'],row['prompt'],row['seed']))
                if old is None or row['sample_id'] in records:continue
                target='targets/'+row['sample_id']+'.png';shutil.copyfile(folder/old['target'],a.output/target)
                record=dict(old,**{k:row[k] for k in SAFE});record.update(target=target,reused_from_run=str(folder))
                records[row['sample_id']]=record;reused+=1
        for row in chosen:
            path=a.output/row['reference']
            if not path.exists():shutil.copyfile(a.plan/row['reference'],path)
        todo=balanced_order([r for r in chosen if r['sample_id']not in records]);durations=[]
        def publish():
            write(a.output/'candidates.json',sorted(records.values(),key=lambda r:index[r['sample_id']]))
            write(a.output/'progress.json',dict(stage='batched_TRAIN_only_candidates_pending_review',done=len(records),total=len(chosen),
                reused=reused,batch_size=a.batch_size,batch_wall_seconds=durations[-1] if durations else None))
        publish();torch.cuda.reset_peak_memory_stats()
        for start in range(0,len(todo),a.batch_size):
            group=todo[start:start+a.batch_size];images=[];generators=[];noises=[];noise_shas=[]
            for row in group:
                with Image.open(a.plan/row['reference'])as im:images.append([im.convert('RGB').copy()])
                generator=torch.Generator(device='cuda').manual_seed(row['seed']);generators.append(generator)
                noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.bfloat16,generator=generator);noises.append(noise)
                noise_shas.append(hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest())
            torch.cuda.synchronize();began=time.perf_counter()
            with torch.inference_mode():
                outputs=pipe(prompt=['<img><|image_1|></img>\n'+r['prompt'] for r in group],input_images=images,
                    height=512,width=512,num_inference_steps=a.steps,max_input_image_size=256,guidance_scale=2.5,img_guidance_scale=1.6,
                    use_input_image_size_as_output=False,generator=generators,latents=torch.cat(noises)).images
            torch.cuda.synchronize();durations.append(time.perf_counter()-began)
            if len(outputs)!=len(group):raise ValueError('Batch output/sample binding changed')
            for row,image,noise_sha in zip(group,outputs,noise_shas):
                target='targets/'+row['sample_id']+'.png';image.save(a.output/target)
                records[row['sample_id']]=dict(row,split='train',target=target,target_sha256=sha(a.output/target),
                    initial_latent_sha256=noise_sha,native_prompt='<img><|image_1|></img>\n'+row['prompt'],teacher_revision=REVISION,
                    GT_masks_or_targets_as_inputs=False,QA_accepted=False,teacher_inference_batch_size=len(group),
                    teacher_generation_steps=a.steps,teacher_kind='frozen_official_OmniGen',
                    batching_may_change_rounding_and_pixels=True)
            publish()
        write(a.output/'report.json',dict(complete=True,schema='scene_paper_expanded_TRAIN_candidates_v1',prompt_profile=plan['prompt_profile'],
            source_commit=os.environ['T12_SOURCE_COMMIT'],TRAIN_request_sha256=plan['TRAIN_request_sha256'],
            candidate_manifest_sha256=sha(a.output/'candidates.json'),candidate_count=len(records),TRAIN_views=len({r['reference_id']for r in chosen}),
            batch_size=a.batch_size,batch_wall_seconds=durations,seconds_per_new_image=sum(durations)/max(1,len(todo)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),reused_completed_targets=reused,
            native_steps=a.steps,semantic_review_not_performed=True,eligible_for_training=False,
            TEST_targets_or_predictions_created=False,pilot_only=a.stage=='pilot'))
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
