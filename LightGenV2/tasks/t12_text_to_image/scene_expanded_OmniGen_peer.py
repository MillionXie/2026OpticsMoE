"""Frozen native OmniGen on all500 same target-free selection requests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from .scene_instruction_data import SAFE,sha
from .scene_paper_expanded_data import PROFILE
from .scene_paper_prompt_protocol import write
from .white_reference_scene_probe import GPU


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','model-path','metadata','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('New published GPU6 run required')
    rows=json.loads((a.data/'inference_inputs.json').read_text());dr=json.loads((a.data/'report.json').read_text())
    if len(rows)!=500 or not dr['complete'] or dr['prompt_profile']!=PROFILE or dr['validation_set'] or not dr['TEST_used_for_model_selection']:
        raise ValueError('Fixed complete TEST500 selection requests required')
    if sha(a.data/'inference_inputs.json')!=dr['request_sha256'] or any(set(r)!=set(SAFE) for r in rows):raise ValueError('Changed public input manifest')
    a.output.mkdir(parents=True)
    write(a.output/'execution.json',dict(source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__),argv=sys.argv,pid=os.getpid()))
    try:
        import torch
        from PIL import Image
        from diffusers import OmniGenPipeline
        from .scene_omnigen_peer import checked_assets,REVISION
        write(a.output/'model_assets.json',checked_assets(a.model_path,a.metadata));torch.set_num_threads(4)
        pipe=OmniGenPipeline.from_pretrained(a.model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True);pipe.enable_vae_slicing()
        for component in pipe.components.values():
            if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
        records=[]
        for i,row in enumerate(rows):
            src=a.data/row['reference']
            if sha(src)!=row['reference_sha256']:raise ValueError('Changed white reference RGB')
            dst=a.output/(row['reference_id']+'.png')
            if not dst.exists():shutil.copyfile(src,dst)
            with Image.open(src) as im:reference=im.convert('RGB').copy()
            generator=torch.Generator(device='cuda').manual_seed(row['seed'])
            noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.bfloat16,generator=generator)
            noise_sha=hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
            with torch.inference_mode():
                image=pipe(prompt='<img><|image_1|></img>\n'+row['prompt'],input_images=[reference],height=512,width=512,
                    num_inference_steps=50,max_input_image_size=256,guidance_scale=2.5,img_guidance_scale=1.6,
                    use_input_image_size_as_output=False,generator=generator,latents=noise).images[0]
            name=row['sample_id']+'.png';image.save(a.output/name)
            records.append(dict(row,split='test',input_sha256=row['reference_sha256'],generated_PNG=name,generated_sha256=sha(a.output/name),
                initial_latent_sha256=noise_sha,checkpoint_sha256=REVISION,resolution=512,sampling_steps=50,
                TEST_used_for_model_selection=True,native_reference_VAE_is_peer_only=True,peer_has_no_task_tuning=True))
            write(a.output/'images.json',records);write(a.output/'progress.json',dict(stage='full_same_TEST500_frozen_peer',done=i+1,total=500))
        protocol=dict(schema='scene_expanded_TEST_selection_v1',prompt_profile=PROFILE,contexts=500,reference_views=26,
            request_sha256=dr['request_sha256'],validation_set=False,TEST_used_for_model_selection=True,independent_generalization_claim=False)
        write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],scope='TEST_selection',prompt_profile=PROFILE,
            generated_count=500,checkpoint_sha256=REVISION,training_performed=False,peer_has_no_task_tuning=True,
            independent_TEST_opened=False,independent_generalization_claim=False,validation_set=False,TEST_used_for_model_selection=True,
            TEST_selection_protocol=protocol,sampling_steps=50,resolution=512,image_feedback=False,quality_accepted=False))
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
