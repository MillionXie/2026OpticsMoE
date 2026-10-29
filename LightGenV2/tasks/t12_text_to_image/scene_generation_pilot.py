"""Generate ALL 64 TRAIN candidates before label review; no DEV/TEST inference."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from PIL import Image,ImageDraw
from .white_reference_scene_probe import GPU,digest,write,checked_tokens,model_manifest
from .white_reference_outline_probe import structural_canny_from_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','model-path','controlnet-path','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();commit=os.environ.get('T12_SOURCE_COMMIT','')
    if a.output.exists() or len(commit)!=40 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('New run, Git source, GPU6 required')
    audit=json.loads((a.data/'data_audit.json').read_text());assert audit['role_disjoint'] and not audit['generation_or_features_performed']
    rows=json.loads((a.data/'pilot_train_requests.json').read_text());assert len(rows)==64 and all(r['split']=='train' for r in rows)
    a.output.mkdir(parents=True);write(a.output/'execution.json',{'pid':os.getpid(),'argv':sys.argv,'source_commit':commit,
      'source_sha256':digest(Path(__file__)),'requests_sha256':digest(a.data/'pilot_train_requests.json'),
      'versions':{n:importlib.metadata.version(n) for n in ['torch','diffusers','transformers']}})
    try:
        import torch
        from diffusers import ControlNetModel,StableDiffusionXLControlNetPipeline
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.7,0)
        write(a.output/'model_assets.json',{'base':model_manifest(a.model_path),'controlnet':model_manifest(a.controlnet_path)})
        cn=ControlNetModel.from_pretrained(a.controlnet_path,variant='fp16',torch_dtype=torch.float16,local_files_only=True)
        pipe=StableDiffusionXLControlNetPipeline.from_pretrained(a.model_path,controlnet=cn,variant='fp16',torch_dtype=torch.float16,local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        for c in pipe.components.values():
            if isinstance(c,torch.nn.Module):c.eval().requires_grad_(False)
        recipe={'white_threshold':245,'minimum_component_area':9,'maximum_product_extent':.68,'center_y':.53,
          'edge_width':2,'canny_blur_sigma':1.5,'canny_low':100,'canny_high':200}
        settings={'resolution':1024,'steps':30,'CFG':7.5,'control_scale':.35,'control_end':.5,
          'negative_prompt':'people, readable text, logo, watermark, duplicate product, malformed structure, flat solid-color backdrop'}
        write(a.output/'generation_config.json',{'geometry':recipe,**settings,'IPAdapter_or_extra_image_encoder':False})
        geometry={};records=[]
        for i,row in enumerate(rows):
            path=a.data/row['reference'];assert digest(path)==row['reference_sha256']
            if row['reference_id'] not in geometry:
                with Image.open(path) as im:edge,meta=structural_canny_from_rgb(im.convert('RGB'),1024,recipe)
                edge.save(a.output/(row['reference_id']+'_edge.png'));geometry[row['reference_id']]=(edge,meta)
            edge,meta=geometry[row['reference_id']]
            token_audit=[checked_tokens(t,row['prompt']) for t in [pipe.tokenizer,pipe.tokenizer_2]]
            generator=torch.Generator(device='cuda').manual_seed(row['seed'])
            noise=torch.randn((1,4,128,128),generator=generator,device='cuda',dtype=torch.float16)
            noise_sha=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest();start=time.time()
            with torch.inference_mode():
                image=pipe(prompt=row['prompt'],negative_prompt=settings['negative_prompt'],image=edge,width=1024,height=1024,
                  latents=noise,generator=generator,num_inference_steps=30,guidance_scale=7.5,controlnet_conditioning_scale=.35,
                  control_guidance_start=0.,control_guidance_end=.5).images[0]
            name=row['sample_id']+'.png';image.save(a.output/name)
            records.append({**row,'generated_PNG':name,'generated_sha256':digest(a.output/name),
              'initial_noise_sha256':noise_sha,'full_prompt_tokens':token_audit,'input_geometry':meta,
              'training_label_status':'candidate_NOT_accepted','generation_seconds_not_benchmark':time.time()-start})
            write(a.output/'images.json',records);write(a.output/'progress.json',{'done':i+1,'total':len(rows),'stage':'TRAIN_candidates_only'})
        # Two counterfactual prompts with a shared source-derived seed must start identically.
        for ref in geometry:
            for seed_index in [0,1]:assert len({r['initial_noise_sha256'] for r in records if r['reference_id']==ref and r['seed_index']==seed_index})==1
        sheets=[]
        for ref in geometry:
            selected=[r for r in records if r['reference_id']==ref];sheet=Image.new('RGB',(1024,1400),'white');draw=ImageDraw.Draw(sheet)
            draw.text((4,4),'TRAIN candidates, not approved labels | '+ref+' | all four conditions and both seeds',fill='black')
            with Image.open(a.data/selected[0]['reference']) as im:sheet.paste(im.resize((160,160)),(432,24))
            for j,r in enumerate(selected):
                x=(j%2)*512;y=200+(j//2)*300
                with Image.open(a.output/r['generated_PNG']) as im:sheet.paste(im.resize((250,250)),(x,y))
                import textwrap
                draw.multiline_text((x+255,y),str(r['seed'])+'\n'+'\n'.join(textwrap.wrap(r['prompt'],34)),fill='black',spacing=2)
            name=ref+'_all_candidates.png';sheet.save(a.output/name);sheets.append(name)
        write(a.output/'report.json',{'complete':True,'source_commit':commit,'generated_count':len(records),'training_targets_accepted':0,
          'independent_TEST_opened':False,'DEV_inference':False,'training_performed':False,'input_whitelist':['white reference RGB','full Generate prompt','source-derived seed'],
          'old_GT_mask_retrieval_or_background_photo_used':False,'all_conditions_and_seeds_retained':True,'panels':sheets,
          'limitation':'Candidate generation for TRAIN label QA, not a trained Qwen baseline or independent evaluation.'})
        del pipe;gc.collect();torch.cuda.empty_cache()
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
