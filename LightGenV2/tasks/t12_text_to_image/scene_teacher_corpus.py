"""Whole-scene public-teacher candidates; no foreground pasted into a room.

ABO white RGB and Generate text are the only model inputs. All candidates,
including failures, are kept. Completion is NOT data acceptance. TEST is sealed.
Fine colors/textures are free; the requested coarse shape, scene/support and
furniture still require QA. TRAIN/DEV keep their existing product identities.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from PIL import Image,ImageDraw
from .appearance_scene_protocol import COARSE_PROFILE,conditions
from .scene_generation_data import request_seed
from .white_reference_scene_probe import GPU,digest,write,checked_tokens,model_manifest


def requests_for(sources,pilot=False,double_seed_scenes=None,train_scene_stride=1):
    if pilot:
        chosen=[]
        for category in ['lamp','pillow']:
            unique={}
            for row in sources:
                if row['split']=='train' and row['category']==category:unique.setdefault(row['product_id'],row)
            keys=sorted(unique);indices=np.linspace(0,len(keys)-1,4).round().astype(int)
            chosen.extend(unique[keys[i]] for i in indices)
        sources=chosen
    rows=[]
    for source_index,source in enumerate(sources):
        assert source['split'] in ['train','dev']
        for scene in conditions(source['category'],COARSE_PROFILE):
            if pilot and scene['condition_id'] not in ['0','1']:continue
            if not pilot and source['split']=='train' and int(scene['condition_id'])%train_scene_stride!=source_index%train_scene_stride:continue
            indices=[2] if source['split']=='dev' else ([0,1] if pilot or double_seed_scenes is None or int(scene['condition_id']) in double_seed_scenes else [0])
            for index in indices:
                rows.append({**scene,'sample_id':source['sample_id']+'-scene'+scene['condition_id']+'-draw'+str(index),
                    'split':source['split'],'reference_id':source['sample_id'],
                    'reference':'references/'+source['sample_id']+'.png','reference_sha256':source['white_reference_sha256'],
                    'source_product_id':source['product_id'],'source_RGBA_sha256':source['rgba_file_sha256'],
                    'seed_index':index,'seed':request_seed(source['white_reference_sha256'],index)})
    if len({(r['reference'],r['prompt'],r['seed']) for r in rows})!=len(rows):raise ValueError('Duplicate model inputs')
    return rows


def generate(a):
    import torch
    import cv2
    from diffusers import StableDiffusionXLControlNetPipeline,ControlNetModel
    from transformers import CLIPVisionModelWithProjection
    cfg=json.loads(a.config.read_text(encoding='utf8'))
    assert cfg['prompt_profile']==COARSE_PROFILE and cfg['resolution'] in [512,1024] and cfg['batch_size_max']==2
    resolution=cfg['resolution']
    write(a.output/'configuration.json',{'profile':cfg,'config_sha256':digest(a.config)})
    source_report=json.loads((a.sources/'report.json').read_text(encoding='utf8'))
    assert source_report['complete'] and not source_report['TEST_opened']
    sources=json.loads((a.sources/'sources.json').read_text(encoding='utf8'))
    rows=requests_for(sources,a.pilot,cfg.get('TRAIN_double_seed_scenes'),cfg.get('TRAIN_scene_stride',1))
    if a.reuse_pilot and cfg.get('include_extra_TRAIN_pilot_requests'):
        old_report=json.loads((a.reuse_pilot/'report.json').read_text())
        assert old_report['complete'] and old_report['pilot_only'] and old_report['resolution']==resolution
        prior=json.loads((a.reuse_pilot/'candidates.json').read_text());keys=set(rows[0]);known={r['sample_id'] for r in rows}
        sources_by_id={r['sample_id']:r for r in sources}
        for row in prior:
            assert row['split']=='train' and row['prompt_profile']==COARSE_PROFILE
            assert sources_by_id[row['reference_id']]['white_reference_sha256']==row['reference_sha256']
            if row['sample_id'] not in known:rows.append({k:row[k] for k in keys});known.add(row['sample_id'])
    source_map={r['sample_id']:r for r in sources}
    (a.output/'references').mkdir();(a.output/'targets').mkdir()
    for ref in {r['reference_id'] for r in rows}:
        source=source_map[ref];src=a.sources/source['white_reference']
        assert digest(src)==source['white_reference_sha256']
        shutil.copyfile(src,a.output/'references'/(ref+'.png'))
    write(a.output/'requests.json',rows)
    write(a.output/'model_assets.json',{'SDXL':model_manifest(a.model_path),
        'IPAdapter':model_manifest(a.adapter_path),'ControlNet':model_manifest(a.controlnet_path)})
    dtype=torch.float16
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    control=ControlNetModel.from_pretrained(a.controlnet_path,variant='fp16',torch_dtype=dtype,local_files_only=True)
    encoder=CLIPVisionModelWithProjection.from_pretrained(a.adapter_path/'models/image_encoder',torch_dtype=dtype,local_files_only=True)
    pipe=StableDiffusionXLControlNetPipeline.from_pretrained(a.model_path,controlnet=control,
        image_encoder=encoder,variant='fp16',torch_dtype=dtype,local_files_only=True).to('cuda')
    pipe.load_ip_adapter(str(a.adapter_path),subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.safetensors',image_encoder_folder=None,local_files_only=True)
    pipe.set_ip_adapter_scale(cfg['IP_scale']);pipe.set_progress_bar_config(disable=True);pipe.enable_vae_slicing()
    for c in pipe.components.values():
        if isinstance(c,torch.nn.Module):c.eval().requires_grad_(False)
    groups=defaultdict(list)
    for row in rows:groups[(row['reference_id'],row['condition_id'])].append(row)
    records=[];reused={}
    if a.reuse_pilot:
        prior=json.loads((a.reuse_pilot/'report.json').read_text(encoding='utf8'))
        assert prior['complete'] and prior['pilot_only'] and prior['prompt_profile']==COARSE_PROFILE
        assert prior['resolution']==resolution,'Do not present resized old teacher images as native new-resolution generations'
        old_cfg=json.loads((a.reuse_pilot/'configuration.json').read_text())['profile']
        assert all(old_cfg[k]==cfg[k] for k in ['resolution','sampling_steps','text_guidance','IP_scale','Canny_scale','Canny_end'])
        assert json.loads((a.reuse_pilot/'model_assets.json').read_text())==json.loads((a.output/'model_assets.json').read_text())
        reused={r['sample_id']:r for r in json.loads((a.reuse_pilot/'candidates.json').read_text())}
    for i,subset in enumerate(groups.values()):
        keep=[]
        for row in subset:
            if row['sample_id'] in reused:
                old=reused[row['sample_id']]
                assert all(row[k]==old[k] for k in ['reference_sha256','prompt','native_caption','seed'])
                assert digest(a.reuse_pilot/old['target'])==old['target_sha256']
                shutil.copyfile(a.reuse_pilot/old['target'],a.output/old['target'])
                records.append(old)
            else:keep.append(row)
        if not keep:continue
        assert len(keep) in [1,2] and len({r['prompt'] for r in keep})==1
        row=keep[0]
        with Image.open(a.output/row['reference']) as im:reference=im.convert('RGB').copy()
        edges=cv2.Canny(np.asarray(reference.resize((resolution,resolution),Image.Resampling.LANCZOS)),100,200)
        audit=[checked_tokens(t,row['native_caption']) for t in [pipe.tokenizer,pipe.tokenizer_2]]
        generators=[torch.Generator(device='cuda').manual_seed(r['seed']) for r in keep]
        noises=[torch.randn((1,4,resolution//8,resolution//8),device='cuda',dtype=dtype,generator=g) for g in generators]
        noise_sha=[hashlib.sha256(n.cpu().numpy().tobytes()).hexdigest() for n in noises]
        began=time.monotonic()
        with torch.inference_mode():
            images=pipe(prompt=row['native_caption'],ip_adapter_image=reference,
                image=Image.fromarray(np.repeat(edges[:,:,None],3,axis=2)),
                controlnet_conditioning_scale=cfg['Canny_scale'],control_guidance_end=cfg['Canny_end'],
                width=resolution,height=resolution,num_inference_steps=cfg['sampling_steps'],guidance_scale=cfg['text_guidance'],
                num_images_per_prompt=len(keep),latents=torch.cat(noises),generator=generators).images
        assert len(images)==len(keep)
        for r,image,noise_hash in zip(keep,images,noise_sha):
            filename='targets/'+r['sample_id']+'.png';image.save(a.output/filename)
            records.append({**r,'target':filename,'target_sha256':digest(a.output/filename),
                'initial_latent_sha256':noise_hash,'native_caption_tokens':audit,
                'teacher':'frozen public SDXL/IP-Adapter/Canny','initial_latent_dtype':'float16',
                'RGB_paste_or_target_inputs':False,'QA_accepted':False})
        write(a.output/'candidates.json',records)
        write(a.output/'progress.json',{'stage':'whole_scene_teacher_candidates','done':len(records),'total':len(rows),
            'batch_size':len(keep),'last_batch_seconds_not_latency_benchmark':time.monotonic()-began})
    assert {r['sample_id'] for r in records}=={r['sample_id'] for r in rows}
    records.sort(key=lambda r:r['sample_id']);write(a.output/'candidates.json',records)
    for page,start in enumerate(range(0,len(records),32)):
        selected=records[start:start+32];sheet=Image.new('RGB',(8*256,((len(selected)+7)//8)*286),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(selected):
            x=(j%8)*256;y=(j//8)*286
            with Image.open(a.output/r['target']) as im:sheet.paste(im.resize((256,256)),(x,y+30))
            draw.text((x+2,y+2),r['reference_id']+' s'+r['condition_id']+' n'+str(r['seed_index']),fill='black')
        sheet.save(a.output/f'all_candidates_QA_page{page+1}.png')
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],
        'prompt_profile':COARSE_PROFILE,'pilot_only':a.pilot,'candidate_count':len(records),'resolution':resolution,
        'candidate_manifest_sha256':digest(a.output/'candidates.json'),'TEST_opened':False,
        'sources_report_sha256':digest(a.sources/'report.json'),
        'no_foreground_compositing':True,'no_rotation_or_image_feedback':True,
        'all_candidates_require_acceptance':True,'eligible_for_training':False,
        'backgrounds_are_AI_generated_not_external_room_photographs':True,
        'foreground_source_attribution':'Amazon Berkeley Objects (ABO), CC BY 4.0',
        'public_model_outputs_license_separate_from_ABO':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/scene_teacher_corpus_20261029.json')
    for k in ['sources','model-path','adapter-path','controlnet-path','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--reuse-pilot',type=Path)
    a=p.parse_args()
    if a.output.exists() or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:
        raise ValueError('New run/published source/GPU6 required')
    a.output.mkdir(parents=True)
    write(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ['T12_SOURCE_COMMIT'],
        'source_sha256':digest(Path(__file__))})
    try:generate(a)
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
