"""Independent frozen OmniGen-v1 peer; white RGB/text, native512, no feedback.

Full DEV32 denominator, official 50-step preset, not the SDXL 30-step preset.
The reference VAE belongs to this native peer, never to the Qwen baseline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from PIL import Image
from .appearance_scene_protocol import COARSE_PROFILE,SHORT_PROFILE,conditions
from .scene_generation_data import request_seed
from .white_reference_scene_probe import GPU,digest,write
from .scene_reference_framing import frame_reference,FRAMING

REVISION='016e2f61d12a98303f6bbdf122687694d7984268'


def matched_natural_inputs(data,revision):
    """Reuse the exact target-free natural DEV128 cohort, never the teacher corpus."""
    from .scene_development_grid import SAFE
    from .scene_natural_prompt import checked_revision
    report=json.loads((data/'report.json').read_text())
    if not report['complete'] or not report['role_disjoint'] or report['TEST_opened']:
        raise ValueError('Completed role-disjoint paired DEV source required')
    original=json.loads((data/'inference_dev_inputs.json').read_text())
    if not all(set(r)==SAFE for r in original):raise ValueError('Non-public inference inputs')
    rows,protocol=checked_revision(revision,data,{'pair_sha256':report['pairs_sha256']},original)
    if len(rows)!=128:raise ValueError('All128 natural DEV requests required')
    return rows,protocol


def matched_visible_inputs(data):
    """Public visible-anchor DEV48 only; never construction hints or targets."""
    from .scene_instruction_render import checked_revision
    from .scene_instruction_data import SAFE
    report=json.loads((data/'report.json').read_text(encoding='utf8'))
    original=json.loads((data/'inference_dev_inputs.json').read_text(encoding='utf8'))
    if not all(set(r)==set(SAFE)for r in original):raise ValueError('Non-public DEV fields')
    rows,protocol=checked_revision(data,data,{'pair_sha256':report['pairs_sha256']},original)
    if len(rows)!=48 or len({r['reference_id']for r in rows})!=16:raise ValueError('Full48/16 visible-anchor DEV required')
    protocol.update(peer_has_no_task_tuning=True,paired_model_not_loaded=True)
    return rows,protocol


def checked_assets(root,metadata):
    meta=json.loads(metadata.read_text(encoding='utf8'))
    assert meta['sha']==REVISION and meta['id']=='Shitao/OmniGen-v1-diffusers'
    checked={}
    for row in meta['siblings']:
        path=root/row['rfilename'];assert path.is_file() and path.stat().st_size==row['size'],row['rfilename']
        if 'lfs' in row:
            value=digest(path);assert value==row['lfs']['sha256'],row['rfilename']
        else:
            data=path.read_bytes();value=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert value==row['blobId'],row['rfilename']
        checked[row['rfilename']]={'bytes':row['size'],'verified_official_digest':value}
    return checked


def inputs_for(sources,corpus=False):
    rows=[]
    if corpus:
        for i,source in enumerate(sources):
            assert source['split'] in ['train','dev']
            for condition in conditions(source['category'],SHORT_PROFILE):
                scene=int(condition['condition_id'])
                if source['split']=='dev' and scene not in [0,1]:continue
                if source['split']=='train' and scene%4!=i%4:continue
                for index in ([0,1] if source['split']=='train' else [2]):
                    rows.append({**condition,'sample_id':source['sample_id']+'-scene'+str(scene)+'-draw'+str(index),
                        'reference_id':source['sample_id'],'reference':'references/'+source['sample_id']+'.png',
                        'reference_sha256':source['white_reference_sha256'],'split':source['split'],
                        'seed_index':index,'seed':request_seed(source['white_reference_sha256'],index),
                        'source_product_id':source['product_id'],'source_RGBA_sha256':source['rgba_file_sha256']})
        return rows
    for source in sources:
        if source['split']!='dev':continue
        for condition in conditions(source['category'],COARSE_PROFILE)[:2]:
            rows.append({**condition,'sample_id':source['sample_id']+'-scene'+condition['condition_id']+'-draw2',
                'reference_id':source['sample_id'],'reference':source['white_reference'],
                'reference_sha256':source['white_reference_sha256'],'split':'dev',
                'seed':request_seed(source['white_reference_sha256'],2)})
    return rows


def generate(a):
    import torch
    from diffusers import OmniGenPipeline
    visible=getattr(a,'visible_anchor_dev',False)
    if visible and(getattr(a,'prompt_revision',None)or a.corpus):raise ValueError('Visible DEV cannot mix natural revision or TRAIN corpus')
    matched=getattr(a,'prompt_revision',None)or(a.sources if visible else None)
    if matched and a.corpus:raise ValueError('Matched DEV must not generate TRAIN candidates')
    protocol=None
    if matched:
        rows,protocol=matched_visible_inputs(a.sources)if visible else matched_natural_inputs(a.sources,matched)
    else:
        report=json.loads((a.sources/'report.json').read_text(encoding='utf8'))
        assert report['complete'] and not report['TEST_opened']
    assets=checked_assets(a.model_path,a.metadata)
    write(a.output/'model_assets.json',{'official_revision':REVISION,'files':assets})
    sources=[] if matched else json.loads((a.sources/'sources.json').read_text(encoding='utf8'))
    corpus=getattr(a,'corpus',False)
    if not matched:rows=inputs_for(sources,corpus)
    if corpus:
        assert len(rows)==336 and sum(r['split']=='train' for r in rows)==304
        (a.output/'references').mkdir();(a.output/'targets').mkdir()
    else:assert len(rows)==(48 if visible else(128 if matched else 32)) and len({r['reference_id'] for r in rows})==16
    source_map={s['sample_id']:s for s in sources}
    write(a.output/'requests.json',rows)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    pipe=OmniGenPipeline.from_pretrained(a.model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True);pipe.enable_vae_slicing()
    for component in pipe.components.values():
        if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
    records=[]
    for i,row in enumerate(rows):
        source=a.sources/(row['reference'] if matched else source_map[row['reference_id']]['white_reference'])
        assert digest(source)==row['reference_sha256']
        reference_path=a.output/(row['reference'] if corpus else row['reference_id']+'.png')
        if not reference_path.exists():shutil.copyfile(source,reference_path)
        with Image.open(source) as im:reference=im.convert('RGB').copy()
        reference,framing=frame_reference(reference,getattr(a,'reference_framing','original'))
        generator=torch.Generator(device='cuda').manual_seed(row['seed'])
        noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.bfloat16,generator=generator)
        noise_sha=hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
        prompt='<img><|image_1|></img>\n'+row['prompt']
        with torch.inference_mode():image=pipe(prompt=prompt,input_images=[reference],height=512,width=512,
            num_inference_steps=50,max_input_image_size=256,guidance_scale=2.5,img_guidance_scale=1.6,
            use_input_image_size_as_output=False,generator=generator,latents=noise).images[0]
        assert image.size==(512,512)
        filename=('targets/' if corpus else '')+row['sample_id']+'.png';image.save(a.output/filename)
        records.append({**row,'split':row.get('split','dev'),'native_prompt':prompt,'generated_PNG':filename,'generated_sha256':digest(a.output/filename),
            'initial_latent_sha256':noise_sha,
            'resolution':512,'sampling_steps':50,'model':'OmniGen-v1 official frozen','input_sha256':digest(source),'reference_framing':framing})
        if corpus:records[-1].update(target=filename,target_sha256=digest(a.output/filename),QA_accepted=False,
            teacher='frozen official OmniGen-v1',RGB_paste_or_target_inputs=False)
        write(a.output/('candidates.json' if corpus else 'images.json'),records)
        write(a.output/'progress.json',{'stage':'TRAIN_DEV_whole_scene_candidates' if corpus else 'raw_frozen_OmniGen_DEV','done':i+1,'total':len(rows)})
    parameters=sum(p.numel() for p in {id(p):p for component in pipe.components.values()
        if isinstance(component,torch.nn.Module) for p in component.parameters()}.values())
    final={'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],'scope':'TRAIN_DEV_candidates' if corpus else ('DEV_expanded' if matched else 'DEV'),
        'generated_count':len(rows),'resolution':512,'reference_max_side':256,'sampling_steps':50,
        'parameters':parameters,'prompt_profile':protocol['prompt_profile'] if matched else (SHORT_PROFILE if corpus else COARSE_PROFILE),'training_performed':False,'quality_accepted':False,
        'native_official_guidance':{'text':2.5,'image':1.6},'official_weights_revision':REVISION,
        'GT_masks_or_targets_opened':False,'independent_TEST_opened':False,'image_feedback':False,
        'native_peer_reference_VAE_not_a_Qwen_branch':True,'factual_evaluation_captions':True,
        'different_official_sampling_budget_from_SDXL30_must_be_labelled':True}
    final['reference_framing']=getattr(a,'reference_framing','original')
    if matched:final.update(expanded_DEV_protocol=protocol,matched_public_prompt_unmodified=True,
        only_native_image_placeholder_added=True,source_request_sha256=protocol['request_sha256'],
        no_task_tuning=True,not_same_training_budget_as_paired_models=True)
    if corpus:final.update(pilot_only=False,candidate_count=len(records),candidate_manifest_sha256=digest(a.output/'candidates.json'),
        TEST_opened=False,all_candidates_require_acceptance=True,eligible_for_training=False,
        backgrounds_are_AI_generated_not_external_room_photographs=True,no_foreground_compositing=True)
    write(a.output/'report.json',final)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',action='store_true',help='New two-sentence TRAIN/DEV candidate corpus; never auto-accepted')
    parser.add_argument('--prompt-revision',type=Path,help='Matched natural DEV128, using --sources as paired-data root; not TRAIN corpus')
    parser.add_argument('--visible-anchor-dev',action='store_true',help='Exact target-free visible-anchor DEV48 on all16 original views; no TRAIN hints')
    parser.add_argument('--reference-framing',choices=['original',FRAMING],default='original',help='Source-RGB-only white-margin framing ablation, no GT/mask')
    for key in ['sources','model-path','metadata','output']:parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists() or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:
        raise ValueError('New run/published source/GPU6 required')
    a.output.mkdir(parents=True)
    write(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ['T12_SOURCE_COMMIT'],
        'source_sha256':digest(Path(__file__))})
    try:generate(a)
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
