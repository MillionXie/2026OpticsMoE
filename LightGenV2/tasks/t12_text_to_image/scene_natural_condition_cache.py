"""Fresh actual Qwen TRAIN features for natural prompts, without scene GT tensors."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
from PIL import Image
from .scene_natural_prompt import revise,PROFILE
from .scene_tail_distill import complete_state
from .qwen3_scene_conditioning import QwenSDXLConditions
from .qwen3_scene_paired_train import new_qwen,raw_scene_features
from .scene_paired_render import install_optical_frontend
from .white_reference_scene_probe import GPU,digest,write


def requests(rows,data):
    selected={}
    for row in rows:
        if row['split']!='train':continue
        q=revise(row)
        # Supervision metadata, target filenames, masks and SKU captions are
        # intentionally omitted from this input-only condition cache.
        safe={k:q[k] for k in ['reference','reference_id','reference_sha256','prompt','native_caption',
            'reward_caption','prompt_profile','condition_id','category','split']}
        path=data/safe['reference']
        if digest(path)!=safe['reference_sha256']:raise ValueError('Changed TRAIN input pixels')
        key=hashlib.sha256(path.read_bytes()+b'\0'+safe['prompt'].encode()+b'\0Qwen256').hexdigest()
        safe['input_key']=key
        if key in selected and safe!=selected[key]:raise ValueError('Conflicting input-only metadata')
        selected[key]=safe
    if not selected or {q['category'] for q in selected.values()}!={'lamp','pillow'}:
        raise ValueError('Both TRAIN categories required')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','teacher','source-cache','qwen','lg-frontend','lg-features','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    assert len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);a.qwen_side=256
    teacher,teacher_report=complete_state(a.teacher);source=json.loads((a.data/'report.json').read_text())
    assert source['complete'] and source['role_disjoint'] and source['pairs_sha256']==teacher['pair_sha256']==digest(a.data/'pairs.json')
    original=json.loads((a.data/'pairs.json').read_text());selected=requests(original,a.data)
    assert len(selected)==144 and len({q['reference_sha256'] for q in selected.values()})==76
    dev=json.loads((a.data/'inference_dev_inputs.json').read_text())
    assert not {q['reference_sha256'] for q in selected.values()}&{q['reference_sha256'] for q in dev}
    a.output.mkdir(parents=True);write(a.output/'TRAIN_input_manifest.json',list(selected.values()))
    revision=dict(prompt_profile=PROFILE,TRAIN_input_manifest_sha256=digest(a.output/'TRAIN_input_manifest.json'),
        paired_label_source_sha256=teacher['pair_sha256'],reference_RGB_still_input=True,
        input_only_new_actual_Qwen_features=True,DEV_or_TEST_forwarded=False)
    qwen,processor=new_qwen(a)
    assert len(qwen.model.visual.blocks)==24 and len(qwen.model.language_model.layers)==28
    head=QwenSDXLConditions(joint_image_context=teacher.get('joint_image_context',False)).cuda().eval().requires_grad_(False)
    head.load_state_dict(teacher['head'],strict=True)
    e_features={};conditions={}
    for i,(key,row) in enumerate(selected.items()):
        with Image.open(a.data/row['reference']) as image:
            h,v,ids=raw_scene_features(qwen,processor,image.convert('RGB'),row['prompt'],256,keep_on_device=True)
        with torch.inference_mode():out=head(h.unsqueeze(0).float(),v.unsqueeze(0).float())
        conditions[key]={n:value[0].bfloat16().cpu() for n,value in out.items()}
        e_features[key]=dict(hidden=h.cpu(),visual=v.cpu(),input_ids=ids.cpu())
        if (i+1)%16==0:write(a.output/'progress.json',dict(stage='actual_full_electronic_Qwen_TRAIN_natural',done=i+1,total=len(selected)))
    e_payload=dict(records=list(selected.values()),features=e_features,pair_sha256=teacher['pair_sha256'],
        input_prompt_revision=revision)
    optical=install_optical_frontend(qwen,a.lg_frontend,a.lg_features,a.source_cache)
    assert optical['retained_original_TF']=={'vision':6,'language':4} and optical['pair_expansion'] is None
    lg_features={}
    for i,(key,row) in enumerate(selected.items()):
        with Image.open(a.data/row['reference']) as image:
            h,v,ids=raw_scene_features(qwen,processor,image.convert('RGB'),row['prompt'],256)
        assert torch.equal(ids,e_features[key]['input_ids'])
        lg_features[key]=dict(hidden=h,visual=v,input_ids=ids)
        if (i+1)%16==0:write(a.output/'progress.json',dict(stage='actual_single_EG_Qwen_TRAIN_natural',done=i+1,total=len(selected)))
    lg_payload=dict(e_payload,features=lg_features,optical_frontend=optical,
        TRAIN_electronic_conditions=conditions,condition_teacher_sha256=teacher_report['best_sha256'])
    outputs={}
    for name,payload in [('electronic',e_payload),('LG',lg_payload)]:
        folder=a.output/name;folder.mkdir();torch.save(payload,folder/'paired_cache.pt')
        shutil.copyfile(a.source_cache/'model_assets.json',folder/'model_assets.json')
        report=dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],cache_sha256=digest(folder/'paired_cache.pt'),
            prompt_profile=PROFILE,TRAIN_only=True,n=len(selected),Qwen_input_side=256,target_resolution=512,
            scene_GT_images_latents_or_masks_opened=False,old_feature_tensors_reused=False,
            source_paired_label_manifest_sha256=teacher['pair_sha256'],input_prompt_revision=revision,
            actual_full_Qwen_or_single_EG_raw_RGB_text_forward=True,TEST_opened=False)
        if name=='LG':report.update(optical_frontend=optical,condition_teacher_sha256=teacher_report['best_sha256'])
        write(folder/'report.json',report);outputs[name]=report
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],TRAIN_input_keys=len(selected),
        TRAIN_unique_references=76,prompt_profile=PROFILE,reference_RGB_still_input=True,outputs=outputs,
        added_deployment_parameters=0,scene_GT_or_mask_or_latent_opened=False,DEV_or_TEST_forwarded=False,
        cache_is_training_optimization_not_deployment=True))
    del qwen,head;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
