"""Read-only TRAIN audit of the conditioning used by E30 and original F9.

This does not establish a cause of sampling errors or approve a model. It opens
only already bound TRAIN features and stored TRAIN trajectory identities; no
generator, GPU, DEV/TEST image, VAE or optimizer is used.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from .qwen3_scene_conditioning import QwenSDXLConditions
from .scene_tail_distill import complete_state,load_cache
from .white_reference_scene_probe import digest,write


def compare(actual,wanted):
    if actual.shape!=wanted.shape or not torch.isfinite(actual).all() or not torch.isfinite(wanted).all():
        raise ValueError('Same finite condition shape required')
    a,b=actual.float().flatten(),wanted.float().flatten()
    return dict(relative_RMSE=float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8)),
        cosine=float(F.cosine_similarity(a[None],b[None]).item()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['electronic','LG','electronic-cache','LG-cache','teacher-paths','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);a.output.mkdir();e,er=complete_state(a.electronic);l,lr=complete_state(a.LG)
    assert er['best_sha256']=='662ff90f9a46670b28bf86d61629c59e4fb619ed69a62e4ae52ed8527ea76f76'
    assert lr['best_sha256']=='f9fbdb7fd36d1c862686616a624e7c18f4e9149cef61b0704bbf97767de6eaec'
    ec,ecr=load_cache(a.electronic_cache);lc,lcr=load_cache(a.LG_cache)
    assert e['pair_sha256']==l['pair_sha256']==ec['pair_sha256']==lc['pair_sha256']
    r=json.loads((a.teacher_paths/'report.json').read_text());path=a.teacher_paths/'TRAIN_full_teacher_paths.pt'
    assert r['complete'] and digest(path)==r['full_TRAIN_teacher_paths_sha256']
    cohort=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    keys=sorted(cohort['paths']);assert cohort['all_TRAIN'] and len(keys)==192
    rows={x['input_key']:x for x in lc['records'] if x['split']=='train'}
    erows={x['input_key']:x for x in ec['records'] if x['split']=='train'}
    assert set(keys)<=set(rows)&set(erows)
    heads=[]
    for state in [e,l]:
        h=QwenSDXLConditions(joint_image_context=state.get('joint_image_context',False)).eval().requires_grad_(False)
        h.load_state_dict(state['head'],strict=True);heads.append(h)
    values=[]
    with torch.inference_mode():
        for key in keys:
            outputs=[]
            for h,cache in zip(heads,[ec,lc]):
                f=cache['features'][key];outputs.append(h(f['hidden'][None].float(),f['visual'][None].float()))
            item=dict(input_key=key,category=rows[key]['category'],conditions={})
            for name in ['text','pooled','image']:
                own=compare(outputs[0][name],outputs[0][name]);assert own['relative_RMSE']==0 and abs(own['cosine']-1)<1e-5
                item['conditions'][name]=compare(outputs[1][name],outputs[0][name])
            values.append(item)
    summary={}
    for category in ['all','lamp','pillow']:
        subset=[v for v in values if category=='all' or v['category']==category]
        summary[category]={name:{metric:dict(mean=sum(v['conditions'][name][metric]for v in subset)/len(subset),min=min(v['conditions'][name][metric]for v in subset),max=max(v['conditions'][name][metric]for v in subset))for metric in ['relative_RMSE','cosine']}for name in ['text','pooled','image']}
    write(a.output/'per_TRAIN_context.json',values)
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__)),argv=sys.argv,
        teacher_E30_sha256=er['best_sha256'],original_F9_sha256=lr['best_sha256'],electronic_cache_sha256=ecr['cache_sha256'],LG_cache_sha256=lcr['cache_sha256'],
        teacher_path_sha256=digest(path),TRAIN_contexts=192,summary=summary,CPU_only=True,model_or_cache_modified=False,DEV_or_TEST_opened=False,
        conclusion_scope='Condition mismatch diagnostic only; does not isolate a cause or prove that a different TRAIN teacher would improve the student'))


if __name__=='__main__':main()
