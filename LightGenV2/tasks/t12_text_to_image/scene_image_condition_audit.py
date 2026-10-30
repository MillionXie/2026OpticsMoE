"""CPU TRAIN-only image-prefix sensitivity, not a generation-quality result.

The shared LG head can read language into IP image queries. Measure this on its
bound one-pair cache, with exactly equal visual features across room prompts.
The frozen two-pair candidate reuses identical head weights; its actual visual
features are NOT claimed to be covered by this one-pair cache diagnostic.
"""
import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import traceback
import torch
from .qwen3_scene_conditioning import QwenSDXLConditions
from .scene_tail_distill import complete_state,load_cache
from .white_reference_scene_probe import digest,write


def image_only_forward(head,features):
    visual=head.project(features['visual'].unsqueeze(0).float())
    if head.joint_image_context:
        visual=torch.cat([visual,head.project(features['hidden'].unsqueeze(0).float())],dim=1)
    return head.image(visual)[0]


def relative_change(a,b):
    return float((a.float()-b.float()).square().mean().sqrt()/a.float().square().mean().sqrt().clamp_min(1e-8))


def run(a):
    state,ready=complete_state(a.generator);cache,report=load_cache(a.cache)
    assert state.get('joint_image_context') is True
    assert state['pair_sha256']==cache['pair_sha256']
    assert state['optical_frontend']['checkpoint_sha256']==cache['optical_frontend']['checkpoint_sha256']
    groups=defaultdict(dict);train={r['input_key'] for r in cache['records'] if r['split']=='train'}
    dev={r['input_key'] for r in cache['records'] if r['split']=='dev'}
    assert train and not train&dev and set(cache['TRAIN_electronic_conditions'])==train
    for row in cache['records']:
        if row['split']=='train':groups[row['reference_sha256']][row['input_key']]=row
    head=QwenSDXLConditions(joint_image_context=True).eval().requires_grad_(False)
    head.load_state_dict(state['head'],strict=True);counterfactual=copy.deepcopy(head);counterfactual.joint_image_context=False
    if a.shared_head_binding:
        other,identity=complete_state(a.shared_head_binding)
        assert all(torch.equal(v,other['head'][k]) for k,v in state['head'].items())
        shared_sha=identity['best_sha256']
    else:shared_sha=None
    rows=[];controls=[]
    with torch.inference_mode():
        for reference,items in groups.items():
            keys=sorted(items)
            if len(keys)<2:continue
            first=cache['features'][keys[0]];reference_target=cache['TRAIN_electronic_conditions'][keys[0]]['image']
            base=image_only_forward(head,first);plain=image_only_forward(counterfactual,first)
            if len(controls)<2:
                actual=head(first['hidden'].unsqueeze(0).float(),first['visual'].unsqueeze(0).float())['image']
                assert torch.equal(base,actual);controls.append(items[keys[0]]['sample_id'])
            for key in keys[1:]:
                f=cache['features'][key];target=cache['TRAIN_electronic_conditions'][key]['image']
                assert torch.equal(first['visual'],f['visual'])
                assert torch.equal(reference_target,target),'Electronic teacher image condition is not reference-only'
                changed=image_only_forward(head,f);plain_changed=image_only_forward(counterfactual,f)
                assert torch.equal(plain,plain_changed)
                rows.append(dict(reference_sha256=reference,first=items[keys[0]]['sample_id'],other=items[key]['sample_id'],
                    different_prompt=items[keys[0]]['prompt']!=items[key]['prompt'],visual_features_bit_identical=True,
                    current_joint_image_prefix_relative_RMS=relative_change(base,changed),
                    image_only_counterfactual_relative_RMS=relative_change(plain,plain_changed),
                    electronic_teacher_image_prefix_relative_RMS=relative_change(reference_target,target)))
    assert rows and all(r['different_prompt'] for r in rows)
    values=[r['current_joint_image_prefix_relative_RMS'] for r in rows]
    write(a.output/'comparisons.json',rows)
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],
        generator_sha256=ready['best_sha256'],cache_sha256=report['cache_sha256'],
        features_optical_frontend_sha256=state['optical_frontend']['checkpoint_sha256'],
        unchanged_head_in_candidate_binding_sha256=shared_sha,actual_two_pair_features_not_forwarded=True,
        TRAIN_reference_groups=len(groups),TRAIN_counterfactual_pairs=len(rows),
        current_joint_prefix_relative_RMS=dict(mean=statistics.mean(values),median=statistics.median(values),maximum=max(values)),
        electronic_teacher_prefix_change_exactly_zero=True,visual_only_counterfactual_prefix_change_exactly_zero=True,
        direct_image_path_vs_complete_head_exact_controls=controls,
        CPU_only=True,DEV_or_TEST_forwarded=False,scene_GT_or_latents_used=False,training_performed=False,
        causal_generation_failure_proved=False,visual_only_head_has_not_been_adapted_or_accepted=True,
        quality_or_speed_accepted=False,TEST_opened=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['generator','cache','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shared-head-binding',type=Path);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.environ.get('T12_SOURCE_COMMIT',''))==40 and not a.output.exists()
    a.output.mkdir(parents=True);torch.set_num_threads(2)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__)),CPU_only=True))
    try:run(a)
    except Exception:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
