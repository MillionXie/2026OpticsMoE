"""Bind an explicit inference-only reference-strength ablation; never train weights."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import torch
from .scene_tail_distill import complete_state
from .white_reference_scene_probe import digest,write


def bind(state, strength, parent_sha, source):
    if not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError('Require a positive finite reference strength <=1')
    if state.get('inference_reference_strength_ablation'):
        raise ValueError('Ablations must start from the original trained parent')
    sampling=state.get('sampling_contract')
    if not sampling or sampling.get('kind')!='progressive_guided_Euler_CFGpreserved_v1':
        raise ValueError('Only the explicitly trained guided Euler tail is supported')
    if state.get('optical_frontend',{}).get('pair_expansion'):
        raise ValueError('This ablation is restricted to the original single-EG frontend')
    result=dict(state)
    result['training_sampling_contract']=copy.deepcopy(sampling)
    result['sampling_contract']=dict(sampling,IP_scale=strength)
    result['source_commit']=source
    result['inference_reference_strength_ablation']=dict(parent_checkpoint_sha256=parent_sha,
        original_IP_scale=sampling['IP_scale'],inference_IP_scale=strength,
        training_performed=False,weights_unchanged=True,TEST_opened=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--strength',type=float,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists() and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    state,report=complete_state(a.parent)
    result=bind(state,a.strength,report['best_sha256'],os.environ['T12_SOURCE_COMMIT'])
    for name in ['head','unet_lora']:
        assert set(result[name])==set(state[name])
        assert all(torch.equal(v,state[name][n]) for n,v in result[name].items())
    assert result['optical_frontend']==state['optical_frontend']
    a.output.mkdir(parents=True)
    torch.save(result,a.output/'best_checkpoint.pt');shutil.copyfile(a.output/'best_checkpoint.pt',a.output/'last_checkpoint.pt')
    write(a.output/'report.json',dict(complete=True,stage='explicit_inference_reference_strength_binding',
        source_commit=os.environ['T12_SOURCE_COMMIT'],best_sha256=digest(a.output/'best_checkpoint.pt'),
        **result['inference_reference_strength_ablation'],training_sampling_contract=result['training_sampling_contract'],
        sampling_contract=result['sampling_contract'],extra_deployment_parameters=0,
        head_and_UNet_tensors_exactly_unchanged=True,quality_or_speed_accepted=False))


if __name__=='__main__':main()
