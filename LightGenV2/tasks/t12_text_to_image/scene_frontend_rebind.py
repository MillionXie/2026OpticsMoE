"""Explicitly bind an unchanged generator to its TRAIN-recovered optical front.

This creates an evaluation candidate, not a claim of paired generator recovery.
The original generator, frontend, and their reports remain untouched.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
from .scene_tail_distill import complete_state
from .white_reference_scene_probe import digest,write


def binding_metadata(state,report,new_sha):
    old=state.get('optical_frontend')
    optical_only=report.get('only_existing_optical_macros_trainable') and report.get('head_UNet_VAE_retained_Qwen_frozen')
    with_native=(report.get('existing_native_readouts_also_trainable') and report.get('head_UNet_VAE_retained_Qwen_TF_frozen')
        and report.get('native_readout_parameters_trained',0)>0 and report.get('extra_deployment_parameters')==0)
    if not old or not report.get('complete') or not (optical_only or with_native):
        raise ValueError('A completed raw optical recovery and an LG generator are required')
    if report['parent_frontend_sha256']!=old['checkpoint_sha256']:raise ValueError('Frontend parent differs from generator binding')
    if report.get('TEST_opened') or report.get('DEV_or_TEST_forward_performed'):
        raise ValueError('Only a completed TRAIN-only frontend update can be rebound here')
    return dict(old,checkpoint_sha256=new_sha,feature_distillation_only_generator_recovery_not_yet_trained=False,
        TRAIN_coupled_optical_recovery=True,native_visual_readouts_adapted=bool(with_native),generator_not_updated_after_frontend_recovery=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['generator','frontend','old-frontend','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    state,ready=complete_state(a.generator);report=json.loads((a.frontend/'report.json').read_text());old_report=json.loads((a.old_frontend/'report.json').read_text())
    assert digest(a.frontend/'best_checkpoint.pt')==report['best_sha256'] and digest(a.old_frontend/'best_checkpoint.pt')==old_report['best_sha256']
    front=torch.load(a.frontend/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    old=torch.load(a.old_frontend/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert front['schema']==old['schema']=='qwen3_scene_optical_frontend_v1' and front['step']>0
    assert front['feature_cache_sha256']==old['feature_cache_sha256'] and set(front['macros'])==set(old['macros'])
    assert all(v.shape==old['macros'][n].shape for n,v in front['macros'].items())
    state=dict(state,optical_frontend=binding_metadata(state,report,report['best_sha256']),source_commit=os.environ['T12_SOURCE_COMMIT'])
    a.output.mkdir(parents=True);torch.save(state,a.output/'best_checkpoint.pt');torch.save(state,a.output/'last_checkpoint.pt')
    write(a.output/'execution.json',dict(argv=sys.argv,source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__)),pid=os.getpid()))
    write(a.output/'report.json',dict(complete=True,best_sha256=digest(a.output/'best_checkpoint.pt'),source_commit=os.environ['T12_SOURCE_COMMIT'],
        parent_generator_sha256=ready['best_sha256'],new_frontend_sha256=report['best_sha256'],old_frontend_sha256=old_report['best_sha256'],
        generator_parameters_unchanged=True,explicit_evaluation_binding_only=True,quality_or_speed_accepted=False,TEST_opened=False))


if __name__=='__main__':main()
