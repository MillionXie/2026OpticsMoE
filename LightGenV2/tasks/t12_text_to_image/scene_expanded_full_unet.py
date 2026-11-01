"""Restore all70 original UNet blocks without changing any saved tensor bytes.

This is an explicit user-requested capacity restoration, not a new trained result.
The64-block parent already retains the inactive original LoRA entries. Refuse a
partial namespace rather than fabricate weights for the six restored blocks.
"""
import argparse,copy,json,os,re,shutil,sys
from pathlib import Path
from .scene_depth_recovery import DEEP
from .scene_expanded_middle import checked_profile
from .scene_instruction_data import sha,write


def restored_state(state,parent):
    old=checked_profile(state['depth_compression']['profile'])
    if(old['schema']!='scene_expanded_middle_v1' or old['UNet_deep_keep']!=9
       or state['sampling_contract']['steps']!=8 or state['sampling_contract']['CFG']!=7.5
       or state.get('joint_image_context',False) or not state.get('no_validation_test_PT_selection')
       or not parent['compression_training_performed']):
        raise ValueError('Exact trained64-block8-step expanded parent required')
    groups={name:set()for name in DEEP}
    for name in state['unet_lora']:
        match=re.match(r'(.+\.transformer_blocks)\.(\d+)\.',name)
        if match and match.group(1)in groups:groups[match.group(1)].add(int(match.group(2)))
    if any(ids!=set(range(10))for ids in groups.values()):
        raise ValueError('All original six by ten deep-block LoRA namespaces required')
    profile=dict(old,schema='scene_expanded_full_UNet8_v1',UNet_deep_keep=10)
    checked_profile(profile)
    result=copy.deepcopy(state)
    result['depth_compression']=dict(state['depth_compression'],profile=profile,
        parent_checkpoint_sha256=parent['best_sha256'],parent_head_unchanged=True,
        restored_full_UNet_blocks=70,UNet_pruning_performed=False)
    result['architecture_restoration']=dict(source_checkpoint_sha256=parent['best_sha256'],
        old_UNet_blocks=64,actual_UNet_blocks=70,six_original_blocks_reenabled=True,
        all_original_LoRA_tensor_bytes_preserved=True,head_optical_sampling_unchanged=True,
        new_training_performed=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);a=p.parse_args()
    if a.output.exists()or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('New published CPU-only restoration required')
    import torch
    from .scene_tail_distill import complete_state
    state,parent=complete_state(a.parent)
    if parent['best_sha256']!=a.parent_sha256:raise ValueError('Changed source PT')
    result=restored_state(state,parent)
    for field in ['head','unet_lora']:
        assert set(result[field])==set(state[field])and all(torch.equal(v,state[field][n])for n,v in result[field].items())
    assert result['optical_frontend']==state['optical_frontend']and result['sampling_contract']==state['sampling_contract']
    result['source_commit']=os.environ['T12_SOURCE_COMMIT']
    a.output.mkdir();torch.save(result,a.output/'best_checkpoint.pt');shutil.copyfile(a.output/'best_checkpoint.pt',a.output/'last_checkpoint.pt')
    report=dict(parent,complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],best_sha256=sha(a.output/'best_checkpoint.pt'),last_sha256=sha(a.output/'last_checkpoint.pt'),
        depth_compression=result['depth_compression'],sampling_steps=8,training_performed=False,
        architecture_restoration=result['architecture_restoration'],actual_UNet_blocks=70,UNet_pruning_performed=False,
        source_checkpoint_sha256=parent['best_sha256'],all_head_and_LoRA_tensors_exact=True,
        quality_or_speed_accepted=False,final_TEST_PT_selection_pending=True,TEST_PT_selection_performed=False,
        selected_from_completed_best_last=False,selection_metrics_sha256=None,
        inherited_training_source_commit=parent['source_commit'],no_new_training_yet=True,CPU_only=True)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__)))
    write(a.output/'full_UNet_profile.json',result['depth_compression']['profile']);write(a.output/'report.json',report)


if __name__=='__main__':main()
