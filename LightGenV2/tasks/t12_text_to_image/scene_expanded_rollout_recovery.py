"""Recover the same expanded 64-block/8-step LG through all internal steps.

Uses only the completed selected electronic TRAIN paths, preserving the actual
scaled noise, frozen head/Qwen/phases/VAE and existing UNet LoRA dimensions.
TEST remains selection-only, never a recovery loss or input to this trainer.
"""
import argparse,json,sys
from pathlib import Path
from .scene_R2_middle_protocol import F6
from .scene_paper_expanded_data import PROFILE
from .scene_expanded_middle import checked_profile


def checked_contract(c,state,parent,cache,teacher,source):
    keys={'schema','middle_profile','warm_checkpoint_sha256','reuse_all_TRAIN_contexts'}
    if set(c)!=keys or c['schema']!='scene_expanded_cached_rollout_v1' or c['reuse_all_TRAIN_contexts']is not True:
        raise ValueError('Explicit same-architecture TRAIN-only rollout contract required')
    profile=checked_profile(c['middle_profile'])
    if (len(c['warm_checkpoint_sha256'])!=64 or any(x not in '0123456789abcdef'for x in c['warm_checkpoint_sha256'])
        or parent['best_sha256']!=c['warm_checkpoint_sha256'] or not parent['compression_training_performed']
        or state['depth_compression']['profile']!=profile or state['sampling_contract']['steps']!=8
        or state['sampling_contract']['CFG']!=7.5 or state.get('joint_image_context',False)
        or not state.get('no_validation_test_PT_selection')):
        raise ValueError('Exact completed expanded 64-block8-step warm model required')
    if (state['pair_sha256']!=profile['pairs_sha256'] or cache['pair_sha256']!=profile['pairs_sha256']
        or state['optical_frontend']['checkpoint_sha256']!=F6
        or cache['optical_frontend']['checkpoint_sha256']!=F6):
        raise ValueError('Actual paired data and frozen F6 identity changed')
    train={r['input_key']for r in cache['records']if r['split']=='train'}
    held={r['input_key']for r in cache['records']if r['split']!='train'}
    paths=set(teacher['paths'])
    if (not teacher['all_TRAIN'] or paths!=train or paths&held or len(paths)!=profile['TRAIN_contexts']
        or teacher['pair_sha256']!=profile['pairs_sha256'] or teacher['teacher_steps']!=30
        or teacher['teacher_sha256']!=profile['electronic_teacher_sha256']
        or source['teacher_sha256']!=profile['electronic_teacher_sha256'] or not source['complete']
        or source['TRAIN_contexts']!=len(paths) or not source['teacher_is_full_electronic_Qwen']
        or source['DEV_or_TEST_targets_created']):
        raise ValueError('Every actual TRAIN path from the same selected E30 is required; no held-out path allowed')
    return dict(TRAIN_contexts=len(paths),teacher_sha256=teacher['teacher_sha256'],
                warm_checkpoint_sha256=parent['best_sha256'],sampling_steps=8,prompt_profile=PROFILE)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--recovery-profile',type=Path,required=True)
    custom,remaining=parser.parse_known_args();contract=json.loads(custom.recovery_profile.read_text())
    from . import scene_cached_rollout_recovery as base
    from .white_reference_scene_probe import digest,write
    original=base.run
    def run(a):
        a.reuse_all_TRAIN_contexts=True
        write(a.output/'recovery_profile.json',contract)
        original(a,admission=lambda *values:checked_contract(contract,*values))
        report=json.loads((a.output/'report.json').read_text())
        report.update(expanded_full_rollout_recovery=True,recovery_profile_sha256=digest(custom.recovery_profile),
            wrapper_source_sha256=digest(Path(__file__)),no_validation_test_PT_selection=True,
            final_TEST_PT_selection_pending=a.stage=='train',epochs=None,actual_paired_updates=a.updates if a.stage=='train' else 0,
            warm_model_training_updates=4992,existing_head_unchanged=True,TEST_used_for_gradient=False,
            all_TRAIN_eligible_including_fixed_monitor=True)
        write(a.output/'report.json',report)
    base.run=run;sys.argv=[sys.argv[0],*remaining]
    try:base.main()
    finally:base.run=original


if __name__=='__main__':main()
