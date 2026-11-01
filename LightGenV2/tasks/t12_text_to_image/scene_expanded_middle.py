"""The same single-EG64-block8-step middle graph on expanded TRAIN/TEST-selection."""
import json
import sys
from pathlib import Path
from .scene_R2_middle_protocol import F6
from .scene_paper_expanded_data import PROFILE


def checked_profile(c):
    keys={'schema','prompt_profile','vision_keep','language_keep','UNet_deep_keep','student_steps',
          'parent_generator_sha256','parent_frontend_sha256','electronic_teacher_sha256','pairs_sha256','TRAIN_contexts'}
    if(set(c)!=keys or (c['schema'],c['UNet_deep_keep'])not in [('scene_expanded_middle_v1',9),('scene_expanded_full_UNet8_v1',10)] or c['prompt_profile']!=PROFILE
       or (c['vision_keep'],c['language_keep'],c['student_steps'])!=(6,4,8)
       or c['parent_frontend_sha256']!=F6 or type(c['TRAIN_contexts'])is not int or not 1<=c['TRAIN_contexts']<=1520):
        raise ValueError('One original middle graph and actual expanded TRAIN coverage required')
    for key in ['parent_generator_sha256','electronic_teacher_sha256','pairs_sha256']:
        if len(c[key])!=64 or any(x not in '0123456789abcdef' for x in c[key]):raise ValueError('Actual completed weight/data SHA required')
    return dict(c)


def checked_training(c,state,parent,targets,guidance,adapt_head):
    checked_profile(c)
    if(state['pair_sha256']!=c['pairs_sha256'] or not state.get('no_validation_test_PT_selection')
       or state.get('joint_image_context',False) or (state.get('optical_frontend')or{}).get('checkpoint_sha256')!=F6
       or parent['best_sha256']!=c['parent_generator_sha256'] or parent['initial_paired_checkpoint_sha256']!=c['electronic_teacher_sha256']
       or parent['condition_teacher_sha256']!=c['electronic_teacher_sha256']):raise ValueError('Expanded electronic-to-LG lineage changed')
    if(targets['teacher_sha256']!=c['electronic_teacher_sha256'] or targets['TRAIN_contexts']!=c['TRAIN_contexts']
       or targets['student_steps']!=8 or targets['teacher_steps']!=30 or targets['DEV_or_TEST_targets_created']
       or not targets['teacher_is_full_electronic_Qwen'] or not targets['teacher_task_LoRA_fusion']['task_LoRA_fused']):
        raise ValueError('Same actual selected E30 TRAIN paths only')
    if guidance!=7.5 or adapt_head:raise ValueError('Preserve CFG and recovered head during few-step KD')


def main():
    from .scene_instruction_middle import main as run
    from .scene_expanded_selection_render import checked_revision,public_report,public_inputs
    from . import scene_paired_render as renderer
    mode=sys.argv[sys.argv.index('--middle-mode')+1]
    old_inputs=renderer.inference_inputs;renderer.inference_inputs=public_inputs
    try:run(checked_profile,checked_training,checked_revision,expected_contexts=500)
    finally:renderer.inference_inputs=old_inputs
    if mode!='train':
        output=Path(sys.argv[sys.argv.index('--output')+1]);public_report(output)


if __name__=='__main__':main()
