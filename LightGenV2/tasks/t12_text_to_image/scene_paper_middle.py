"""The same64-block8-step single-EG graph, bound to actual new v7 training."""
from .scene_R2_middle_protocol import F6
from .scene_paper_prompt_protocol import PROFILE


def checked_profile(c):
    keys={'schema','prompt_profile','vision_keep','language_keep','UNet_deep_keep','student_steps','parent_generator_sha256','parent_frontend_sha256','electronic_teacher_sha256','pairs_sha256','TRAIN_contexts'}
    if set(c)!=keys or c['schema']!='scene_paper_middle_v1'or c['prompt_profile']!=PROFILE:raise ValueError('Explicit reviewed v7 middle profile required')
    if (c['vision_keep'],c['language_keep'],c['UNet_deep_keep'],c['student_steps'])!=(6,4,9,8)or not isinstance(c['TRAIN_contexts'],int)or not 1<=c['TRAIN_contexts']<=132 or c['parent_frontend_sha256']!=F6:raise ValueError('Keep one single-EG64-block8-step middle graph and actual TRAIN coverage')
    for name in ['parent_generator_sha256','electronic_teacher_sha256','pairs_sha256']:
        s=c[name]
        if len(s)!=64 or any(x not in'0123456789abcdef'for x in s):raise ValueError('Completed actual new training identities required')
    return dict(c)


def checked_training(c,state,parent,targets,guidance,adapt_head):
    checked_profile(c)
    if (state.get('pair_sha256')!=c['pairs_sha256']or state.get('joint_image_context',False)or(state.get('optical_frontend')or{}).get('checkpoint_sha256')!=F6 or parent.get('best_sha256')!=c['parent_generator_sha256']or parent.get('initial_paired_checkpoint_sha256')!=c['electronic_teacher_sha256']or parent.get('condition_teacher_sha256')!=c['electronic_teacher_sha256']):raise ValueError('Actual same-new-task electronic-to-LG paired lineage required')
    if (targets.get('teacher_sha256')!=c['electronic_teacher_sha256']or targets.get('TRAIN_contexts')!=c['TRAIN_contexts']or targets.get('student_steps')!=8 or targets.get('teacher_steps')!=30 or targets.get('DEV_or_TEST_targets_created')is not False or not targets.get('teacher_is_full_electronic_Qwen')or not(targets.get('teacher_task_LoRA_fusion')or{}).get('task_LoRA_fused')):raise ValueError('Only the actual new fused electronic30-step TRAIN paths')
    if guidance!=7.5 or adapt_head:raise ValueError('Preserve CFG and the newly recovered head during few-step distillation')
    return True


def main():
    from .scene_instruction_middle import main as run
    from .scene_paper_paired_render import checked_revision
    run(checked_profile,checked_training,checked_revision)


if __name__=='__main__':main()
