"""SHA-bound admission for the one user-selected R2 middle LG candidate."""
F6='6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e'
ELECTRONIC='a5e7be463fe434ee99048e6f29ed6847f7f87beda7f69e94a008807608aed14f'
PAIRS='536dac05bb15d7226ad79e7573c886f1af39982c75d9beaa7197b3102f1d3970'


def checked_profile(c):
    required={'schema','vision_keep','language_keep','UNet_deep_keep','student_steps','parent_generator_sha256',
              'parent_frontend_sha256','electronic_teacher_sha256','pairs_sha256','TRAIN_contexts'}
    if set(c)!=required or c['schema']!='scene_R2_middle_v1':raise ValueError('Explicit R2 middle profile required')
    if(c['vision_keep'],c['language_keep'],c['UNet_deep_keep'],c['student_steps'],c['TRAIN_contexts'])!=(6,4,9,8,132):
        raise ValueError('Only the selected single-EG 64-block8-step R2 candidate')
    if(c['parent_frontend_sha256'],c['electronic_teacher_sha256'],c['pairs_sha256'])!=(F6,ELECTRONIC,PAIRS):
        raise ValueError('Do not reuse old-task teachers or targets')
    s=c['parent_generator_sha256']
    if len(s)!=64 or any(x not in'0123456789abcdef'for x in s):raise ValueError('Actual recovered parent SHA required')
    return dict(c)


def checked_training(c,state,parent,targets,guidance,adapt_head):
    checked_profile(c)
    if(parent.get('best_sha256')!=c['parent_generator_sha256'] or state.get('pair_sha256')!=PAIRS
       or(state.get('optical_frontend')or{}).get('checkpoint_sha256')!=F6
       or parent.get('initial_paired_checkpoint_sha256')!=ELECTRONIC
       or parent.get('condition_teacher_sha256')!=ELECTRONIC or state.get('joint_image_context',False)):
        raise ValueError('Actual same-task electronic-to-LG recovery lineage required')
    if(targets.get('teacher_sha256')!=ELECTRONIC or targets.get('TRAIN_contexts')!=132
       or targets.get('student_steps')!=8 or targets.get('teacher_steps')!=30
       or targets.get('DEV_or_TEST_targets_created')is not False or not targets.get('teacher_is_full_electronic_Qwen')
       or not(targets.get('teacher_task_LoRA_fusion')or{}).get('task_LoRA_fused')):
        raise ValueError('Only complete same-task fused E30 TRAIN paths')
    if guidance!=7.5 or adapt_head:raise ValueError('Preserve CFG and freeze the recovered existing head')
    return True
