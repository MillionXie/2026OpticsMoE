"""Admission for decoder recovery after the two explicit condition corrections."""


def checked_freeze(profile,parent,warmup,variant):
    enabled=profile.get('freeze_corrected_condition_head',False)
    if not enabled:return False
    if variant!='qwen'or parent is None or warmup!=0 or parent.get('optical_frontend'):
        raise ValueError('Fixed condition recovery needs completed electronic Qwen parent and no head warmup')
    if not parent.get('condition_correction')or not parent.get('image_condition_correction'):
        raise ValueError('Actual token/source correction lineage required')
    if not parent['condition_correction']['UNet_LoRA_unchanged']or not parent['image_condition_correction']['text_pooled_projection_unchanged']:
        raise ValueError('Expected isolated TRAIN correction protocol')
    return True
