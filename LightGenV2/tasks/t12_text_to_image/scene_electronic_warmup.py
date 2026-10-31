"""Explicit admission of same-task electronic conditions for LG head warmup."""


def checked_target(profile,payload,parent_sha,variant):
    target=profile.get('condition_warmup_target','native_encoders')
    if target=='native_encoders':return target
    if target!='bound_electronic':raise ValueError('Unknown condition warmup target')
    if variant!='qwen' or not payload.get('optical_frontend') or not parent_sha:
        raise ValueError('Bound electronic warmup requires actual LG features and electronic parent')
    if payload.get('condition_teacher_sha256')!=parent_sha:
        raise ValueError('Condition teacher must be the exact initialized electronic generator')
    train={r['input_key'] for r in payload['records'] if r['split']=='train'}
    dev={r['input_key'] for r in payload['records'] if r['split']=='dev'}
    bank=payload.get('TRAIN_electronic_conditions',{})
    if not train or train&dev or set(bank)!=train:
        raise ValueError('Exact disjoint TRAIN-only condition targets required')
    if any(set(v)!={'text','image','pooled'} for v in bank.values()):
        raise ValueError('Complete electronic conditions required')
    if profile.get('joint_image_context',False):
        raise ValueError('This recovery preserves the source-only image head of its electronic parent')
    return target
