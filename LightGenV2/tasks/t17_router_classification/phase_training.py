"""Explicit phase-update policies; no change to the canonical inference model."""


def phase_update_mode(cfg):
    mode=cfg.get('phase_update_mode','mini_batch')
    if mode not in ('mini_batch','full_TRAIN'):raise ValueError('Unknown phase update mode')
    if mode=='full_TRAIN' and (cfg['architecture']!='LG' or cfg['phase_optimizer']!='wrapped' or not cfg.get('alternating_head')):
        raise ValueError('Full TRAIN phase accumulation requires wrapped LG and explicit head blocks')
    return mode


def phase_parameter_groups(model,phases,cfg):
    multiplier=cfg.get('router_lr_multiplier',1.)
    if multiplier<=0:raise ValueError('Router learning-rate multiplier must be positive')
    if multiplier==1.:return [dict(params=phases,lr=cfg['phase_lr'],lr_multiplier=1.)]
    router=getattr(model,'router_phase',None)
    if router is None or not any(p is router for p in phases):raise ValueError('Scaled router must be trainable')
    return [dict(params=[p for p in phases if p is not router],lr=cfg['phase_lr'],lr_multiplier=1.),
            dict(params=[router],lr=cfg['phase_lr']*multiplier,lr_multiplier=multiplier)]


def backward_training_loss(loss,batch_count,total_count,aggregate):
    if batch_count<=0 or total_count<batch_count:raise ValueError('Invalid training batch weights')
    (loss*(batch_count/total_count) if aggregate else loss).backward()
