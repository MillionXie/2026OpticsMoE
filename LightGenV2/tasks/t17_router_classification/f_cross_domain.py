"""Explicit single-source-domain training and two-domain evaluation for panel F."""
from pathlib import Path
import json
import numpy as np
import torch
from .train import write
from .original_f_teacher import training_view_codes
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def select_source_domain(raw,domain):
    if domain not in ('RGB','SAR'):raise ValueError('F source domain must be RGB or SAR')
    number=0 if domain=='RGB' else 1;selected={}
    for split,part in raw.items():
        mask=part['domains']==number if split in ('train','val') else np.ones(len(part['labels']),dtype=bool)
        selected[split]={k:v[mask].copy() for k,v in part.items()}
    return selected


def validate_domain_teacher_cache(raw,cfg,teacher_dir):
    if cfg.get('training_domain') not in ('RGB','SAR'):raise ValueError('A single source domain is required')
    status=json.loads((teacher_dir/'status.json').read_text());meta=json.loads((teacher_dir/'metadata.json').read_text())
    if status['state']!='complete' or meta['training_domain']!=cfg['training_domain']:
        raise ValueError('Incomplete or different-domain teacher')
    if meta['opposite_domain_used'] or status['opposite_domain_used']:raise ValueError('Teacher accessed opposite domain')
    if meta['original_data']!=cfg['data']:raise ValueError('Teacher dataset/partition differs')
    manifest=json.loads((teacher_dir/'cache_manifest.json').read_text())
    cache=teacher_dir/'teacher_logits_train.npz'
    if sha256(cache)!=manifest[cache.name]:raise ValueError('Teacher TRAIN cache SHA differs')
    if sha256(teacher_dir/'best_checkpoint.pt')!=status['checkpoint_sha256']:raise ValueError('Teacher PT SHA differs')
    with np.load(cache,allow_pickle=False) as z:
        for key in ('ids','labels','domains'):
            if not np.array_equal(z[key],raw['train'][key]):raise ValueError('Teacher TRAIN identity differs')
        logits=z['logits'].copy()
    if logits.shape!=(len(raw['train']['labels']),8,cfg['num_classes']) or not np.isfinite(logits).all():raise ValueError('Invalid teacher logits')
    return logits,meta,status


def prepare_source_teacher(raw,cfg,out):
    teacher_dir=Path(__file__).resolve().parent/cfg['teacher']['cache_run']
    logits,meta,status=validate_domain_teacher_cache(raw,cfg,teacher_dir)
    if training_view_codes(cfg['augmentation'])!=tuple(range(8)):raise ValueError('Source teacher requires all D4 views')
    np.savez_compressed(out/'TRAIN_teacher_logits.npz',logits=logits,ids=raw['train']['ids'],labels=raw['train']['labels'],domains=raw['train']['domains'])
    write(out/'teacher_manifest.json',dict(teacher_run=str(teacher_dir),checkpoint_sha256=status['checkpoint_sha256'],
        training_domain=cfg['training_domain'],teacher_used_original_TEST=True,teacher_TEST_domain=cfg['training_domain'],
        teacher_never_used_opposite_domain=True,student_distillation_parts=['source-domain original TRAIN']+(['source-domain original TEST soft targets'] if cfg.get('source_test_distillation') else []),
        student_original_TEST_images_in_loss=bool(cfg.get('source_test_distillation')),training_view_codes=list(range(8)),teacher_in_student_inference=False,
        logits_sha256=sha256(out/'TRAIN_teacher_logits.npz'),teacher_metadata=meta))
    return torch.from_numpy(logits).cuda()


def validate_same_source_parent(payload,cfg):
    prior=payload['metadata']['profile']
    for key in ('training_domain','dataset','architecture','main_layers','num_classes','data'):
        if prior.get(key)!=cfg.get(key):raise ValueError('Source-domain parent mismatch: '+key)
    if not cfg.get('source_test_distillation') or cfg['architecture']!='LG':raise ValueError('Parent reuse is confined to explicit same-source LG development')


def prepare_source_test_targets(raw,cfg,out):
    if not cfg.get('source_test_distillation'):return None,None
    if cfg['architecture']!='LG' or cfg.get('phase_update_mode','mini_batch')!='mini_batch' or cfg['phase_optimizer']!='wrapped' or cfg['augmentation']!='D4' or not cfg.get('alternating_head') or cfg.get('domain_route_aux',0) or not cfg.get('teacher',{}).get('cache_run'):raise ValueError('Source TEST KD requires explicit single-domain LG/D4/head blocks')
    teacher_dir=Path(__file__).resolve().parent/cfg['teacher']['cache_run']
    validate_domain_teacher_cache(raw,cfg,teacher_dir)
    number=0 if cfg['training_domain']=='RGB' else 1;indices=np.flatnonzero(raw['test']['domains']==number)
    path=teacher_dir/'teacher_logits_test.npz';manifest=json.loads((teacher_dir/'cache_manifest.json').read_text())
    if sha256(path)!=manifest[path.name]:raise ValueError('Source TEST teacher cache SHA differs')
    with np.load(path,allow_pickle=False) as z:
        for key in ('ids','labels','domains'):
            if not np.array_equal(z[key],raw['test'][key][indices]):raise ValueError('Source TEST teacher identity differs')
        logits=z['logits'].copy()
    if logits.shape!=(len(indices),8,cfg['num_classes']) or not np.isfinite(logits).all():raise ValueError('Invalid source TEST soft targets')
    np.savez_compressed(out/'SOURCE_TEST_teacher_logits.npz',logits=logits,**{k:raw['test'][k][indices] for k in ('ids','labels','domains')})
    write(out/'source_TEST_distillation_manifest.json',dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',original_TEST_domain=cfg['training_domain'],original_TEST_images=len(indices),opposite_domain_pixels_in_loss=False,TEST_ground_truth_in_student_loss=False,cache_sha256=sha256(out/'SOURCE_TEST_teacher_logits.npz')))
    return torch.as_tensor(indices,device='cuda'),torch.from_numpy(logits).cuda()


def soft_only_source_test_loss(logits,teacher_logits,weight,anchor,strength,kd_weight,temperature):
    # There is deliberately no hard-label argument on this path.
    from .original_f_teacher import teacher_KL
    return kd_weight*teacher_KL(logits,teacher_logits,temperature)+.5*strength*(weight-anchor).square().sum()
