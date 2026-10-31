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
        teacher_never_used_opposite_domain=True,student_distillation_parts=['source-domain original TRAIN'],
        student_original_TEST_images_in_loss=False,training_view_codes=list(range(8)),teacher_in_student_inference=False,
        logits_sha256=sha256(out/'TRAIN_teacher_logits.npz'),teacher_metadata=meta))
    return torch.from_numpy(logits).cuda()
