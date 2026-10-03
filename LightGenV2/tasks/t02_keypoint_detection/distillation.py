"""Training-only frozen low-alpha heatmap targets; no deployment modules."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.training import preprocess_vision
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import masked_heatmap_mse, masked_heatmap_distillation
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router import training as base
from .modeling import architecture_label, build_student, sha256_file
from .settings import load_settings


def train_identity(records, test_records):
    ids = [r.sample_id for r in records]
    if len(set(ids)) != len(ids) or set(ids) & {r.sample_id for r in test_records}:
        raise RuntimeError('Teacher cache requires unique train identities disjoint from test')
    entries = [{'id':r.sample_id,'image':str(r.image_path),'keypoints':r.keypoints.tolist()} for r in records]
    digest = hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    return ids, digest


def build_train_cache(loaded, bundle, settings, checkpoint: Path, expected_sha: str):
    """Build before the student: one shared frozen Qwen front end, sequential models."""
    if sha256_file(checkpoint) != expected_sha:
        raise RuntimeError('Distillation teacher SHA mismatch')
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    teacher_settings = load_settings(Path(__file__).parent/'configs/moe_optical_router_scale_matched_dc20_no_shift_warmstart.yaml')
    for field in ('image_size','heatmap_size','crop_margin'):
        if getattr(teacher_settings,field)!=getattr(settings,field):
            raise RuntimeError(f'Teacher and student crop/heatmap geometry differs: {field}')
    if payload.get('checkpoint_architecture') != architecture_label(teacher_settings):
        raise RuntimeError('Teacher must be the pinned low-alpha optical model')
    if payload.get('router_contract_sha256') != settings.router_contract_sha256 or payload.get('weight_variant') != 'ema':
        raise RuntimeError('Teacher optical/EMA contract mismatch')
    ids, digest = train_identity(bundle.train,bundle.test)
    cache_settings = copy.copy(settings)
    # Cache workers end before training; avoid inheriting active CUDA state via fork.
    cache_settings.num_workers = 0
    loader = base._loader(bundle.train,cache_settings,training=False)
    teacher = build_student(loaded,teacher_settings)
    teacher.core.load_state_dict(payload['core'],strict=True)
    teacher.head.load_state_dict(payload['head'],strict=True)
    teacher.requires_grad_(False).eval()
    teacher.core.set_phase_dropout_active(False)
    cached = torch.empty((len(ids),14,settings.heatmap_size,settings.heatmap_size),dtype=torch.float32)
    count = 0
    try:
        with torch.no_grad():
            for index,batch in enumerate(loader):
                n=len(batch['images'])
                if list(batch['index']) != list(range(count,count+n)):
                    raise RuntimeError('Teacher cache row ordering mismatch')
                inputs=preprocess_vision(loaded.processor,batch['images'],loaded.device)
                prediction=teacher(**inputs)[0].detach().float().cpu()
                if prediction.shape != cached[count:count+n].shape or not torch.isfinite(prediction).all():
                    raise RuntimeError('Invalid teacher heatmap shape/values')
                cached[count:count+n].copy_(prediction);count+=n
                if index%100==0 or count==len(ids):print('TEACHER_CACHE',count,len(ids),flush=True)
        if count != len(ids):raise RuntimeError('Incomplete teacher cache')
        if any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('Teacher received gradients')
    finally:
        teacher.restore_native()
    del teacher,payload,loader
    gc.collect()
    if loaded.device.type=='cuda':torch.cuda.empty_cache()
    directory=settings.output_dir/'training_cache';directory.mkdir(exist_ok=False)
    path=directory/'teacher_heatmaps.npy'
    np.save(path,cached.numpy())
    report={'teacher_checkpoint':str(checkpoint.resolve()),'teacher_sha256':expected_sha,
            'teacher_epoch':50,'teacher_deployed':False,'cache_shape':list(cached.shape),
            'cache_dtype':'float32','cache_sha256':sha256_file(path),'cache_path':str(path),
            'ordered_train_ids':ids,'records_sha256':digest,'test_rows_cached':0,
            'canonical_crops':True,'augmentation_alignment':'existing warp_cached_heatmaps including joint flip permutation',
            'frozen_teacher':True,'deployment_extra_parameters':0}
    (directory/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return cached,report


def audit_distillation_gradient(model, bundle, loaded, settings, cache, opt):
    """One canonical training minibatch, no update. Verify phase supervision is live."""
    cfg=copy.copy(settings);cfg.num_workers=0;cfg.inference_batch_size=4
    batch=next(iter(base._loader(bundle.train[:4],cfg,training=False)))
    model.eval()
    prediction=model(**preprocess_vision(loaded.processor,batch['images'],loaded.device))[0]
    visible=batch['visible'].to(loaded.device)
    target=cache[:len(batch['images'])].to(loaded.device).detach()
    losses={'ground_truth':masked_heatmap_mse(prediction,batch['heatmaps'].to(loaded.device),visible),
            'distillation':masked_heatmap_distillation(prediction,target,visible)}
    params=[p for group in opt.param_groups for p in group['params']]
    report={}
    for name,loss in losses.items():
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        offset=0;norms={}
        for group in opt.param_groups:
            values=grads[offset:offset+len(group['params'])];offset+=len(group['params'])
            norm=sum(float(g.detach().float().square().sum()) for g in values if g is not None)**.5
            if not np.isfinite(norm):raise RuntimeError('Non-finite distillation gradient')
            norms[group['name']]=norm
        report[name]={'loss':float(loss.detach()),'gradient_l2_by_group':norms}
    if report['distillation']['gradient_l2_by_group']['feature_phase'] <= 0:
        raise RuntimeError('Teacher heatmap loss does not reach optical phases')
    opt.zero_grad(set_to_none=True)
    report['update_applied']=False
    return report
