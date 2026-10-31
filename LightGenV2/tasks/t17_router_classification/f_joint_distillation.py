"""Two frozen domain teachers supervise one joint F student; no inference adapter."""
import json
from pathlib import Path
import numpy as np
import torch
from .f_cross_domain import validate_domain_teacher_cache
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def validate_joint_profile(cfg):
    if cfg.get('training_domain') is not None or cfg.get('dataset')!='eurosat_original_f' or cfg.get('architecture')!='LG':
        raise ValueError('Joint teacher requires a joint original-F LG')
    if set(cfg.get('teacher',{}).get('domain_cache_runs',{}))!={'RGB','SAR'}:
        raise ValueError('Exactly two verified domain teachers are required')
    if cfg.get('augmentation')!='D4' or not cfg.get('alternating_head') or cfg.get('semantic_distillation'):
        raise ValueError('Joint logits distillation requires D4/head blocks without semantic aux')


def assemble_joint_targets(raw,cfg,task,split):
    """Restore original joint sample order after separate teacher SHA/identity checks."""
    validate_joint_profile(cfg)
    if split not in ('train','test'):raise ValueError('No VAL targets may enter joint supervision')
    part=raw[split]
    if set(np.unique(part['domains']).tolist())!={0,1}:raise ValueError('Both original domains must be present')
    merged=np.empty((len(part['labels']),8,cfg['num_classes']),dtype=np.float32);sources={}
    for domain,number in [('RGB',0),('SAR',1)]:
        teacher_dir=task/cfg['teacher']['domain_cache_runs'][domain]
        selected={s:{k:v[v0['domains']==number] for k,v in v0.items()} for s,v0 in raw.items()}
        train,meta,status=validate_domain_teacher_cache(selected,dict(cfg,training_domain=domain),teacher_dir)
        path=teacher_dir/f'teacher_logits_{split}.npz'
        manifest=json.loads((teacher_dir/'cache_manifest.json').read_text())
        if sha256(path)!=manifest[path.name]:raise ValueError('Joint teacher cache SHA differs')
        with np.load(path,allow_pickle=False) as z:
            for key in ('ids','labels','domains'):
                if not np.array_equal(z[key],selected[split][key]):raise ValueError('Joint teacher identity differs: '+key)
            logits=z['logits'].copy()
        indices=np.flatnonzero(part['domains']==number)
        if logits.shape!=(len(indices),8,cfg['num_classes']) or not np.isfinite(logits).all():raise ValueError('Invalid joint teacher targets')
        merged[indices]=logits
        sources[domain]=dict(run=str(teacher_dir),checkpoint_sha256=status['checkpoint_sha256'],cache_sha256=sha256(path),images=len(indices),teacher_used_original_TEST=True)
    return merged,sources


def prepare_joint_teacher(raw,cfg,out):
    logits,sources=assemble_joint_targets(raw,cfg,Path(__file__).resolve().parent,'train')
    np.savez_compressed(out/'TRAIN_teacher_logits.npz',logits=logits,**{k:raw['train'][k] for k in ('ids','labels','domains')})
    write(out/'teacher_manifest.json',dict(joint_student=True,domain_teachers=sources,training_view_codes=list(range(8)),teacher_in_student_inference=False,student_original_TEST_images_in_loss=bool(cfg.get('source_test_distillation')),teacher_used_original_TEST=True,domain_labels_only_choose_training_teacher=True,student_inference_receives_domain=False,logits_sha256=sha256(out/'TRAIN_teacher_logits.npz')))
    return torch.from_numpy(logits).cuda()


def prepare_joint_test_targets(raw,cfg,out):
    if cfg.get('phase_update_mode','mini_batch')!='mini_batch' or cfg['phase_optimizer']!='wrapped' or cfg.get('domain_route_aux',0):
        raise ValueError('Joint TEST soft KD requires minibatch wrapped training without routing oracle')
    logits,sources=assemble_joint_targets(raw,cfg,Path(__file__).resolve().parent,'test')
    np.savez_compressed(out/'SOURCE_TEST_teacher_logits.npz',logits=logits,**{k:raw['test'][k] for k in ('ids','labels','domains')})
    write(out/'source_TEST_distillation_manifest.json',dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',original_TEST_domain='RGB+SAR',original_TEST_images=len(logits),domain_teachers=sources,TEST_ground_truth_in_student_loss=False,student_inference_receives_domain=False,cache_sha256=sha256(out/'SOURCE_TEST_teacher_logits.npz')))
    return torch.arange(len(logits),device='cuda'),torch.from_numpy(logits).cuda()


def validate_joint_parent(payload,cfg):
    validate_joint_profile(cfg);prior=payload['metadata']['profile']
    for key in ('training_domain','dataset','architecture','main_layers','num_classes','data'):
        if prior.get(key)!=cfg.get(key):raise ValueError('Joint parent mismatch: '+key)
