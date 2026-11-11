"""Strict cached-Qwen input and original video identities; no quality14 inputs."""
from pathlib import Path
import csv
import hashlib
import json
import torch


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_inputs(paths):
    """Load trusted local historical caches; preserve manifest order exactly."""
    v=torch.load(paths['vision_cache'],map_location='cpu',weights_only=False)
    l=torch.load(paths['language_cache'],map_location='cpu',weights_only=False)
    with Path(paths['manifest']).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    ids=[r['sample_id'] for r in rows];splits=[r['split'] for r in rows]
    if len(ids)!=2808 or len(set(ids))!=2808:raise ValueError('Expected 2808 unique video IDs')
    if splits.count('train')!=2250 or splits.count('test')!=558 or set(splits)!={'train','test'}:
        raise ValueError('Original 2250/558 split required')
    if ids!=v['sample_ids'] or splits!=v['splits']:raise ValueError('Cache/manifest IDs or order differ')
    if v.get('manifest_sha256')!=sha256(paths['manifest']):raise ValueError('Manifest SHA differs from cache')
    if tuple(v['vision_tokens'].shape)!=(2808,4,49,1024):raise ValueError('Wrong feature geometry')
    if v.get('qwen_front_pair_identity',{}).get('sha256')!=l.get('qwen_front_pair_identity',{}).get('sha256'):
        raise ValueError('Front/language identity pair mismatch')
    if v.get('qwen_front_contract',{}).get('vision_merger_executed') is not False:
        raise ValueError('Wrong Qwen feature boundary')
    if l.get('target_name')!='temporal' or not bool(l['attention_mask'].all()):
        raise ValueError('Wrong prompt')
    identity={name:{'path':str(paths[name]),'sha256':sha256(paths[name])}
              for name in ['vision_cache','language_cache','manifest']}
    targets=torch.tensor([float(r['temporal']) for r in rows])
    if not bool(torch.isfinite(targets).all()):raise ValueError('Nonfinite target')
    # Deliberately no quality_tokens, soft targets or teacher predictions returned.
    return {'vision':v['vision_tokens'],'target':targets,'ids':ids,'splits':splits,
            'prompt':l['language_tokens'][0],'identity':identity,
            'front_identity':v['qwen_vision_front_fingerprint'],
            'video_paths':[r['video_path'] for r in rows]}


def groups(payload,split,seed=None):
    indices=torch.tensor([i for i,s in enumerate(payload['splits']) if s==split])
    if seed is not None:indices=indices[torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed))]
    valid=torch.ones(len(indices),dtype=torch.bool)
    padding=(-len(indices))%16
    if padding:
        indices=torch.cat((indices,indices[:padding]));valid=torch.cat((valid,torch.zeros(padding,dtype=torch.bool)))
    return indices.reshape(-1,16),valid.reshape(-1,16)


def batch(payload,index,valid,device):
    result={'vision':payload['vision'][index].to(device,dtype=torch.float32),
            'target':payload['target'][index].to(device), 'valid':valid.to(device), 'indices':index}
    if 'teacher' in payload:
        result['teacher']=payload['teacher'][index].to(device)
        result['teacher_present']=payload['teacher_present'][index].to(device)
    return result


def attach_train_soft_targets(payload,path):
    teacher=torch.load(path,map_location='cpu',weights_only=False)
    ids=teacher['sample_ids'];names=teacher['target_names'];scores=teacher['predictions']
    expected={name for name,split in zip(payload['ids'],payload['splits']) if split=='train'}
    if len(set(ids))!=len(ids) or set(ids)!=expected:
        raise ValueError('Teacher IDs must match exactly the TRAIN set')
    if scores.shape!=(len(ids),len(names)) or 'temporal' not in names or not bool(torch.isfinite(scores).all()):
        raise ValueError('Invalid temporal teacher scores')
    if teacher.get('inference_dependency') is not False:
        raise ValueError('Unexpected teacher asset contract')
    target=scores[:,names.index('temporal')].float();lookup={name:i for i,name in enumerate(ids)}
    values=torch.zeros(len(payload['ids']));present=torch.zeros(len(payload['ids']),dtype=torch.bool)
    for i,(name,split) in enumerate(zip(payload['ids'],payload['splits'])):
        if split=='train':values[i]=target[lookup[name]];present[i]=True
    payload['teacher']=values;payload['teacher_present']=present
    payload['identity']['training_soft_targets']={'path':str(path),'sha256':sha256(path),
        'count':len(ids),'TEST_entries':0,'teacher_checkpoint':teacher.get('teacher_checkpoint')}
