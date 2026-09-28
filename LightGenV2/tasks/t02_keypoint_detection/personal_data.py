"""Shared personal-photo data contract for optical and frozen-Qwen models."""
import json
from pathlib import Path
import numpy as np
from .personal_prepare import sha256


def validate_manifest(path,allow_provisional=False):
    path=Path(path);data=json.loads(path.read_text(encoding='utf-8'));root=path.parent
    groups={};hashes={};ids=set();counts={'train':0,'test':0}
    for row in data['images']:
        if row['id'] in ids:raise ValueError('Duplicate photo ID')
        ids.add(row['id'])
        split=row.get('split');g=row.get('group_id')
        if split not in counts or not g:raise ValueError('Missing frozen group/split')
        for key,lookup in [(g,groups),('original:'+row['original_sha256'],hashes),('working:'+row['image_sha256'],hashes)]:
            if key in lookup and lookup[key]!=split:raise ValueError('Group/duplicate leakage')
            lookup[key]=split
        image=(root/row['image']).resolve()
        if not image.is_relative_to(root.resolve()):raise ValueError('Image outside dataset')
        if sha256(image)!=row['image_sha256']:raise ValueError('Image hash mismatch')
        if not row.get('orientation_reviewed'):raise ValueError('Orientation not checked')
        pids=set()
        for person in row['people']:
            if person['id'] in pids:raise ValueError('Duplicate person ID')
            pids.add(person['id'])
            if not person['include']:continue
            xy=np.asarray(person['keypoints'],dtype=float)
            if xy.shape!=(14,3) or not np.isfinite(xy).all():raise ValueError('Invalid keypoints')
            valid=xy[:,2]>0;w,h=row['size_wh']
            if ((xy[valid,:2]<0).any() or (xy[valid,0]>=w).any() or (xy[valid,1]>=h).any()):raise ValueError('Valid joint outside image')
            if not allow_provisional and (not person['reviewed'] or person.get('manual_required')):
                raise ValueError('Human review required; provisional mode is NOT formal evaluation')
            counts[split]+=1
    if min(counts.values())<1:raise ValueError('Empty split')
    return data,counts


def load_personal(path,allow_provisional=False):
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import PoseRecord,DatasetBundle
    data,counts=validate_manifest(path,allow_provisional);root=Path(path).parent
    splits={'train':[],'test':[]}
    for row in data['images']:
        for p in row['people']:
            if not p['include']:continue
            arr=np.asarray(p['keypoints'],dtype=np.float32);xy=arr[:,:2].copy();valid=arr[:,2]>0
            if allow_provisional:
                # Neither extrapolated head-top nor proxy neck is a trustworthy LSP target.
                valid[12:]=False
            xy[~valid]=np.nan
            splits[row['split']].append(PoseRecord(f"{row['id']}_{p['id']}",'personal',row['split'],len(splits[row['split']]),
                (root/row['image']).resolve(),xy,valid.astype(np.float32)))
    return DatasetBundle(splits['train'],splits['test'],{'counts':counts,'annotation_sha256':sha256(path),
        'provisional':allow_provisional,'supervised_joints':12 if allow_provisional else 14,
        'protocol':data['split_protocol'],'test_labels_are_human_gt':not allow_provisional})
