"""Fixed source split, all-class disjoint task training pools, train-only replay."""
import hashlib
import json
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def balanced_indices(labels, total, seed):
    if total<8: raise ValueError('Need at least one image per class')
    rng=np.random.default_rng(seed); result=[]
    for c in range(8):
        ids=np.flatnonzero(labels==c); count=total//8+(c<total%8)
        if len(ids)<count: raise ValueError('Insufficient class support')
        result.extend(rng.permutation(ids)[:count])
    return np.asarray(result,dtype=np.int64)


def load(path, manifest, seed):
    metadata=json.loads(manifest.read_text())
    if metadata.get('license')!='CC BY 4.0': raise ValueError('Unverified license')
    if metadata.get('cache_sha256')!=sha(path): raise ValueError('Data hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files if k.startswith(('train_','val_'))}
        # Test images/labels are deliberately not read during development.
        test_ids=set(z['test_ids'].tolist())
    tr=set(arrays['train_ids'].tolist()); va=set(arrays['val_ids'].tolist())
    if tr & va or tr & test_ids or va & test_ids: raise ValueError('Split identity leakage')
    for split in ('train','val'):
        x,y=arrays[split+'_images'],arrays[split+'_labels']
        if x.dtype!=np.uint8 or x.shape[1:]!=(150,150,3): raise ValueError('Invalid Kather images')
        if set(y.tolist())!=set(range(8)): raise ValueError('Missing classes')
        if len(set(arrays[split+'_ids']))!=len(y): raise ValueError('Repeated IDs')
    rng=np.random.default_rng(seed); a=[]; b=[]
    for c in range(8):
        ids=rng.permutation(np.flatnonzero(arrays['train_labels']==c)); mid=len(ids)//2
        a.extend(ids[:mid]); b.extend(ids[mid:])
    arrays['A']=np.array(a); arrays['B']=np.array(b)
    return arrays,metadata


def domain(images, name):
    if name=='A': return images
    if name!='B': raise ValueError(name)
    # Deterministic synthetic color/illumination shift, not a clinical stain model.
    gains=images.new_tensor([1.12,.90,1.04],dtype=torch.float32)
    return (images.float()*gains+images.new_tensor([3.,-3.,1.],dtype=torch.float32)).clamp(0,255).round().to(torch.uint8)
