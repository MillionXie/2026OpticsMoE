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


def domain(images, name, view="color_shift"):
    if name=='A': return images
    if name!='B': raise ValueError(name)
    if view in ('gray','edges'):
        weights=images.new_tensor([.299,.587,.114],dtype=torch.float32)
        gray=(images.float()*weights).sum(-1)
        if view=='edges':
            from torch.nn import functional as F
            # Fixed Sobel filters, per-image contrast scaling; no fitted parameters.
            k=gray.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])/8
            kernels=torch.stack((k,k.T))[:,None]
            grad=F.conv2d(F.pad(gray[:,None],(1,1,1,1),mode='replicate'),kernels)
            mag=grad.square().sum(1).sqrt()
            peak=mag.amax((-2,-1),keepdim=True)
            # A one-count floor prevents zero-power fields on flat images.
            gray=1+254*mag/peak.clamp_min(1e-12)
        return gray.clamp(0,255).round().to(torch.uint8)[...,None].expand(-1,-1,-1,3).contiguous()
    if view!='color_shift': raise ValueError('Unknown B representation: '+view)
    # Deterministic synthetic color/illumination shift, not a clinical stain model.
    gains=images.new_tensor([1.12,.90,1.04],dtype=torch.float32)
    return (images.float()*gains+images.new_tensor([3.,-3.,1.],dtype=torch.float32)).clamp(0,255).round().to(torch.uint8)
