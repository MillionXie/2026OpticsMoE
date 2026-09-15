import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.data import BalancedClassBatchSampler, build_eval_loader

CORRUPTIONS=('gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur','zoom_blur','snow','frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression')
from .protocol import load_policy
DATA_ROOT=Path(load_policy()['data_root'])
C_ROOT=Path(load_policy()['official_corrupted_root'])

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');os.replace(t,path)

def balanced_ids(ids,targets,n,seed):
    rng=np.random.default_rng(seed)
    return np.array(sorted(i for c in range(10) for i in rng.choice([x for x in ids if targets[x]==c],n,replace=False)),dtype=np.int64)

def transform_one(job):
    from . import corruptions
    import cv2
    cv2.setNumThreads(1)
    data,kind,severity,seed=job
    np.random.seed(int(seed))
    result=np.asarray(getattr(corruptions,CORRUPTIONS[kind])(Image.fromarray(data),severity=severity))
    if result.shape!=(32,32,3) or not np.isfinite(result).all():raise RuntimeError('Invalid corruption output')
    return np.clip(result,0,255).astype(np.uint8)

def prepare(bundle,smoke=False):
    root=Path(str(DATA_ROOT)+('_smoke' if smoke else ''));root.mkdir(parents=True,exist_ok=True)
    raw=bundle.validation.dataset.data;targets=np.asarray(bundle.validation.dataset.targets)
    replay=balanced_ids(bundle.train.indices,targets,2 if smoke else 600,4281)
    val=balanced_ids(bundle.validation.indices,targets,1 if smoke else 20,4282)
    assert not set(replay)&set(bundle.validation.indices)
    np.save(root/'replay_ids.npy',replay);np.save(root/'replay_clean.npy',raw[replay]);np.save(root/'replay_labels.npy',targets[replay])
    np.save(root/'val_ids.npy',val);np.save(root/'val_labels.npy',targets[val])
    epochs=[1,6,10,11,31,40] if smoke else range(1,41)
    with ProcessPoolExecutor(max_workers=2 if smoke else 6) as pool:
        for epoch in epochs:
            out=root/f'train_{epoch:03d}.npy'
            if out.exists():continue
            rng=np.random.default_rng(8042+epoch)
            kinds=np.arange(len(replay))%15;rng.shuffle(kinds)
            severity=np.ones(len(replay),dtype=np.int64) if epoch<=5 else rng.integers(1,4,len(replay))
            jobs=((raw[i],int(k),int(sev),10000000+epoch*100000+j) for j,(i,k,sev) in enumerate(zip(replay,kinds,severity)))
            values=np.stack(list(pool.map(transform_one,jobs,chunksize=32)))
            np.save(out,values);np.save(root/f'train_{epoch:03d}_conditions.npy',np.stack([kinds,severity],1))
            print(f'[data] train epoch={epoch} samples={len(values)}',flush=True)
        for k in range(15):
            for sev in (1,2,3):
                out=root/f'val_{k:02d}_{sev}.npy'
                if out.exists():continue
                jobs=((raw[i],k,sev,90000000+k*1000000+sev*100000+j) for j,i in enumerate(val))
                np.save(out,np.stack(list(pool.map(transform_one,jobs,chunksize=32))))
    manifest=dict(status='complete',split_sha256=bundle.metadata['split_sha256'],replay_count=len(replay),validation_count_per_condition=len(val),corruptions=CORRUPTIONS,train_severities=[1,2,3],validation_severities=[1,2,3],test_used_for_generation=False,train_epoch_files=[int(e) for e in epochs])
    manifest['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.npy'))}
    atomic_json(root/'manifest.json',manifest)
    return root

class ArrayDataset(Dataset):
    def __init__(self,images,labels):self.images=images;self.labels=labels
    def __len__(self):return len(self.images)
    def __getitem__(self,i):return Image.fromarray(np.asarray(self.images[i])),int(self.labels[i])

class PairedDataset(Dataset):
    def __init__(self,root,epoch):
        self.clean=np.load(root/'replay_clean.npy',mmap_mode='r');self.labels=np.load(root/'replay_labels.npy')
        self.corrupt=np.load(root/f'train_{epoch:03d}.npy',mmap_mode='r');self.conditions=np.load(root/f'train_{epoch:03d}_conditions.npy')
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):return self.clean[i],self.corrupt[i],int(self.labels[i]),i,self.conditions[i]

def pair_collate(rows):
    # Pair-adjacent order is deterministic and shared by all variants.
    images=[];meta=[]
    for a,b,y,i,(k,sev) in rows:
        images.extend([Image.fromarray(a),Image.fromarray(b)])
        meta.extend([[y,0,i,-1,0],[y,1,i,int(k),int(sev)]])
    return images,torch.tensor(meta,dtype=torch.long)

def pair_loader(root,epoch,s,smoke=False):
    ds=PairedDataset(root,epoch)
    sampler=BalancedClassBatchSampler(ds.labels,classes_per_batch=10,samples_per_class=1 if smoke else 3,steps=2 if smoke else 200,seed=9242)
    sampler.set_epoch(epoch)
    return DataLoader(ds,batch_sampler=sampler,num_workers=s.num_workers,collate_fn=pair_collate,pin_memory=False)

def validation_b(root):
    labels=np.load(root/'val_labels.npy')
    for k,name in enumerate(CORRUPTIONS):
        for sev in (1,2,3):yield name,sev,ArrayDataset(np.load(root/f'val_{k:02d}_{sev}.npy',mmap_mode='r'),labels)

def official_b():
    labels=np.load(C_ROOT/'labels.npy')
    for name in CORRUPTIONS:
        arr=np.load(C_ROOT/f'{name}.npy',mmap_mode='r')
        if arr.shape!=(50000,32,32,3):raise RuntimeError('Invalid official CIFAR-C shape')
        for sev in range(1,6):
            y=labels[:10000] if len(labels)==10000 else labels[(sev-1)*10000:sev*10000]
            yield name,sev,ArrayDataset(arr[(sev-1)*10000:sev*10000],y)
