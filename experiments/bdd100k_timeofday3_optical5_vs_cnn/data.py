from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from .data_prepare import TIMEOFDAY3_CLASSES, ensure_timeofday3_dataset


CLASS_NAMES=list(TIMEOFDAY3_CLASSES)


class GrayscaleImageFolder(Dataset[Any]):
    def __init__(self,base:Any,mapping:dict[int,int],input_size:int)->None:
        self.base=base;self.mapping=mapping;self.input_size=input_size
        self.labels=[mapping[int(value)] for value in base.targets]
        self.paths=[str(path) for path,_ in base.samples]
    def __len__(self):return len(self.base)
    def __getitem__(self,index:int):
        image,old_label=self.base[index];image=ImageOps.grayscale(image.convert("RGB")).resize((self.input_size,self.input_size),Image.Resampling.BICUBIC)
        value=torch.from_numpy(np.asarray(image,dtype=np.float32)/255.0).unsqueeze(0)
        return value,self.mapping[int(old_label)],index,self.paths[index]


@dataclass
class DataBundle:
    train:Dataset[Any];validation:Dataset[Any];test:Dataset[Any];class_names:list[str];metadata:dict[str,Any]


def load_data(settings:Any)->DataBundle:
    try:from torchvision.datasets import ImageFolder
    except (ImportError,RuntimeError) as exc:raise RuntimeError("A compatible torchvision installation is required") from exc
    manifest=ensure_timeofday3_dataset(settings.data_root,settings.imagefolder_train,settings.imagefolder_test) if settings.download else None
    train_dir=settings.data_root/settings.imagefolder_train;test_dir=settings.data_root/settings.imagefolder_test
    for path in (train_dir,test_dir):
        if not path.is_dir():raise FileNotFoundError(f"TimeOfDay-3 ImageFolder missing: {path}")
    train_base=ImageFolder(str(train_dir));test_base=ImageFolder(str(test_dir));_validate(train_base.classes);_validate(test_base.classes)
    train:Dataset[Any]=GrayscaleImageFolder(train_base,{train_base.class_to_idx[name]:CLASS_NAMES.index(name) for name in CLASS_NAMES},settings.input_size)
    test:Dataset[Any]=GrayscaleImageFolder(test_base,{test_base.class_to_idx[name]:CLASS_NAMES.index(name) for name in CLASS_NAMES},settings.input_size)
    train=_per_class_limit(train,settings.train_limit_per_class,settings.seed);test=_per_class_limit(test,settings.test_limit_per_class,settings.seed+1)
    train=_total_limit(train,settings.train_limit,settings.seed+2);test=_total_limit(test,settings.test_limit,settings.seed+3)
    train_indices,val_indices=stratified_split_indices(train,settings.validation_fraction,settings.seed+4)
    train_split=Subset(train,train_indices);validation=Subset(train,val_indices)
    train_counts=class_counts(train_split);epoch_counts={name:min(count,settings.train_samples_per_class_per_epoch) if settings.train_samples_per_class_per_epoch is not None else count for name,count in train_counts.items()}
    return DataBundle(train_split,validation,test,list(CLASS_NAMES),{
        "dataset":"bdd100k_timeofday3","root":str(settings.data_root),"class_names":list(CLASS_NAMES),
        "train_samples":len(train_split),"validation_samples":len(validation),"test_samples":len(test),
        "per_class_train_counts":train_counts,"per_class_epoch_sample_counts":epoch_counts,"epoch_train_samples":sum(epoch_counts.values()),"per_class_validation_counts":class_counts(validation),"per_class_test_counts":class_counts(test),
        "validation_fraction":settings.validation_fraction,"train_limit":settings.train_limit,"test_limit":settings.test_limit,
        "train_limit_per_class":settings.train_limit_per_class,"test_limit_per_class":settings.test_limit_per_class,
        "train_samples_per_class_per_epoch":settings.train_samples_per_class_per_epoch,
        "manifest":manifest,
    })


def make_loader(dataset:Dataset[Any],batch_size:int,workers:int,shuffle:bool,seed:int,sampler:Sampler[int]|None=None)->DataLoader[Any]:
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle if sampler is None else False,sampler=sampler,num_workers=workers,collate_fn=collate,
                      pin_memory=torch.cuda.is_available(),persistent_workers=workers>0,generator=torch.Generator().manual_seed(seed))


def collate(batch:Sequence[Any]):
    images,labels,indices,paths=zip(*batch)
    return torch.stack(images),torch.tensor(labels,dtype=torch.long),torch.tensor(indices,dtype=torch.long),list(paths)


def labels_of(dataset:Dataset[Any])->list[int]:
    if hasattr(dataset,"labels"):return list(dataset.labels)
    if isinstance(dataset,Subset):
        parent=labels_of(dataset.dataset);return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset has no labels")


def stratified_split_indices(dataset:Dataset[Any],fraction:float,seed:int)->tuple[list[int],list[int]]:
    labels=labels_of(dataset);generator=torch.Generator().manual_seed(seed);train=[];validation=[]
    for cls in range(3):
        indices=[i for i,value in enumerate(labels) if value==cls];order=torch.randperm(len(indices),generator=generator).tolist()
        count=min(max(round(len(indices)*fraction),1),len(indices)-1) if len(indices)>1 else 0
        validation.extend(indices[p] for p in order[:count]);train.extend(indices[p] for p in order[count:])
    return sorted(train),sorted(validation)


def class_counts(dataset:Dataset[Any])->dict[str,int]:
    counts=Counter(labels_of(dataset));return {name:int(counts.get(i,0)) for i,name in enumerate(CLASS_NAMES)}


def _validate(classes:list[str])->None:
    if set(classes)!=set(CLASS_NAMES):raise ValueError(f"Expected classes {CLASS_NAMES}, found {classes}")


def _per_class_limit(dataset:Dataset[Any],limit:int|None,seed:int)->Dataset[Any]:
    if limit is None:return dataset
    labels=labels_of(dataset);generator=torch.Generator().manual_seed(seed);selected=[]
    for cls in range(3):
        indices=[i for i,value in enumerate(labels) if value==cls];order=torch.randperm(len(indices),generator=generator).tolist();selected.extend(indices[p] for p in order[:limit])
    return Subset(dataset,sorted(selected))


def _total_limit(dataset:Dataset[Any],limit:int|None,seed:int)->Dataset[Any]:
    if limit is None or limit>=len(dataset):return dataset
    labels=labels_of(dataset);generator=torch.Generator().manual_seed(seed);selected=[];base,remainder=divmod(limit,3)
    for cls in range(3):
        indices=[i for i,value in enumerate(labels) if value==cls];order=torch.randperm(len(indices),generator=generator).tolist();selected.extend(indices[p] for p in order[:base+int(cls<remainder)])
    if len(selected)<limit:
        remaining=sorted(set(range(len(dataset)))-set(selected));selected.extend(remaining[:limit-len(selected)])
    return Subset(dataset,sorted(selected))
