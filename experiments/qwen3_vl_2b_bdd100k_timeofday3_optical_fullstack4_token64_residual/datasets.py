from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any,Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader,Dataset,Subset

from .data_prepare import TIMEOFDAY3_CLASSES,ensure_timeofday3_dataset


TIMEOFDAY3_CLASS_NAMES=list(TIMEOFDAY3_CLASSES)


@dataclass
class DatasetBundle:
    train:Dataset[Any];test:Dataset[Any];class_names:list[str];metadata:dict[str,Any]


class RGBImageFolder(Dataset[Any]):
    def __init__(self,base:Any,mapping:dict[int,int])->None:
        self.base=base;self.mapping=mapping;self.labels=[mapping[int(value)] for value in base.targets]
    def __len__(self):return len(self.base)
    def __getitem__(self,index:int):
        image,label=self.base[index];return image.convert("RGB"),self.mapping[int(label)]


def load_timeofday3(settings:Any)->DatasetBundle:
    try:from torchvision.datasets import ImageFolder
    except (ImportError,RuntimeError) as exc:raise RuntimeError("A compatible torchvision installation is required") from exc
    manifest=ensure_timeofday3_dataset(settings.data_root,settings.imagefolder_train,settings.imagefolder_test) if settings.download else None
    train_base=ImageFolder(str(settings.data_root/settings.imagefolder_train));test_base=ImageFolder(str(settings.data_root/settings.imagefolder_test))
    _validate(train_base.classes);_validate(test_base.classes)
    train:Dataset[Any]=RGBImageFolder(train_base,{train_base.class_to_idx[name]:TIMEOFDAY3_CLASS_NAMES.index(name) for name in TIMEOFDAY3_CLASS_NAMES})
    test:Dataset[Any]=RGBImageFolder(test_base,{test_base.class_to_idx[name]:TIMEOFDAY3_CLASS_NAMES.index(name) for name in TIMEOFDAY3_CLASS_NAMES})
    train=_per_class_limit(train,settings.train_limit_per_class,settings.seed);test=_per_class_limit(test,settings.test_limit_per_class,settings.seed+1)
    train=_total_limit(train,settings.train_limit,settings.seed+2);test=_total_limit(test,settings.test_limit,settings.seed+3)
    train_indices,val_indices=stratified_split_indices(train,settings.validation_fraction,settings.seed)
    train_counts=class_counts(Subset(train,train_indices));epoch_counts={name:min(count,settings.train_samples_per_class_per_epoch) if settings.train_samples_per_class_per_epoch is not None else count for name,count in train_counts.items()}
    return DatasetBundle(train,test,list(TIMEOFDAY3_CLASS_NAMES),{
        "dataset":"bdd100k_timeofday3","root":str(settings.data_root),"class_names":list(TIMEOFDAY3_CLASS_NAMES),
        "full_train_samples":len(train),"train_samples":len(train_indices),"validation_samples":len(val_indices),"test_samples":len(test),
        "per_class_full_train_counts":class_counts(train),"per_class_train_counts":train_counts,"per_class_epoch_sample_counts":epoch_counts,"epoch_train_samples":sum(epoch_counts.values()),
        "per_class_validation_counts":class_counts(Subset(train,val_indices)),"per_class_test_counts":class_counts(test),
        "train_limit":settings.train_limit,"test_limit":settings.test_limit,"train_limit_per_class":settings.train_limit_per_class,"test_limit_per_class":settings.test_limit_per_class,"train_samples_per_class_per_epoch":settings.train_samples_per_class_per_epoch,
        "validation_fraction":settings.validation_fraction,"manifest":manifest,
    })


def labels_of(dataset:Dataset[Any])->list[int]:
    if hasattr(dataset,"labels"):return list(dataset.labels)
    if isinstance(dataset,Subset):
        parent=labels_of(dataset.dataset);return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset has no labels")


def class_counts(dataset:Dataset[Any])->dict[str,int]:
    counts=Counter(labels_of(dataset));return {name:int(counts.get(index,0)) for index,name in enumerate(TIMEOFDAY3_CLASS_NAMES)}


def stratified_split_indices(dataset:Dataset[Any],fraction:float,seed:int)->tuple[list[int],list[int]]:
    labels=labels_of(dataset);generator=torch.Generator().manual_seed(seed);train=[];validation=[]
    for cls in range(3):
        indices=[i for i,value in enumerate(labels) if value==cls];order=torch.randperm(len(indices),generator=generator).tolist();count=min(max(round(len(indices)*fraction),1),len(indices)-1) if len(indices)>1 else 0
        validation.extend(indices[p] for p in order[:count]);train.extend(indices[p] for p in order[count:])
    return sorted(train),sorted(validation)


class IndexedDataset(Dataset[Any]):
    def __init__(self,dataset:Dataset[Any])->None:self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index:int):
        image,label=self.dataset[index];return image,label,index


def indexed_collate(batch:Sequence[Any]):
    images,labels,indices=zip(*batch);return list(images),torch.tensor(labels,dtype=torch.long),torch.tensor(indices,dtype=torch.long)


def make_indexed_loader(dataset:Dataset[Any],batch_size:int,workers:int,shuffle:bool,seed:int):
    return DataLoader(IndexedDataset(dataset),batch_size=batch_size,shuffle=shuffle,num_workers=workers,collate_fn=indexed_collate,pin_memory=torch.cuda.is_available(),persistent_workers=workers>0,generator=torch.Generator().manual_seed(seed))


def _validate(classes:list[str])->None:
    if set(classes)!=set(TIMEOFDAY3_CLASS_NAMES):raise ValueError(f"Expected classes {TIMEOFDAY3_CLASS_NAMES}, found {classes}")
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
    return Subset(dataset,sorted(selected))
