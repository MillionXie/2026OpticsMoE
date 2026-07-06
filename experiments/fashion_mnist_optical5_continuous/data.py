from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any,Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader,Dataset,Sampler,Subset


CLASS_NAMES=["t_shirt_top","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle_boot"]


class FashionDataset(Dataset[Any]):
    def __init__(self,base:Any,input_size:int)->None:self.base=base;self.input_size=input_size;self.labels=[int(x) for x in base.targets]
    def __len__(self):return len(self.base)
    def __getitem__(self,index:int):
        image,label=self.base[index];image=image.convert("L").resize((self.input_size,self.input_size),Image.Resampling.BICUBIC);value=torch.from_numpy(np.asarray(image,dtype=np.float32)/255).unsqueeze(0);return value,int(label),index,f"fashion_mnist:{index}"


@dataclass
class DataBundle:
    train:Dataset[Any];validation:Dataset[Any];test:Dataset[Any];class_names:list[str];metadata:dict[str,Any]


def load_data(settings:Any)->DataBundle:
    try:from torchvision.datasets import FashionMNIST
    except (ImportError,RuntimeError) as exc:raise RuntimeError("A compatible torchvision installation is required") from exc
    full:Dataset[Any]=FashionDataset(FashionMNIST(str(settings.data_root),train=True,download=settings.download),settings.input_size);test:Dataset[Any]=FashionDataset(FashionMNIST(str(settings.data_root),train=False,download=settings.download),settings.input_size)
    full=_total_limit(full,settings.train_limit,settings.seed);test=_total_limit(test,settings.test_limit,settings.seed+1);train_indices,val_indices=stratified_split_indices(full,settings.validation_fraction,settings.seed+2);train=Subset(full,train_indices);validation=Subset(full,val_indices);train_counts=class_counts(train);epoch_counts={name:min(count,settings.train_samples_per_class_per_epoch) if settings.train_samples_per_class_per_epoch is not None else count for name,count in train_counts.items()}
    return DataBundle(train,validation,test,list(CLASS_NAMES),{"dataset":"fashion_mnist","root":str(settings.data_root),"class_names":list(CLASS_NAMES),"train_samples":len(train),"validation_samples":len(validation),"test_samples":len(test),"per_class_train_counts":train_counts,"per_class_validation_counts":class_counts(validation),"per_class_test_counts":class_counts(test),"per_class_epoch_sample_counts":epoch_counts,"epoch_train_samples":sum(epoch_counts.values()),"validation_fraction":settings.validation_fraction,"train_limit":settings.train_limit,"test_limit":settings.test_limit,"train_samples_per_class_per_epoch":settings.train_samples_per_class_per_epoch})


def labels_of(dataset:Dataset[Any])->list[int]:
    if hasattr(dataset,"labels"):return list(dataset.labels)
    if isinstance(dataset,Subset):
        parent=labels_of(dataset.dataset);return [parent[int(i)] for i in dataset.indices]
    raise TypeError("Dataset does not expose labels")


def stratified_split_indices(dataset:Dataset[Any],fraction:float,seed:int)->tuple[list[int],list[int]]:
    labels=labels_of(dataset);generator=torch.Generator().manual_seed(seed);train=[];validation=[]
    for cls in range(10):
        indices=[i for i,value in enumerate(labels) if value==cls];order=torch.randperm(len(indices),generator=generator).tolist();count=round(len(indices)*fraction);validation.extend(indices[p] for p in order[:count]);train.extend(indices[p] for p in order[count:])
    return sorted(train),sorted(validation)


def class_counts(dataset:Dataset[Any])->dict[str,int]:
    counts=Counter(labels_of(dataset));return {name:int(counts.get(i,0)) for i,name in enumerate(CLASS_NAMES)}


def make_loader(dataset:Dataset[Any],batch_size:int,workers:int,shuffle:bool,seed:int,sampler:Sampler[int]|None=None)->DataLoader[Any]:
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle if sampler is None else False,sampler=sampler,num_workers=workers,collate_fn=collate,pin_memory=torch.cuda.is_available(),persistent_workers=workers>0,generator=torch.Generator().manual_seed(seed))


def collate(batch:Sequence[Any]):
    images,labels,indices,paths=zip(*batch);return torch.stack(images),torch.tensor(labels),torch.tensor(indices),list(paths)


def _total_limit(dataset:Dataset[Any],limit:int|None,seed:int)->Dataset[Any]:
    if limit is None or limit>=len(dataset):return dataset
    labels=labels_of(dataset);generator=torch.Generator().manual_seed(seed);selected=[];base,remainder=divmod(limit,10)
    for cls in range(10):
        indices=[i for i,value in enumerate(labels) if value==cls];order=torch.randperm(len(indices),generator=generator).tolist();selected.extend(indices[p] for p in order[:base+int(cls<remainder)])
    return Subset(dataset,sorted(selected))
