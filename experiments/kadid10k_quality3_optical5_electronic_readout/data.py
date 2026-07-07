from __future__ import annotations

from dataclasses import dataclass
from typing import Any,Sequence

import numpy as np
import torch
from PIL import ImageOps,Image
from torch.utils.data import DataLoader,Dataset,Sampler,Subset

from experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.datasets import (
    QUALITY3_CLASS_NAMES,
    class_counts,
    labels_of,
    load_kadid10k,
    sample_metadata_of,
    stratified_split_indices,
)


class GrayscaleKADIDDataset(Dataset[Any]):
    def __init__(self,base:Dataset[Any],input_size:int)->None:
        self.base=base;self.input_size=int(input_size);self.labels=labels_of(base)
    def __len__(self)->int:return len(self.base)
    def __getitem__(self,index:int):
        image,label=self.base[index]
        image=ImageOps.grayscale(image.convert("RGB")).resize((self.input_size,self.input_size),Image.Resampling.BICUBIC)
        tensor=torch.from_numpy(np.asarray(image,dtype=np.float32)/255.0).unsqueeze(0)
        return tensor,label,index,sample_metadata_of(self.base,index)


@dataclass
class DataBundle:
    train:Dataset[Any];validation:Dataset[Any];test:Dataset[Any];class_names:list[str];metadata:dict[str,Any]


def load_data(settings:Any)->DataBundle:
    source=load_kadid10k(settings)
    train_base=GrayscaleKADIDDataset(source.train,settings.input_size)
    test=GrayscaleKADIDDataset(source.test,settings.input_size)
    train_indices,validation_indices=stratified_split_indices(source.train,settings.validation_fraction,settings.seed)
    train=Subset(train_base,train_indices);validation=Subset(train_base,validation_indices)
    metadata=dict(source.metadata)
    metadata.update({
        "baseline":"optical5_electronic_readout","input":"grayscale",
        "train_samples":len(train),"validation_samples":len(validation),"test_samples":len(test),
        "class_counts_train":class_counts(Subset(source.train,train_indices)),
        "class_counts_validation":class_counts(Subset(source.train,validation_indices)),
        "class_counts_test":class_counts(source.test),
    })
    return DataBundle(train,validation,test,list(QUALITY3_CLASS_NAMES),metadata)


def make_loader(dataset:Dataset[Any],batch_size:int,workers:int,shuffle:bool,seed:int,sampler:Sampler[int]|None=None)->DataLoader[Any]:
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle if sampler is None else False,sampler=sampler,
        num_workers=workers,collate_fn=collate,pin_memory=torch.cuda.is_available(),
        persistent_workers=workers>0,generator=torch.Generator().manual_seed(seed))


def collate(batch:Sequence[Any]):
    images,labels,indices,metadata=zip(*batch)
    return torch.stack(images),torch.tensor(labels,dtype=torch.long),torch.tensor(indices,dtype=torch.long),list(metadata)


__all__=["DataBundle","GrayscaleKADIDDataset","labels_of","load_data","make_loader"]
