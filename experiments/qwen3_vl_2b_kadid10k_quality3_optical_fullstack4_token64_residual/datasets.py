from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .data_prepare import ensure_kadid10k_dataset


QUALITY3_CLASS_NAMES=["high_quality","medium_quality","low_quality"]
IMAGE_COLUMNS=("distorted_image","image","image_name","filename","file_name","img","dist_img")
REFERENCE_COLUMNS=("reference_image","ref_image","ref_img","reference","ref","ref_id","reference_id")
SCORE_COLUMNS=("dmos","dmos_mean","mos","mos_mean","score","quality_score")
LEVEL_COLUMNS=("distortion_level","level","dist_level","severity")
TYPE_COLUMNS=("distortion_type","dist_type","distortion")


@dataclass(frozen=True)
class QualityRecord:
    image_path: Path
    image_name: str
    reference_image: str
    reference_id: int
    quality_score: float | None
    distortion_level: int | None
    distortion_type: str | None
    label: int


@dataclass
class DatasetBundle:
    train: Dataset[Any]
    test: Dataset[Any]
    class_names: list[str]
    metadata: dict[str,Any]


class KADIDQualityDataset(Dataset[Any]):
    def __init__(self,records:Sequence[QualityRecord])->None:
        self.records=list(records)
        self.labels=[record.label for record in self.records]
        self.references=[record.reference_image for record in self.records]
        self.reference_ids=[record.reference_id for record in self.records]

    def __len__(self)->int:return len(self.records)

    def __getitem__(self,index:int):
        record=self.records[index]
        with Image.open(record.image_path) as image:
            return image.convert("RGB"),record.label

    def sample_metadata(self,index:int)->dict[str,Any]:
        record=self.records[index]
        return {
            "image_path":str(record.image_path),
            "image_name":record.image_name,
            "reference_image":record.reference_image,
            "quality_score":record.quality_score,
            "distortion_level":record.distortion_level,
            "distortion_type":record.distortion_type,
        }


def load_kadid10k(settings:Any)->DatasetBundle:
    preparation=None
    if settings.download:
        preparation=ensure_kadid10k_dataset(settings.data_root,settings.metadata_csv,settings.image_dir,getattr(settings,"dataset_download_url",None))
        settings.metadata_csv=preparation["metadata_csv"]
        settings.image_dir=preparation["image_dir"]
    csv_path=_metadata_path(settings.data_root,settings.metadata_csv)
    rows,columns=_read_csv(csv_path)
    image_column=_find_column(columns,IMAGE_COLUMNS,"distorted image")
    reference_column=_find_column(columns,REFERENCE_COLUMNS,"reference image")
    score_column=_find_column(columns,SCORE_COLUMNS,"quality score",required=settings.quality_label_mode=="score_tertile")
    level_column=_find_column(columns,LEVEL_COLUMNS,"distortion level",required=settings.quality_label_mode=="distortion_level_3class")
    type_column=_find_column(columns,TYPE_COLUMNS,"distortion type",required=False)

    parsed=[]
    for row_number,row in enumerate(rows,start=2):
        image_name=str(row.get(image_column,"") or "").strip()
        reference=str(row.get(reference_column,"") or "").strip()
        if not image_name:raise RuntimeError(f"Empty image filename in {csv_path} at CSV row {row_number}")
        if not reference:raise RuntimeError(f"Empty reference image in {csv_path} at CSV row {row_number}")
        image_path=_resolve_image(settings.data_root,settings.image_dir,image_name,row_number)
        score=_optional_float(row.get(score_column) if score_column else None,"quality score",row_number)
        level=_optional_level(row.get(level_column) if level_column else None,row_number)
        distortion_type=str(row.get(type_column,"") or "").strip() or None if type_column else None
        parsed.append({"image_path":image_path,"image_name":image_name,"reference_image":reference,
                       "quality_score":score,"distortion_level":level,"distortion_type":distortion_type})
    if not parsed:raise RuntimeError(f"KADID metadata contains no samples: {csv_path}")

    if settings.quality_label_mode=="score_tertile":
        direction=_score_direction(score_column,settings.quality_score_higher_is_better)
        labels,thresholds=score_tertile_labels([item["quality_score"] for item in parsed],direction)
    else:
        direction=settings.quality_score_higher_is_better
        labels=[distortion_level_label(item["distortion_level"]) for item in parsed]
        thresholds=None
    settings.quality_score_column=score_column
    settings.quality_score_higher_is_better=direction

    reference_ids={name:index for index,name in enumerate(sorted({item["reference_image"] for item in parsed}))}
    records=[QualityRecord(**item,reference_id=reference_ids[item["reference_image"]],label=label) for item,label in zip(parsed,labels)]
    base=KADIDQualityDataset(records)
    train_indices,test_indices=reference_split_indices(base,settings.test_reference_fraction,settings.seed)
    train:Dataset[Any]=Subset(base,train_indices);test:Dataset[Any]=Subset(base,test_indices)
    train=_per_class_limit(train,settings.train_limit_per_class,settings.seed+1)
    test=_per_class_limit(test,settings.test_limit_per_class,settings.seed+2)
    train=_total_limit(train,settings.train_limit,settings.seed+3)
    test=_total_limit(test,settings.test_limit,settings.seed+4)
    student_train_indices,validation_indices=stratified_split_indices(train,settings.validation_fraction,settings.seed)
    student_train=Subset(train,student_train_indices);validation=Subset(train,validation_indices)
    train_counts=class_counts(student_train)
    epoch_counts={name:min(count,settings.train_samples_per_class_per_epoch) if settings.train_samples_per_class_per_epoch is not None else count for name,count in train_counts.items()}
    metadata={
        "dataset":"kadid10k_quality3","root":str(settings.data_root),"samples":len(base),
        "train_pool_samples":len(train),"train_samples":len(student_train),
        "validation_samples":len(validation),"test_samples":len(test),
        "class_names":list(QUALITY3_CLASS_NAMES),"class_counts_total":class_counts(base),
        "class_counts_train":train_counts,"class_counts_validation":class_counts(validation),
        "class_counts_test":class_counts(test),"per_class_epoch_sample_counts":epoch_counts,
        "epoch_train_samples":sum(epoch_counts.values()),
        "reference_count_total":len(set(references_of(base))),
        "reference_count_train":len(set(references_of(student_train))),
        "reference_count_validation":len(set(references_of(validation))),
        "reference_count_test":len(set(references_of(test))),
        "reference_disjoint_train_validation_test":_are_disjoint(student_train,validation,test),
        "train_reference_fraction":settings.train_reference_fraction,
        "test_reference_fraction":settings.test_reference_fraction,
        "validation_fraction":settings.validation_fraction,
        "quality_label_mode":settings.quality_label_mode,"quality_score_column":score_column,
        "quality_score_higher_is_better":direction,"score_tertile_thresholds":thresholds,
        "metadata_csv":str(csv_path),"image_dir":settings.image_dir,
        "distortion_level_column":level_column,"distortion_type_column":type_column,
        "preparation":preparation,
        "train_limit":settings.train_limit,"test_limit":settings.test_limit,
        "train_limit_per_class":settings.train_limit_per_class,
        "test_limit_per_class":settings.test_limit_per_class,
        "train_samples_per_class_per_epoch":settings.train_samples_per_class_per_epoch,
    }
    return DatasetBundle(train,test,list(QUALITY3_CLASS_NAMES),metadata)


def score_tertile_labels(scores:Sequence[float|None],higher_is_better:bool)->tuple[list[int],list[float]]:
    if any(value is None for value in scores):raise RuntimeError("score_tertile requires a numeric quality score for every sample")
    values=[float(value) for value in scores if value is not None]
    low=_quantile(values,1.0/3.0);high=_quantile(values,2.0/3.0)
    labels=[]
    for score in values:
        if higher_is_better:
            labels.append(2 if score<=low else 1 if score<=high else 0)
        else:
            labels.append(0 if score<=low else 1 if score<=high else 2)
    return labels,[low,high]


def distortion_level_label(level:int|None)->int:
    if level not in {1,2,3,4,5}:raise RuntimeError(f"distortion_level_3class requires levels 1-5, found {level}")
    return 0 if level in {1,2} else 1 if level==3 else 2


def infer_score_direction(column:str,configured:bool|None)->bool:
    return _score_direction(column,configured)


def reference_split_indices(dataset:Dataset[Any],test_fraction:float,seed:int)->tuple[list[int],list[int]]:
    references=references_of(dataset);unique=sorted(set(references))
    if len(unique)<2:raise RuntimeError("Reference-disjoint split requires at least two distinct reference images")
    random.Random(seed).shuffle(unique)
    test_count=min(max(round(len(unique)*test_fraction),1),len(unique)-1)
    test_references=set(unique[:test_count])
    train=[index for index,value in enumerate(references) if value not in test_references]
    test=[index for index,value in enumerate(references) if value in test_references]
    return train,test


def stratified_split_indices(dataset:Dataset[Any],fraction:float,seed:int)->tuple[list[int],list[int]]:
    try:return reference_split_indices(dataset,fraction,seed)
    except TypeError:pass
    labels=labels_of(dataset);generator=torch.Generator().manual_seed(seed);train=[];validation=[]
    for cls in range(3):
        indices=[i for i,value in enumerate(labels) if value==cls];order=torch.randperm(len(indices),generator=generator).tolist();count=min(max(round(len(indices)*fraction),1),len(indices)-1) if len(indices)>1 else 0
        validation.extend(indices[p] for p in order[:count]);train.extend(indices[p] for p in order[count:])
    return sorted(train),sorted(validation)


def labels_of(dataset:Dataset[Any])->list[int]:
    if hasattr(dataset,"labels"):return list(dataset.labels)
    if isinstance(dataset,Subset):
        parent=labels_of(dataset.dataset);return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset has no labels")


def references_of(dataset:Dataset[Any])->list[str]:
    if hasattr(dataset,"references"):return list(dataset.references)
    if isinstance(dataset,Subset):
        parent=references_of(dataset.dataset);return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset has no reference identities")


def reference_ids_of(dataset:Dataset[Any])->list[int]:
    if hasattr(dataset,"reference_ids"):return list(dataset.reference_ids)
    if isinstance(dataset,Subset):
        parent=reference_ids_of(dataset.dataset);return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset has no reference IDs")


def sample_metadata_of(dataset:Dataset[Any],index:int)->dict[str,Any]:
    if isinstance(dataset,Subset):return sample_metadata_of(dataset.dataset,int(dataset.indices[index]))
    if hasattr(dataset,"sample_metadata"):return dict(dataset.sample_metadata(index))
    return {}


def class_counts(dataset:Dataset[Any])->dict[str,int]:
    counts=Counter(labels_of(dataset));return {name:int(counts.get(index,0)) for index,name in enumerate(QUALITY3_CLASS_NAMES)}


class IndexedDataset(Dataset[Any]):
    def __init__(self,dataset:Dataset[Any])->None:self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index:int):
        image,label=self.dataset[index];return image,label,index


def indexed_collate(batch:Sequence[Any]):
    images,labels,indices=zip(*batch);return list(images),torch.tensor(labels,dtype=torch.long),torch.tensor(indices,dtype=torch.long)


def make_indexed_loader(dataset:Dataset[Any],batch_size:int,workers:int,shuffle:bool,seed:int):
    return DataLoader(IndexedDataset(dataset),batch_size=batch_size,shuffle=shuffle,num_workers=workers,collate_fn=indexed_collate,pin_memory=torch.cuda.is_available(),persistent_workers=workers>0,generator=torch.Generator().manual_seed(seed))


def _metadata_path(root:Path,value:str)->Path:
    path=Path(value).expanduser();path=path if path.is_absolute() else root/path
    if not path.is_file():raise FileNotFoundError(f"KADID metadata CSV not found: {path}. Set download=true and run --phase prepare_data, or configure data_root/metadata_csv manually.")
    return path.resolve()


def _read_csv(path:Path)->tuple[list[dict[str,str]],list[str]]:
    with path.open("r",encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle)
        if not reader.fieldnames:raise RuntimeError(f"KADID metadata CSV has no header: {path}")
        return list(reader),[str(name).strip() for name in reader.fieldnames]


def _find_column(columns:Sequence[str],candidates:Sequence[str],purpose:str,required:bool=True)->str|None:
    lookup={column.strip().lower():column for column in columns}
    for candidate in candidates:
        if candidate in lookup:return lookup[candidate]
    if required:
        raise RuntimeError(f"Missing {purpose} column. Accepted names: {list(candidates)}. CSV columns: {list(columns)}")
    return None


def _resolve_image(root:Path,image_dir:str,filename:str,row_number:int)->Path:
    normalized=Path(filename.replace("\\","/"))
    candidates=[normalized] if normalized.is_absolute() else [root/image_dir/normalized,root/normalized]
    for candidate in candidates:
        if candidate.is_file():return candidate.resolve()
    attempted="\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"KADID image not found for CSV row {row_number}: {filename}\nTried:\n{attempted}")


def _optional_float(value:Any,purpose:str,row_number:int)->float|None:
    if value is None or str(value).strip()=="":return None
    try:return float(str(value).strip())
    except ValueError as exc:raise RuntimeError(f"Invalid {purpose} at CSV row {row_number}: {value!r}") from exc


def _optional_level(value:Any,row_number:int)->int|None:
    parsed=_optional_float(value,"distortion level",row_number)
    if parsed is None:return None
    if not parsed.is_integer():raise RuntimeError(f"Distortion level must be an integer at CSV row {row_number}: {value!r}")
    return int(parsed)


def _score_direction(column:str|None,configured:bool|None)->bool:
    if configured is not None:return bool(configured)
    normalized=str(column or "").lower()
    if "dmos" in normalized:return False
    if "mos" in normalized:return True
    raise RuntimeError(f"Cannot infer whether higher quality score is better from column {column!r}; set quality_score_higher_is_better explicitly")


def _quantile(values:Sequence[float],fraction:float)->float:
    ordered=sorted(values)
    if not ordered:raise RuntimeError("Cannot compute score tertiles for an empty dataset")
    position=(len(ordered)-1)*fraction;lower=int(position);upper=min(lower+1,len(ordered)-1);weight=position-lower
    return float(ordered[lower]*(1.0-weight)+ordered[upper]*weight)


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


def _are_disjoint(train:Dataset[Any],validation:Dataset[Any],test:Dataset[Any])->bool:
    train_refs=set(references_of(train));validation_refs=set(references_of(validation));test_refs=set(references_of(test))
    return not (train_refs&validation_refs or train_refs&test_refs or validation_refs&test_refs)
