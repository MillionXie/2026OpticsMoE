"""Convert CC-BY-4.0 Kather2018 VAL7K Arrow splits into the binary optical contract."""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from PIL import Image

from .data import sha


def convert(path, prefix):
    dataset=Dataset.from_file(str(path)); images=[]; labels=[]; ids=[]
    for row in dataset:
        image=row['image']
        if not isinstance(image,Image.Image): image=Image.open(image)
        images.append(np.asarray(image.convert('RGB').resize((150,150),Image.Resampling.LANCZOS),dtype=np.uint8))
        labels.append(0 if int(row['label'])==0 else 1)
        ids.append(prefix+str(row.get('image_id',row.get('filename',len(ids)))))
    return np.stack(images),np.asarray(labels,dtype=np.int64),np.asarray(ids)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--train-arrow',type=Path,required=True); parser.add_argument('--val-arrow',type=Path,required=True); parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    tx,ty,ti=convert(args.train_arrow,'kather2018:train:'); vx,vy,vi=convert(args.val_arrow,'kather2018:val:')
    np.savez_compressed(args.out,train_images=tx,train_labels=ty,train_ids=ti,val_images=vx,val_labels=vy,val_ids=vi)
    manifest={
        'dataset':'Kather2018 CRC-VAL-HE-7K binary','license':'CC BY 4.0',
        'source':'https://huggingface.co/datasets/nirschl-lab/kather_et_al_2018_val7k',
        'original_source':'https://zenodo.org/records/1214456','cache_sha256':sha(args.out),
        'source_sha256':{'train_arrow':sha(args.train_arrow),'validation_arrow':sha(args.val_arrow)},
        'split_protocol':'Use republisher train and validation splits; test Arrow not downloaded or read.',
        'label_map':{'0':'tumor','1':'non_tumor'},
        'limitations':'Binary non-tumor merges eight tissue classes. The original VAL7K is patient-independent from NCT-CRC-HE-100K, but no cross-dataset patient claim is made relative to Kather2016 or LC25000.'}
    args.out.with_name('kather2018_val7k_binary_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
