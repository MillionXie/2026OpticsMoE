"""All original PATCH16 tiles with the publisher's split and label identities."""
import csv,io,json,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .prepare_patch16 import digest,write

SPLITS={'Train':'train','Val':'val','Validation':'val','Test':'test'}

def build_manifest(root):
    root=Path(root);receipt=json.loads((root/'download_receipt.json').read_text())
    if receipt['license']!='CC BY 4.0':raise ValueError('unapproved license')
    if digest(root/'data.zip')!=receipt['sha256']:raise ValueError('archive checksum changed')
    with (root/'tiles-v2.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    required={'file','class','class_int','set'}
    if not rows or not required.issubset(rows[0]):raise ValueError(f'CSV columns: {list(rows[0]) if rows else []}')
    mapping={}
    for row in rows:
        label=int(row['class_int']);name=row['class']
        if name in mapping and mapping[name]!=label:raise ValueError('inconsistent label')
        mapping[name]=label
    if sorted(mapping.values())!=list(range(16)):raise ValueError('original sixteen classes required')
    if len(rows)!=129364:raise ValueError(f'publisher tile count mismatch: {len(rows)}')
    with zipfile.ZipFile(root/'data.zip') as z:
        names=set(z.namelist())
        lookup={}
        for name in names:
            if name.lower().endswith(('.jpg','.jpeg','.png')):
                base=Path(name).name
                if base in lookup:raise ValueError('ambiguous tile basename')
                lookup[base]=name
    ids=set();splits={key:[] for key in ('train','val','test')}
    for row in rows:
        if row['set'] not in SPLITS:raise ValueError(f'unknown official split: {row["set"]}')
        base=Path(row['file']).name
        if base not in lookup:raise ValueError(f'tile absent: {base}')
        member=lookup[base]
        if member in ids:raise ValueError('duplicate tile across original split')
        ids.add(member)
        splits[SPLITS[row['set']]].append(dict(id=member,label=int(row['class_int'])))
    counts={s:[Counter(r['label'] for r in records)[i] for i in range(16)] for s,records in splits.items()}
    if any(min(values)<1 for values in counts.values()):raise ValueError('class missing from split')
    # Preserve original metadata; do not infer patient independence from filenames.
    manifest=dict(dataset='PATCH16',license='CC BY 4.0',source_identifier=receipt['identifier'],
        archive_sha256=receipt['sha256'],csv_sha256=digest(root/'tiles-v2.csv'),
        class_names=[next(name for name,label in mapping.items() if label==i) for i in range(16)],
        total=len(ids),split_counts={s:len(v) for s,v in splits.items()},class_counts=counts,
        split_policy='unchanged original CSV set column; no sampling',csv_columns=list(rows[0]),
        patient_independence='not yet established from metadata',splits=splits)
    write(root/'manifest.json',manifest);return manifest

class Patch16Dataset(Dataset):
    def __init__(self,root,split,manifest=None):
        self.root=Path(root);self.manifest=manifest or json.loads((self.root/'manifest.json').read_text())
        self.records=self.manifest['splits'][split];self.archive=None
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        if self.archive is None:self.archive=zipfile.ZipFile(self.root/'data.zip')
        item=self.records[index]
        with Image.open(io.BytesIO(self.archive.read(item['id']))) as im:
            im=im.convert('RGB')
            if im.size!=(395,395):raise ValueError(f'original tile dimensions changed: {im.size}')
            image=np.array(im,copy=True)
        return image,item['label'],index

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    m=build_manifest(a.root);print(json.dumps({k:v for k,v in m.items() if k!='splits'},indent=2))
