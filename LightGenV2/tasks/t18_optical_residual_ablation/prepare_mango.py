"""Download pinned author v2 archive; fixed grouped stratified split, no augmentation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import requests
from PIL import Image, ImageOps

URL='https://data.mendeley.com/public-files/datasets/hb3kvgfcvm/files/3aba6ece-ea1d-4532-b098-47fe700203a2/file_downloaded'
SHA='8fa888ec8b2795f9f2e1785a2343201998d3eb005c22f3f04621680edaddc80b'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    archive=a.out/'author_v2.zip'
    if not archive.exists():
        temp=a.out/'author_v2.zip.partial'
        with requests.get(URL,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            h=hashlib.sha256();received=0
            with temp.open('wb') as f:
                for chunk in response.iter_content(8*1024*1024):
                    f.write(chunk);h.update(chunk);received+=len(chunk)
                    if received//(256*1024*1024)!=(received-len(chunk))//(256*1024*1024):
                        print(json.dumps(dict(downloaded_bytes=received)),flush=True)
        assert h.hexdigest()==SHA,'Author archive hash mismatch'
        temp.rename(archive)
    assert sha(archive)==SHA
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if Path(n).suffix.lower() in
            ['.jpg','.jpeg','.png'] and not n.startswith('__MACOSX/'))
        assert len(names)==2744,('Expected author v2 raw photos only',len(names))
        classes=sorted({Path(n).parent.name for n in names})
        assert len(classes)==8,classes
        rows=[];seen={}
        for n in names:
            with Image.open(io.BytesIO(z.read(n))) as raw:
                image=ImageOps.exif_transpose(raw).convert('RGB')
                pixels=np.asarray(image)
                identity=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
                label=classes.index(Path(n).parent.name)
                if identity in seen:assert seen[identity]==label,'Duplicate has conflicting label'
                seen[identity]=label
                resized=np.asarray(image.resize((150,150),Image.Resampling.BICUBIC)).copy()
            rows.append(dict(name=n,label=label,group=identity,image=resized))
    rng=np.random.default_rng(20261010)
    groups={split:set() for split in ['train','val','test']}
    for label in range(8):
        unique=sorted({row['group'] for row in rows if row['label']==label})
        rng.shuffle(unique);n=len(unique);ntrain=int(n*.7);nval=int(n*.15)
        assert min(ntrain,nval,n-ntrain-nval)>0
        for split,items in zip(groups,[unique[:ntrain],unique[ntrain:ntrain+nval],unique[ntrain+nval:]]):
            groups[split].update(items)
    arrays={};supports={};ids={};hashes={}
    for split,group in groups.items():
        subset=[row for row in rows if row['group'] in group]
        arrays[split+'_images']=np.stack([row['image'] for row in subset])
        arrays[split+'_labels']=np.array([row['label'] for row in subset],dtype=np.int64)
        arrays[split+'_ids']=np.array([row['name'] for row in subset])
        supports[split]=np.bincount(arrays[split+'_labels'],minlength=8).tolist()
        ids[split]=arrays[split+'_ids'].tolist();hashes[split]=sorted(group)
    assert sum(map(len,ids.values()))==2744
    cache=a.out/'mango_variety_fixed_split.npz'
    assert not cache.exists(),'Do not overwrite prepared split'
    np.savez_compressed(cache,**arrays)
    manifest=dict(dataset='MangoLeafVarietyBD_raw_v2',license='CC BY 4.0',
        doi='10.17632/hb3kvgfcvm.2',source='https://data.mendeley.com/datasets/hb3kvgfcvm/2',
        archive_sha256=SHA,cache_sha256=sha(cache),classes=classes,supports=supports,
        split_ids=ids,original_pixel_groups=hashes,total_records=2744,
        exact_duplicate_records=2744-len(seen),split_seed=20261010,
        split_policy='70/15/15 per-class exact-pixel groups; all records retained',
        preprocessing='EXIF transpose RGB -> bicubic150; originals kept in ZIP; no offline augmentation',
        limitations='Image-level split; leaf/tree identity unavailable, near-duplicate dependence not ruled out')
    (a.out/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(state='prepared',classes=classes,supports=supports,cache_sha256=manifest['cache_sha256'])),flush=True)


if __name__=='__main__':main()
