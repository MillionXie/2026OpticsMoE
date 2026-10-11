"""Author v1 full segmented medicinal leaves; verify all native files, no class subset."""
import argparse, concurrent.futures, hashlib, io, json, time
from pathlib import Path
import requests
import numpy as np
from PIL import Image, ImageOps
from prepare_butterfly import grouped_split

BASE='https://data.mendeley.com/public-api/datasets/nnytj2v3n5'
HEADERS={'User-Agent':'Mozilla/5.0','Accept':'application/vnd.mendeley-public-dataset.1+json'}


def request(url):
    for attempt in range(4):
        try:
            r=requests.get(url,headers=HEADERS,timeout=(15,60));r.raise_for_status();return r
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(attempt+1)


def author_index():
    snapshot=request(BASE+'/snapshot/1').json()
    assert snapshot['doi']=='10.17632/nnytj2v3n5.1' and snapshot['licence']['short_name']=='CC BY 4.0'
    folders=request(BASE+'/folders/1').json()
    records=[]
    for folder in sorted(folders,key=lambda x:x['name']):
        start=0
        while True:
            url=BASE+'/files?folder_id='+folder['id']+'&version=1&$start='+str(start)+'&$limit=50'
            page=request(url).json();assert isinstance(page,list)
            for f in page:
                c=f['content_details']
                assert f['status']=='COMPLETED' and Path(f['filename']).name==f['filename']
                assert Path(f['filename']).suffix.lower() in ('.jpg','.jpeg','.png')
                records.append(dict(folder=folder['name'],name=f['filename'],id=f['id'],size=c['size'],
                                    sha256=c['sha256_hash'],url=c['download_url']))
            if len(page)<50:break
            start+=len(page)
    records.sort(key=lambda x:(x['folder'],x['name'],x['id']))
    assert len({x['id'] for x in records})==len(records) and 8<len({x['folder'] for x in records})<=60
    canonical=[[x[k] for k in ['folder','name','id','size','sha256']] for x in records]
    digest=hashlib.sha256(json.dumps(canonical,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    return snapshot,records,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--expected-index-sha',required=True);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'medicinal_fixed_split.npz').exists(),'Preserve existing prepared data'
    snapshot,records,digest=author_index();assert digest==a.expected_index_sha,'Published author index changed'
    (a.out/'author_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    (a.out/'author_file_index.json').write_text(json.dumps(records,indent=2))
    classes=sorted({x['folder'] for x in records})
    def download(row):
        folder=a.out/'originals'/row['folder'];folder.mkdir(parents=True,exist_ok=True)
        target=folder/row['name']
        if target.exists():content=target.read_bytes()
        else:
            content=request(row['url']).content
            assert len(content)==row['size'] and hashlib.sha256(content).hexdigest()==row['sha256']
            target.write_bytes(content)
        assert len(content)==row['size'] and hashlib.sha256(content).hexdigest()==row['sha256']
        with Image.open(io.BytesIO(content)) as source:
            image=ImageOps.exif_transpose(source).convert('RGB');pixels=np.asarray(image)
            original_hash=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
            cached=np.asarray(image.resize((150,150),Image.Resampling.BICUBIC)).copy()
        return dict(name=row['folder']+'/'+row['name'],label=classes.index(row['folder']),
            specimen=row['id'],original_hash=original_hash,cache_hash=hashlib.sha256(cached.tobytes()).hexdigest(),image=cached)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(download,records):
            rows.append(row)
            if len(rows)%100==0:print(json.dumps(dict(decoded=len(rows),total=len(records))),flush=True)
    split,groups=grouped_split(rows,20261011)
    arrays={};supports={};ids={}
    for key,indices in split.items():
        selected=[rows[i] for i in indices]
        arrays[key+'_images']=np.stack([x['image'] for x in selected])
        arrays[key+'_labels']=np.array([x['label'] for x in selected],dtype=np.int64)
        arrays[key+'_ids']=np.array([x['name'] for x in selected])
        supports[key]=np.bincount(arrays[key+'_labels'],minlength=len(classes)).tolist()
        ids[key]=arrays[key+'_ids'].tolist()
    cache=a.out/'medicinal_fixed_split.npz';np.savez_compressed(cache,**arrays)
    manifest=dict(dataset='MedicinalLeaf_author_segmented_v1',source='https://data.mendeley.com/datasets/nnytj2v3n5/1',
        doi=snapshot['doi'],license='CC BY 4.0',classes=classes,class_counts=np.bincount([x['label'] for x in rows],minlength=len(classes)).tolist(),
        total_records=len(rows),unique_groups=groups,supports=supports,split_ids=ids,split_seed=20261011,
        author_index_sha256=digest,cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grouping_source_sha256=hashlib.sha256((Path(__file__).parent/'prepare_butterfly.py').read_bytes()).hexdigest(),
        split_policy='Per-class 70/15/15 exact original/cache pixel connected groups, all native files/classes retained',
        preprocessing='Author-provided segmented full RGB image; EXIF transpose and bicubic150 only; no added segmentation, crop, offline augmentation or learned encoder',
        limitations='Author description contradicts counts; use verified files. File IDs do not establish leaf/plant identity; near-duplicate/session dependence not ruled out; image-level evaluation only')
    (a.out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ['total_records','classes','class_counts','supports','unique_groups','cache_sha256','author_index_sha256']}),flush=True)


if __name__=='__main__':main()
