"""Fetch the author release and audit official fold 0 before permitting training."""
import argparse
import base64
import csv
import hashlib
import io
import json
import socket
import subprocess
import urllib.request
import zipfile
from collections import Counter,defaultdict
from pathlib import Path
import gdown
import urllib3.util.connection
import numpy as np
from PIL import Image
from .train import save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    # Author revision verified before download; IPv4 avoids this server's stalled IPv6 route.
    rev='da084f62bcd1a2b0afeb8b81f1a27be3186391a1'
    urllib3.util.connection.allowed_gai_family=lambda:socket.AF_INET
    files={};tables={}
    for split in ['train','val','test']:
        name=split+'_subset0.csv';url=f'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/{rev}/labels/{name}'
        dest=a.root/name
        if not dest.exists():
            api=f'https://api.github.com/repos/AlexOlsen/DeepWeeds/contents/labels/{name}?ref={rev}'
            response=subprocess.check_output(['curl','-4','-fsSL','--connect-timeout','10','--max-time','30',api],timeout=40)
            payload=json.loads(response)
            assert payload['encoding']=='base64'
            dest.write_bytes(base64.b64decode(payload['content']))
        print('Loaded labels',name,flush=True)
        tables[split]=list(csv.DictReader(dest.open(encoding='utf-8-sig')))
        files[name]=dict(url=url,sha256=sha(dest))
    save(a.root/'source.json',dict(repository_commit=rev,files=files,license='CC-BY-4.0',
                                 license_source='https://github.com/AlexOlsen/DeepWeeds',fold_index=0))
    archive=a.root/'images.zip'
    if not archive.exists():
        part=a.root/'images.zip.part'
        gdown.download(id='1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj',output=str(part),quiet=False,resume=True)
        if not zipfile.is_zipfile(part):raise ValueError('Download is not a ZIP')
        part.rename(archive)
    arrays={};identity=[];hash_groups=defaultdict(list);shape_counts=Counter();names_by_split={}
    with zipfile.ZipFile(archive) as z:
        members={Path(n).name:n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png'))}
        for split,rows in tables.items():
            images=[];labels=[];ids=[]
            for row in rows:
                name=row['Filename'];label=int(row['Label']);raw=z.read(members[name])
                im=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'));shape_counts[str(im.shape)]+=1
                if im.shape!=(256,256,3):raise ValueError((name,im.shape))
                digest=hashlib.sha256(im.tobytes()).hexdigest();hash_groups[digest].append((split,name))
                images.append(im);labels.append(label);ids.append(name)
                identity.append(dict(sample_id=name,label=label,split=split,raw_sha256=hashlib.sha256(raw).hexdigest(),decoded_sha256=digest))
            arrays[split+'_images']=np.stack(images);arrays[split+'_labels']=np.array(labels,dtype=np.int64);arrays[split+'_ids']=np.array(ids)
            names_by_split[split]=ids
    overlaps=[v for v in hash_groups.values() if len({s for s,_ in v})>1]
    save(a.root/'image_manifest.json',identity)
    # Hash collisions of nearby frames are not a substitute for a scene audit.
    audit=dict(exact_cross_split_duplicates=overlaps,shape_counts=dict(shape_counts),
               supports={s:np.bincount(arrays[s+'_labels'],minlength=9).tolist() for s in tables},
               raw_archive_sha256=sha(archive),near_duplicate_scene_audit='pending',training_ready=False)
    save(a.root/'audit.json',audit)
    if overlaps:raise ValueError('Cross-split duplicate images; grouped profile required before training')
    cache=a.root/'deepweeds_fold0.npz'
    if cache.exists():raise FileExistsError(cache)
    np.savez_compressed(cache,**arrays)
    classes=['Chinee Apple','Lantana','Parkinsonia','Parthenium','Prickly Acacia','Rubber Vine','Siam Weed','Snake Weed','Negative']
    save(a.root/'data_manifest.json',dict(dataset='deepweeds',classes=classes,cache_sha256=sha(cache),
                                         split_ids=names_by_split,supports=audit['supports'],training_ready=False,
                                         scene_audit_pending=True,source_revision=rev,license='CC-BY-4.0'))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
