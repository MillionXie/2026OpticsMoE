"""Prepare a fixed balanced subset of the original spatial TEST partition."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pure_optical'))
spec=importlib.util.spec_from_file_location('original_image_preparation',HERE.parent/'pure_optical/prepare.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    split=a.root/'SPLIT.json';assert p.digest(split,'sha256')=='cfe8373dd33cc0fe64f083b9ca32377e767c21b078c3f1d91f2dacecc25cb776'
    rows=json.loads(split.read_text())['records'];archives={};maps={};hashes={}
    for name,(url,filename,size,algorithm,expected) in p.SOURCES.items():
        path=a.root/'archives'/filename;assert path.stat().st_size==size and p.digest(path,algorithm)==expected
        hashes[name]=dict(sha256=p.digest(path,'sha256'),filename=filename);archives[name]=zipfile.ZipFile(path)
        suffix='.jpg' if name=='rgb' else '.tif'
        maps[name]={Path(x.filename).stem:x for x in archives[name].infolist() if Path(x.filename).suffix.lower()==suffix and not x.filename.startswith('__MACOSX/')}
    # Check extraction refactor against already used training arrays, never against test metrics.
    with np.load(a.root/'phase_only_v1/data.npz',allow_pickle=False) as z:
        for i in range(0,40,2):
            pid=str(z['train_ids'][i]).rsplit(':',1)[0]
            rgb,sar=p.decode_pair(pid,archives,maps)
            assert np.array_equal(rgb,z['train_images'][i]) and np.array_equal(sar,z['train_images'][i+1])
    selected=[]
    for label in range(10):
        candidates=[r for r in rows if r['domain']=='A' and r['split']=='test' and r['label']==label]
        candidates.sort(key=lambda r:hashlib.sha256(('shared_frontend_holdout_v1|'+r['pair_id']).encode()).hexdigest())
        assert len(candidates)>=100;selected.extend(candidates[:100])
    training_groups={r['spatial_group'] for r in rows if r['split'] in ['train','validation']}
    assert not training_groups&{r['spatial_group'] for r in selected}
    images=[];labels=[];domains=[];ids=[];records=[]
    for i,row in enumerate(selected):
        for domain,image in enumerate(p.decode_pair(row['pair_id'],archives,maps)):
            images.append(image);labels.append(row['label']);domains.append(domain);ids.append(row['pair_id']+':'+str(domain))
            records.append(dict(pair_id=row['pair_id'],spatial_group=row['spatial_group'],domain=domain,label=row['label'],split='test',pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        if (i+1)%250==0:print('test pairs prepared',i+1,flush=True)
    np.savez_compressed(a.out/'data.npz',test_images=np.stack(images),test_labels=np.array(labels),test_domains=np.array(domains),test_ids=np.array(ids))
    metadata=dict(protocol='eurosat_original_spatial_test_balanced100_v1',pairs_per_class=100,images=2000,
                  selection='sha256(shared_frontend_holdout_v1|pair_id), first100 per class in original test split',
                  original_split_sha256=p.digest(split,'sha256'),data_sha256=p.digest(a.out/'data.npz','sha256'),
                  source_archives=hashes,records=records,train_validation_spatial_overlap=0,preprocessing_parity_images=40,
                  preprocessing_source_sha256=p.digest(HERE.parent/'pure_optical/prepare.py','sha256'))
    (a.out/'manifest.json').write_text(json.dumps(metadata,indent=2));print(json.dumps({k:v for k,v in metadata.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
