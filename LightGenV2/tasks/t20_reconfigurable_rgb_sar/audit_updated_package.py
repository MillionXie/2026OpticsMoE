"""Audit the supplied20261025 archive without executing source or loading PT."""
import argparse,csv,gzip,hashlib,io,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np

ZIP_SHA256='fa35dd1406d3d3575caf99e7a8d5f9c2abac1327e24e5778fad89e0aedc15f7a'
ROOT='EuroSAT_MoE_D2NN_20261025/'
DATA_NAMES=('source_data/data/data.npz','source_data/data/manifest.json','data/test_original_spatial_2000.npz','data/test_manifest.json','data/test_config.json','data/original_split.json.gz')


def sha(data):return hashlib.sha256(data).hexdigest()


def safe_member(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive member')


def audit(path,materialize=None):
    path=Path(path)
    if sha(path.read_bytes())!=ZIP_SHA256:raise ValueError('Unregistered updated package')
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate member')
        for name in names:safe_member(name)
        manifest=json.loads(z.read(ROOT+'BUNDLE_MANIFEST.json'));verified=[]
        for row in manifest['files']:
            safe_member(row['path']);data=z.read(ROOT+row['path'])
            if len(data)!=row['bytes'] or sha(data)!=row['sha256']:raise ValueError('Manifest mismatch '+row['path'])
            verified.append(row['path'])
        if len(verified)!=len(set(verified)):raise ValueError('Duplicate manifest path')
        extras=sorted(set(names)-{ROOT+n for n in verified}-{ROOT+'BUNDLE_MANIFEST.json'})
        if any(n.endswith(('.py','.pt','.npz')) for n in extras):raise ValueError('Unmanifested executable/data artifact')
        with np.load(io.BytesIO(z.read(ROOT+DATA_NAMES[0])),allow_pickle=False) as d:arrays={k:d[k] for k in d.files}
        with np.load(io.BytesIO(z.read(ROOT+DATA_NAMES[2])),allow_pickle=False) as d:arrays.update({k:d[k] for k in d.files})
        records=json.loads(z.read(ROOT+DATA_NAMES[1]))['records']+json.loads(z.read(ROOT+DATA_NAMES[3]))['records'];summary={};identities={}
        for split,n,per_class in [('train',6000,300),('validation',2000,100),('test',2000,100)]:
            x,y,domain,ids=(arrays[split+'_'+k] for k in ('images','labels','domains','ids'));rows=[r for r in records if r['split']==split]
            if x.shape!=(n,56,56,3) or x.dtype!=np.uint8 or len(set(ids))!=n or len(rows)!=n:raise ValueError('Data shape/unique identity changed')
            if not np.array_equal(domain.reshape(-1,2),np.tile([0,1],(n//2,1))) or not np.array_equal(y[::2],y[1::2]):raise ValueError('Complete RGB/SAR pairs required')
            if not np.array_equal(np.char.partition(ids[::2],':')[:,0],np.char.partition(ids[1::2],':')[:,0]):raise ValueError('Pair IDs differ')
            for i,row in enumerate(rows):
                if str(ids[i])!=row['pair_id']+':'+str(row['domain']) or int(y[i])!=row['label'] or int(domain[i])!=row['domain'] or sha(x[i].tobytes())!=row['pixel_sha256']:raise ValueError('Image/manifest identity differs')
            identities[split]={k:{str(r[k]) for r in rows} for k in ('pair_id','spatial_group','pixel_sha256')}
            counts={label:np.bincount(y[domain==d],minlength=10).tolist() for d,label in ((0,'RGB'),(1,'SAR'))}
            if any(v!=[per_class]*10 for v in counts.values()):raise ValueError('Fixed class balance changed')
            summary[split]=dict(images=n,paired_locations=n//2,class_counts=counts,IDs_sha256=sha(json.dumps(ids.tolist(),separators=(',',':')).encode()))
        overlaps=[]
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            for key in identities[a]:
                count=len(identities[a][key]&identities[b][key]);overlaps.append(dict(splits=[a,b],identity=key,count=count))
                if count:raise ValueError('Cross-split '+key+' overlap')
        cfg=json.loads(z.read(ROOT+'data/test_config.json'))
        for row in cfg['files'].values():
            if sha(z.read(ROOT+'data/'+row['path']))!=row['sha256']:raise ValueError('TEST source SHA differs')
        original=json.loads(gzip.decompress(z.read(ROOT+'data/original_split.json.gz')))
        index=json.loads(z.read(ROOT+'MODEL_INDEX.json'));results=[]
        for name,item in index['classifiers'].items():
            if sha(z.read(ROOT+item['checkpoint']))!=item['checkpoint_sha256']:raise ValueError('PT identity differs')
            with np.load(io.BytesIO(z.read(ROOT+item['predictions'])),allow_pickle=False) as d:
                if any(not np.array_equal(d[k],arrays['test_'+k]) for k in ('labels','domains','ids')):raise ValueError('Prediction/data identity differs')
                probability=d['probabilities'];pred=probability.argmax(1)
                if probability.shape!=(2000,10) or not np.isfinite(probability).all() or not np.allclose(probability.sum(1),1,atol=1e-5):raise ValueError('Invalid stored probabilities')
                counts=[int((pred[arrays['test_domains']==domain]==arrays['test_labels'][arrays['test_domains']==domain]).sum()) for domain in (0,1)]
            if any(abs(v/1000-e)>1e-12 for v,e in zip(counts,item['test_accuracy_RGB_SAR'])):raise ValueError('Published accuracy mismatch')
            results.append(dict(model=name,RGB_correct=counts[0],SAR_correct=counts[1],samples_per_domain=1000,checkpoint_SHA256=item['checkpoint_sha256'],scope='Recomputed stored predictions, not fresh inference or hardware'))
        data_files=[dict(member=name,sha256=sha(z.read(ROOT+name)),bytes=len(z.read(ROOT+name))) for name in DATA_NAMES]
        if materialize is not None:
            dest=Path(materialize);dest.mkdir(parents=True,exist_ok=False)
            for item in data_files:
                target=dest/item['member'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(ROOT+item['member']))
                if sha(target.read_bytes())!=item['sha256']:raise RuntimeError('DATA extraction differs')
            (dest/'manifest.json').write_text(json.dumps(dict(source_zip_SHA256=ZIP_SHA256,files=data_files,source_code_extracted=False),indent=2))
        return dict(passed=True,zip_SHA256=ZIP_SHA256,manifest_files_verified=len(verified),extra_nonexecuted_members=extras,data=summary,cross_split_overlaps=overlaps,
            models=results,data_files=data_files,original_split_record_count=len(original['records']),
            frontend='Frozen jointly RGB/SAR pretrained93472-parameter CNN, spatial128x7x7, differs fromT21 frozen32128 CLEVR frontend',
            geometry=dict(pixel_um=17,active=478,canvas=518,intermediate_CCD_OEO=True,current_native8_compatible_without_retraining=False),
            archived_code_executed=False,PT_unpickled=False,SDK=False)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--zip',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--materialize-data',type=Path);a=p.parse_args()
    r=audit(a.zip,a.materialize_data);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf8') as f:json.dump(r,f,indent=2,ensure_ascii=False)
    print(json.dumps(r,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
