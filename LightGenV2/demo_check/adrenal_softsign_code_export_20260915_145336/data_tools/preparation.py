from pathlib import Path
import argparse,csv,hashlib,json,shutil,time,urllib.request
import numpy as np
from .common import SPLITS,MD5,URL,sha,write_json,read_json,write_csv,now

def download(root,local_file=None):
    raw=Path(root)/'raw';raw.mkdir(parents=True,exist_ok=True);target=raw/'adrenalmnist3d.npz'
    valid=lambda p:p.is_file() and hashlib.md5(p.read_bytes()).hexdigest()==MD5
    errors=[];final_url=URL
    if target.exists() and not valid(target):
        shutil.copy2(target,target.with_name(target.name+'.invalid_'+str(time.time_ns())))
    if not valid(target):
        temp=target.with_suffix('.npz.part')
        for attempt in range(4):
            try:
                if local_file:shutil.copyfile(local_file,temp)
                else:
                    with urllib.request.urlopen(URL,timeout=45) as response,temp.open('wb') as out:
                        final_url=response.url
                        while chunk:=response.read(64*1024):out.write(chunk)
                if not valid(temp):raise ValueError('Official MD5 mismatch')
                temp.replace(target);break
            except Exception as exc:
                errors.append({'time':now(),'attempt':attempt+1,'error':repr(exc)})
                write_json(raw/'download_errors.json',errors)
                if attempt==3 or local_file:raise
                time.sleep(2)
    write_json(raw/'download_manifest.json',{'source':URL,'final_url':final_url,'time':now(),'local_path':str(target.resolve()),'local_file':local_file,'bytes':target.stat().st_size,'md5':MD5,'sha256':sha(target),'verified':True})
    print('Official data verified:',target,sha(target),flush=True)
    return target

def audit(root):
    root=Path(root);path=root/'raw/adrenalmnist3d.npz'
    if hashlib.md5(path.read_bytes()).hexdigest()!=MD5:raise ValueError('Official raw file MD5 mismatch')
    entries={};hashes={};fatal=[];zero_ids=[];all_unit=True;all_uint8=True
    with np.load(path,allow_pickle=False) as data:
        expected={s+'_'+k for s in SPLITS for k in ['images','labels']}
        if not expected.issubset(data.files):raise ValueError('Missing raw arrays')
        for split,n in SPLITS.items():
            x=data[split+'_images'];y=data[split+'_labels']
            if x.shape!=(n,28,28,28) or y.size!=n or not np.isin(y,[0,1]).all():raise ValueError((split,x.shape,y.shape))
            if not np.isfinite(x).all() or x.min()<0:raise ValueError('Invalid voxels')
            unique=np.unique(x);all_unit=all_unit and x.max()<=1;all_uint8=all_uint8 and x.dtype==np.uint8
            zeros=[]
            for i,(image,label) in enumerate(zip(x,y.reshape(-1))):
                sid=f'{split}_{i}';digest=hashlib.sha256(image.tobytes()).hexdigest()
                hashes.setdefault(digest,[]).append({'id':sid,'split':split,'source_index':i,'label':int(label)})
                if not np.any(image):zeros.append(sid);zero_ids.append(sid)
            entries[split]={'shape':list(x.shape),'dtype':str(x.dtype),'labels_shape':list(y.shape),'labels_dtype':str(y.dtype),
                'min':float(x.min()),'max':float(x.max()),'unique':unique.tolist(),'finite':True,
                'class_counts':np.bincount(y.reshape(-1).astype(int),minlength=2).tolist(),'zero_ids':zeros}
    encoding='unit' if all_unit else 'uint8_255' if all_uint8 else None
    if encoding is None:fatal.append('Unknown raw encoding')
    duplicates=[v for v in hashes.values() if len(v)>1]
    conflicts=[v for v in duplicates if len({i['label'] for i in v})>1]
    if conflicts:fatal.append('Exact duplicate volume has conflicting labels')
    if len(zero_ids)>15:fatal.append('More than 1% all-zero volumes; review before training')
    report={'source_sha256':sha(path),'md5':MD5,'arrays':entries,'raw_encoding':encoding,'duplicate_groups':duplicates,
        'cross_split_duplicate_groups':[v for v in duplicates if len({i['split'] for i in v})>1],
        'label_conflicts':conflicts,'all_zero_ids':zero_ids,'fatal_errors':fatal,'training_allowed':not fatal,
        'patient_ids_available':False,'patient_split':'Official patient-level split retained; patient grouping cannot be independently checked from this NPZ.',
        'label_mapping':{'0':'normal','1':'hyperplasia'},'description_terms':'normal adrenal gland or adrenal mass; official label dictionary normal/hyperplasia',
        'license':'CC BY 4.0','clinical_use':False}
    write_json(root/'reports/data_audit.json',report)
    (root/'reports/data_audit.md').write_text('# AdrenalMNIST3D data audit\n\n'+json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('DATA_AUDIT',json.dumps({k:report[k] for k in ['raw_encoding','all_zero_ids','label_conflicts','training_allowed']}),flush=True)
    if fatal:raise RuntimeError(fatal)
    return report

def normalize_raw(raw,encoding):
    if raw.ndim!=4 or raw.shape[1:]!=(28,28,28) or raw.shape[0]==0 or raw.dtype.kind not in 'buif':raise ValueError('Invalid raw shape/dtype')
    v=raw.astype(np.float32)
    if not np.isfinite(v).all():raise ValueError('Nonfinite voxels')
    if encoding=='unit':
        if v.min()<0 or v.max()>1:raise ValueError('Not unit data')
    elif encoding=='uint8_255':
        if raw.dtype!=np.uint8:raise ValueError('Expected uint8')
        v=v/np.float32(255.)
    else:raise ValueError('Unknown encoding')
    return np.ascontiguousarray(v)

def project_mean(volumes,volume_axis=2):
    if volume_axis not in (0,1,2) or volumes.ndim!=4 or volumes.shape[1:]!=(28,28,28):raise ValueError('Invalid spatial axis/shape')
    if not np.isfinite(volumes).all() or volumes.min()<0 or volumes.max()>1:raise ValueError('Expected normalized finite volumes')
    return np.ascontiguousarray(volumes.mean(axis=volume_axis+1,dtype=np.float32))

def prepare(root,axes=(0,1,2)):
    root=Path(root);report=read_json(root/'reports/data_audit.json')
    if not report['training_allowed']:raise RuntimeError('Audit blocks training')
    source=root/'raw/adrenalmnist3d.npz'
    if sha(source)!=report['source_sha256']:raise ValueError('Source changed after audit')
    with np.load(source,allow_pickle=False) as raw:
        for axis in axes:
            dest=root/f'projected2d/axis{axis}_mean';dest.mkdir(parents=True,exist_ok=True);arrays={};rows=[]
            for split,n in SPLITS.items():
                x=project_mean(normalize_raw(raw[split+'_images'],report['raw_encoding']),axis)
                y=raw[split+'_labels'].reshape(-1,1).astype(np.int64)
                ids=np.asarray([f'{split}_{i}' for i in range(n)],dtype='<U16')
                arrays.update({split+'_images':x,split+'_labels':y,split+'_ids':ids})
                for i in range(n):rows.append({'sample_id':ids[i],'split':split,'source_index':i,'label':int(y[i,0]),'volume_axis':axis,'projection':'mean','source_sha256':sha(source),'projection_sha256':hashlib.sha256(x[i].tobytes()).hexdigest()})
            if (dest/'data.npz').exists():
                with np.load(dest/'data.npz',allow_pickle=False) as prior:
                    if not all(np.array_equal(prior[k],v) for k,v in arrays.items()):raise ValueError('Existing projection differs')
            else:np.savez_compressed(dest/'data.npz',**arrays)
            write_csv(dest/'manifest.csv',rows)
            write_json(dest/'preprocessing.json',{'version':'adrenal_mean_v1','source_sha256':sha(source),'projection_npz_sha256':sha(dest/'data.npz'),'raw_encoding':report['raw_encoding'],'axis':axis,'batch_axis':axis+1,'projection':'mean','formula':f'normalized_volume.mean(axis={axis})','source_resolution':[28,28,28],'output_resolution':[28,28],'dtype':'float32','per_image_normalization':False,'anatomical_axis':'not established; array axis only','range':[float(min(arrays[s+'_images'].min() for s in SPLITS)),float(max(arrays[s+'_images'].max() for s in SPLITS))]})
            preview(arrays['train_images'],arrays['train_labels'].reshape(-1),arrays['train_ids'],dest/'preview_train.png')
            print('PROJECTED',axis,sha(dest/'data.npz'),flush=True)

def preview(images,labels,ids,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(7)
    indices=np.concatenate([rng.choice(np.flatnonzero(labels==c),16,replace=False) for c in (0,1)])
    fig,axs=plt.subplots(4,8,figsize=(16,8))
    for ax,i in zip(axs.flat,indices):
        ax.imshow(images[i],cmap='gray',vmin=0,vmax=1);ax.set_title(f'{ids[i]} / y={labels[i]}',fontsize=8);ax.axis('off')
    fig.suptitle('Fixed train examples: normalized mean projections, grayscale range [0,1]')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def cli(phase):
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--local-file');p.add_argument('--axes',type=int,nargs='+',default=[0,1,2]);p.add_argument('--projection',default='mean',choices=['mean']);a=p.parse_args()
    if phase=='download':download(a.root,a.local_file)
    elif phase=='audit':audit(a.root)
    else:prepare(a.root,a.axes)
