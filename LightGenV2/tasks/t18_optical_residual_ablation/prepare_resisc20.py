"""Fixed-name RESISC20 preparation; original images/data outside source repository."""
import argparse,collections,hashlib,io,json,os,subprocess,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from prepare_caltech50 import split_groups
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];PLAN=HERE/'resisc20_data_plan.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())

def prepare(out,verified_archive=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists()
    plan=json.loads(PLAN.read_text());assert out==Path(plan['out']) and len(plan['classes'])==20
    out.mkdir(parents=True,exist_ok=False)
    save(out/'preparation_claim.json',dict(pid=os.getpid(),started_unix=time.time(),command=os.sys.argv,plan=plan,
         git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         source_sha256={p.name:sha(p) for p in (Path(__file__),PLAN,HERE/'prepare_caltech50.py')},automatic_retry=False))
    if verified_archive is not None:
        archive=Path(verified_archive);receipt=json.loads((archive.parent/'acquisition_receipt.json').read_text())
        assert receipt['passed'] and receipt['sha256']==sha(archive)==plan['archive_sha256']
        transfer=dict(verified_local_archive=str(archive),acquisition_receipt_sha256=sha(archive.parent/'acquisition_receipt.json'),network_download_repeated=False)
    else:
        archive=out/'NWPU-RESISC45.zip';part=out/'NWPU-RESISC45.download.part'
        request=urllib.request.Request(plan['download_url'],headers={'User-Agent':'T18 research dataset preparation'})
        with urllib.request.urlopen(request,timeout=60) as response,part.open('xb') as f:
            transfer=dict(requested_url=plan['download_url'],content_length=response.headers.get('Content-Length'),started_unix=time.time())
            while True:
                b=response.read(4*1024*1024)
                if not b:break
                f.write(b)
            f.flush();os.fsync(f.fileno())
        assert sha(part)==plan['archive_sha256'],'Archive differs from pinned known mirror'
        part.rename(archive)
    transfer.update(archive_sha256=sha(archive),bytes=archive.stat().st_size,completed_unix=time.time())
    save(out/'download_receipt.json',transfer)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=sorted(n for n in z.namelist() if n.lower().endswith('.jpg') and not n.startswith('__MACOSX/'))
        counts=collections.Counter(Path(n).parent.name for n in names)
        assert len(counts)==45 and sum(counts.values())==31500 and set(counts.values())=={700}
        selected=plan['classes'];assert set(selected)<=set(counts)
        assert selected==sorted(np.random.default_rng(plan['selection_seed']).choice(sorted(counts),20,replace=False).tolist())
        records=[];files=[]
        for name in names:
            category=Path(name).parent.name
            if category not in selected:continue
            raw=z.read(name)
            with Image.open(io.BytesIO(raw)) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');original=np.asarray(im).copy()
                assert list(original.shape)==plan['original_rgb_shape']
                cached=np.asarray(im.resize((150,150),Image.Resampling.BICUBIC)).copy()
            records.append(dict(name=name,label=selected.index(category),image=cached,
                                original_hash=hashlib.sha256(str(original.shape).encode()+original.tobytes()).hexdigest(),
                                cache_hash=hashlib.sha256(cached.tobytes()).hexdigest()))
            files.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    assert len(records)==14000
    owners={}
    for record in records:
        for key in ('original_hash','cache_hash'):
            identity=(key,record[key]);assert owners.get(identity,record['label'])==record['label'],'Conflicting duplicate labels'
            owners[identity]=record['label']
    splits={k:[] for k in ('train','val','test')};group_counts=[]
    for label in range(20):
        indices=[i for i,v in enumerate(records) if v['label']==label];rows=[records[i] for i in indices]
        _,groups=split_groups(rows,train_groups=1,val_groups=1)
        nt=int(.8*groups);nv=max(1,int(.1*groups));assert nt+nv<groups
        result,again=split_groups(rows,seed=plan['split_seed']+label,train_groups=nt,val_groups=nv);assert again==groups
        for role,selection in result.items():splits[role].extend(indices[i] for i in selection)
        group_counts.append(groups)
    arrays={};supports={};ids={}
    for role,indices in splits.items():
        indices.sort();rows=[records[i] for i in indices]
        arrays[role+'_images']=np.stack([v['image'] for v in rows]);arrays[role+'_labels']=np.array([v['label'] for v in rows],dtype=np.int64)
        arrays[role+'_ids']=np.array([v['name'] for v in rows]);supports[role]=np.bincount(arrays[role+'_labels'],minlength=20).tolist();ids[role]=arrays[role+'_ids'].tolist()
    for key in ('original_hash','cache_hash'):
        groups=[{records[i][key] for i in splits[role]} for role in splits]
        assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    cache=out/'resisc20_fixed_split.npz';np.savez_compressed(cache,**arrays)
    save(out/'original_file_manifest.json',files);save(out/'group_identity.json',[{k:v for k,v in row.items() if k!='image'} for row in records])
    manifest=dict(dataset=plan['dataset'],classes=selected,class_counts=[700]*20,total_records=14000,
        supports=supports,split_ids=ids,group_counts=group_counts,unique_groups=sum(group_counts),
        source=plan['author_source'],mirror=plan['download_url'],license=plan['license'],checksum_provenance=plan['checksum_provenance'],
        archive_sha256=sha(archive),cache_sha256=sha(cache),preparation_plan_sha256=sha(PLAN),preparation_source_sha256=sha(__file__),
        original_file_manifest_sha256=sha(out/'original_file_manifest.json'),group_identity_sha256=sha(out/'group_identity.json'),
        split_seed=plan['split_seed'],selection_seed=plan['selection_seed'],category_selection=plan['selection_policy'],
        preprocessing='Full RGB256 original -> bicubic150 cache; fixed RGB/RG/B/mean optical encoding unchanged. No crop/segmentation/learned features.',
        split_policy=plan['split_policy'],test_evaluations=0,
        limitations='Predeclared20-name subset, not all45; image-level exact-pixel groups do not establish location/scene independent generalization. No performance-guided category selection or optical test inference.')
    save(out/'data_manifest.json',manifest)
    print(json.dumps(dict(dataset=manifest['dataset'],classes=selected,total_records=14000,unique_groups=manifest['unique_groups'],supports=supports,
                         cache_sha256=manifest['cache_sha256'],manifest_sha256=sha(out/'data_manifest.json'),test_evaluations=0)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--archive',type=Path);a=p.parse_args()
    try:prepare(a.out,a.archive)
    except BaseException as e:
        if a.out.exists() and not (a.out/'failure.json').exists():save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False))
        raise
