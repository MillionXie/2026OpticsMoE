"""Fixed public traffic-sign cohort; no performance-dependent selection."""
import argparse,collections,csv,hashlib,io,json,os,subprocess,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import caltech20_alt246_richsplit_current6_sqrt_head as frozen
HERE,ROOT,s,engine=frozen.HERE,frozen.ROOT,frozen.s,frozen.engine
_BASE_PATHS=tuple(frozen.sources())
FILES=('gtsrb20_deadline_data.py','gtsrb20_deadline_profile.json','gtsrb20_deadline.py','check_gtsrb20_deadline.py','audit_gtsrb20_deadline.py','GTSRB20_DEADLINE_PROTOCOL_20261017.md')
def profile():
    p=s.read(HERE/FILES[1]);assert p['class_count']==20 and p['TRAIN_cap_per_class']==200 and p['maximum_epochs']==100 and p['horizontal_flip_probability']==0 and p['automatic_retry'] is False;return p
def sources():
    q={name:s.sha(ROOT/name) for name in _BASE_PATHS}
    for n in FILES:q[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return q
def identity():
    q=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(q,commit);return dict(sources=q,git_commit=commit)
def download():
    p=profile();directory=Path(p['archive_directory']);assert not directory.exists();s.save(directory.with_name(directory.name+'.claim.json'),dict(identity(),phase='single_GTSRB_author_archive_acquisition',pid=os.getpid()),exclusive=True);directory.mkdir();receipts={}
    for name,md5 in p['archives'].items():
        dest=directory/name;url=p['archive_base_url']+name;start=time.monotonic()
        with urllib.request.urlopen(url,timeout=45) as response,dest.open('xb') as out:
            while True:
                block=response.read(1024*1024)
                if not block:break
                out.write(block);assert time.monotonic()-start<1200,'Download time guard; preserve partial/no retry'
        raw=dest.read_bytes();assert hashlib.md5(raw).hexdigest()==md5,name
        with zipfile.ZipFile(dest) as z:assert z.testzip() is None
        receipts[name]=dict(url=url,bytes=len(raw),md5=md5,sha256=hashlib.sha256(raw).hexdigest());s.save(directory/'status.json',dict(state='archives_downloading',completed=list(receipts)))
    s.save(directory/'completed.json',dict(identity(),passed=True,archives=receipts,license='Author research download; no unverified CC license asserted; images not redistributed'),exclusive=True)
def archives():
    p=profile();d=Path(p['archive_directory']);receipt=s.read(d/'completed.json');assert receipt['passed'] and receipt['sources']==sources()
    for name,md5 in p['archives'].items():assert receipt['archives'][name]['md5']==md5 and s.sha(d/name)==receipt['archives'][name]['sha256']
    return d,receipt
def annotation_rows(z):
    rows=[]
    for name in sorted(n for n in z.namelist() if n.lower().endswith('.csv')):
        for raw in csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig')),delimiter=';'):
            q={k.lower():v for k,v in raw.items()};row=dict(path=str(Path(name).parent/ q['filename']).replace('\\','/'),class_id=int(q['classid']),roi=[int(q[k]) for k in ('roi.x1','roi.y1','roi.x2','roi.y2')]);rows.append(row)
    assert rows and all(r['path'] in z.namelist() for r in rows);return rows
def decode(z,row):
    raw=z.read(row['path'])
    with Image.open(io.BytesIO(raw)) as im:
        im=im.convert('RGB');x1,y1,x2,y2=row['roi'];assert 0<=x1<=x2<im.width and 0<=y1<=y2<im.height
        a=np.array(im.crop((x1,y1,x2+1,y2+1)).resize((150,150),Image.Resampling.BICUBIC)),hashlib.sha256(raw).hexdigest()
    return a
def sample(rows,cap,seed):
    groups=collections.defaultdict(list)
    for row in rows:groups[row['track']].append(row)
    rng=np.random.default_rng(seed);keys=sorted(groups);rng.shuffle(keys)
    for key in keys:rng.shuffle(groups[key])
    chosen=[]
    for offset in range(max(map(len,groups.values()))):
        for key in keys:
            if offset<len(groups[key]):chosen.append(groups[key][offset])
            if len(chosen)==cap:return sorted(chosen,key=lambda r:r['path'])
    return sorted(chosen,key=lambda r:r['path'])
def prepare():
    p=profile();out=Path(p['data_directory']);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and frozen.b.slots()['total']==0 and not out.exists();d,receipt=archives();s.save(out.with_name(out.name+'.claim.json'),dict(identity(),phase='fixed_count_track_component_preparation',pid=os.getpid()),exclusive=True);out.mkdir()
    z=zipfile.ZipFile(d/'GTSRB-Training_fixed.zip');rows=annotation_rows(z);counts=collections.Counter(r['class_id'] for r in rows);classes=sorted(sorted(counts,key=lambda k:(-counts[k],k))[:20]);rows=[r for r in rows if r['class_id'] in classes];parent={};pixel_owner={}
    def root(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for row in rows:
        row['track']=str(row['class_id'])+':'+Path(row['path']).stem.split('_')[0];image,rawsha=decode(z,row);row.update(raw_sha256=rawsha,cache_sha256=hashlib.sha256(image.tobytes()).hexdigest(),label=classes.index(row['class_id']));a=root(row['track']);key=row['cache_sha256']
        if key in pixel_owner:
            old=pixel_owner[key];assert old['label']==row['label'],'Conflicting exact pixels across classes';b=root(old['track']);parent[max(a,b)]=min(a,b)
        else:pixel_owner[key]=row
    chosen={role:[] for role in ('train','val','test')};all_train_records=[]
    for label,original in enumerate(classes):
        group=collections.defaultdict(list)
        for row in rows:
            if row['label']==label:group[root(row['track'])].append(row)
        keys=sorted(group);np.random.default_rng(p['split_seed']+original).shuffle(keys);cut=int(.8*len(keys));assert 0<cut<len(keys)
        for role,ks,cap in [('train',keys[:cut],p['TRAIN_cap_per_class']),('val',keys[cut:],p['VAL_cap_per_class'])]:
            part=[r for k in ks for r in group[k]]
            for r in part:r['component']=root(r['track']);r['assigned_role']=role
            selected=sample(part,cap,p['split_seed']+original+(1 if role=='train' else 2));assert len(selected)==cap if role=='train' else len(selected)>0;chosen[role].extend(selected);all_train_records.extend(part)
    test=zipfile.ZipFile(d/'GTSRB_Final_Test_Images.zip');gt=zipfile.ZipFile(d/'GTSRB_Final_Test_GT.zip');csvname=next(n for n in gt.namelist() if n.lower().endswith('.csv'));testrows=[]
    for raw in csv.DictReader(io.StringIO(gt.read(csvname).decode('utf-8-sig')),delimiter=';'):
        q={k.lower():v for k,v in raw.items()};c=int(q['classid'])
        if c not in classes:continue
        path=next(n for n in test.namelist() if Path(n).name==q['filename']);testrows.append(dict(path=path,class_id=c,label=classes.index(c),roi=[int(q[k]) for k in ('roi.x1','roi.y1','roi.x2','roi.y2')],track=None,component=None))
    chosen['test']=sorted(testrows,key=lambda r:r['path']);arrays={};ids={};supports={};hash_sets=[];selected_records={}
    for role,part in chosen.items():
        images=[];records=[]
        for row in part:
            image,rawsha=decode(test if role=='test' else z,row);row=dict(row,raw_sha256=rawsha,cache_sha256=hashlib.sha256(image.tobytes()).hexdigest());records.append(row);images.append(image)
        selected_records[role]=records;arrays[role+'_images']=np.stack(images);arrays[role+'_labels']=np.array([r['label'] for r in records],np.int64);arrays[role+'_ids']=np.array([('official_test/' if role=='test' else 'official_training/')+r['path'] for r in records]);ids[role]=arrays[role+'_ids'].tolist();supports[role]=np.bincount(arrays[role+'_labels'],minlength=20).tolist();hash_sets.append({r['cache_sha256'] for r in records})
    assert all(not hash_sets[i]&hash_sets[j] for i in range(3) for j in range(i)),'Cross-role exact cached pixels: retain evidence/no automatic repartition';assert supports['train']==[200]*20
    for key in ('track','component'):assert not {r[key] for r in chosen['train']}&{r[key] for r in chosen['val']}
    cache=out/'gtsrb20_track_split.npz';np.savez_compressed(cache,**arrays);s.save(out/'selected_records.json',selected_records,exclusive=True);s.save(out/'training_group_records.json',all_train_records,exclusive=True)
    manifest=dict(identity(),dataset=p['dataset'],classes=['GTSRB_'+str(k).zfill(5) for k in classes],original_class_ids=classes,training_raw_class_counts=dict(counts),split_ids=ids,supports=supports,cache_sha256=s.sha(cache),profile_sha256=s.sha(HERE/FILES[1]),selected_records_sha256=s.sha(out/'selected_records.json'),training_groups_sha256=s.sha(out/'training_group_records.json'),archive_receipts=receipt['archives'],cross_role_cached_pixel_intersections=0,TRAIN_VAL_track_component_intersections=0,license=receipt['license'],source=p['source'],cohort='20 count-selected classes and capped TRAIN/VAL; complete selected-class official final TEST; not full43 benchmark',models=0,TEST_model_evaluations=0)
    s.save(out/'data_manifest.json',manifest,exclusive=True);s.save(out/'completed.json',dict(passed=True,manifest_sha256=s.sha(out/'data_manifest.json'),cache_sha256=s.sha(cache)),exclusive=True);print(json.dumps(dict(passed=True,classes=classes,supports=supports,n={r:len(v) for r,v in ids.items()})))
def manifest():
    p=profile();d=Path(p['data_directory']);m=s.read(d/'data_manifest.json');c=s.read(d/'completed.json');assert c['passed'] and c['manifest_sha256']==s.sha(d/'data_manifest.json') and c['cache_sha256']==m['cache_sha256'] and m['sources']==sources() and m['profile_sha256']==s.sha(HERE/FILES[1]);return m
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['download','prepare']);a=p.parse_args()
    try:globals()[a.phase]()
    except BaseException as e:
        cfg=profile();out=Path(cfg['archive_directory'] if a.phase=='download' else cfg['data_directory']);mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
