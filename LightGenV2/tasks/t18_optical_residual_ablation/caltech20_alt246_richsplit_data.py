"""Single frozen count-selected, grouped re-split; all previous experiments preserved."""
import argparse,collections,gzip,hashlib,io,json,os,subprocess,tarfile,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import caltech20_alt246_deepdistill_continue as frozen
HERE,ROOT,s,engine=frozen.HERE,frozen.ROOT,frozen.s,frozen.engine
FILES=('caltech20_alt246_richsplit_data.py','caltech20_alt246_richsplit_profile.json','CALTECH20_ALT246_RICHSPLIT_PROTOCOL_20261016.md','caltech20_alt246_richsplit.py','check_caltech20_alt246_richsplit.py','audit_caltech20_alt246_richsplit.py')
PROFILE=HERE/FILES[1]
ROLES=('train','val','test')

def profile():
    p=s.read(PROFILE);assert p['maximum_epochs']==100 and p['previous_roles_may_move'] and len(p['classes'])==20 and p['classes']==sorted(p['classes']);return p

def sources():
    src=frozen.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src

def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)

def archive_rows():
    cfg=profile();raw=Path(cfg['archive']).read_bytes();assert hashlib.sha256(raw).hexdigest()==cfg['archive_sha256'] and hashlib.md5(raw).hexdigest()==cfg['archive_md5']
    z=zipfile.ZipFile(io.BytesIO(raw));names=[n for n in z.namelist() if Path(n).name=='101_ObjectCategories.tar.gz'];assert len(names)==1
    t=tarfile.open(fileobj=io.BytesIO(gzip.decompress(z.read(names[0]))),mode='r:');members=sorted((m for m in t.getmembers() if m.isfile() and m.name.lower().endswith('.jpg')),key=lambda m:m.name)
    counts=collections.Counter(Path(m.name).parent.name for m in members);eligible=[c for c in counts if c not in cfg['excluded_categories']];selected=sorted(sorted(eligible,key=lambda c:(-counts[c],c))[:20]);assert selected==cfg['classes'] and sum(counts[c] for c in selected)==3935
    rows=[]
    for m in members:
        category=Path(m.name).parent.name
        if category not in selected:continue
        raw=t.extractfile(m).read()
        with Image.open(io.BytesIO(raw)) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');original=np.asarray(im).copy();cached=np.asarray(im.resize((150,150),Image.Resampling.BICUBIC)).copy()
        rows.append(dict(name=m.name,label=selected.index(category),image=cached,original_hash=hashlib.sha256(str(original.shape).encode()+original.tobytes()).hexdigest(),cache_hash=hashlib.sha256(cached.tobytes()).hexdigest(),jpeg_sha256=hashlib.sha256(raw).hexdigest(),jpeg_bytes=len(raw)))
    return rows,dict(counts)

def assignments(rows):
    cfg=profile();parent=list(range(len(rows)));seen={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        for kind in ('original_hash','cache_hash'):
            token=(kind,r[kind])
            if token in seen:
                j=seen[token];assert r['label']==rows[j]['label'],'Cross-category exact pixels'
                a,b=root(i),root(j);parent[max(a,b)]=min(a,b)
            else:seen[token]=i
    groups={}
    for i in range(len(rows)):groups.setdefault(root(i),[]).append(i)
    splits={k:[] for k in ROLES};counts=[]
    for label,c in enumerate(cfg['classes']):
        keys=sorted(k for k,ix in groups.items() if rows[ix[0]]['label']==label);counts.append(len(keys));rg=np.random.default_rng(cfg['split_seed']+int(hashlib.sha256(c.encode()).hexdigest()[:6],16));rg.shuffle(keys)
        nt,nv=int(.8*len(keys)),int(.1*len(keys));assert min(nt,nv,len(keys)-nt-nv)>0
        for role,ks in zip(ROLES,(keys[:nt],keys[nt:nt+nv],keys[nt+nv:])):splits[role].extend(i for k in ks for i in groups[k])
    return {k:sorted(v) for k,v in splits.items()},counts

def prepare():
    cfg=profile();out=Path(cfg['data_directory']);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and frozen.b.slots()['total']==0 and not out.exists()
    s.save(out.with_name(out.name+'.claim.json'),dict(identity(),pid=os.getpid(),phase='single_count_selected_grouped_resplit',automatic_retry=False),exclusive=True);out.mkdir()
    rows,counts=archive_rows();splits,groups=assignments(rows);arrays={};ids={};supports={}
    for role,ix in splits.items():
        arrays[role+'_images']=np.stack([rows[i]['image'] for i in ix]);arrays[role+'_labels']=np.array([rows[i]['label'] for i in ix],np.int64);arrays[role+'_ids']=np.array([rows[i]['name'] for i in ix]);ids[role]=arrays[role+'_ids'].tolist();supports[role]=np.bincount(arrays[role+'_labels'],minlength=20).tolist()
    assert sorted(i for ix in splits.values() for i in ix)==list(range(len(rows)))
    for kind in ('original_hash','cache_hash'):
        sets=[{rows[i][kind] for i in splits[k]} for k in ROLES];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    cache=out/'caltech20_rich_split.npz';np.savez_compressed(cache,**arrays)
    old=frozen.b.old.manifest(Path(frozen.b.config()['data']));previous={sid:k for k in ROLES for sid in old['split_ids'][k]};overlap={oldrole:{newrole:sum(previous.get(sid)==oldrole for sid in ids[newrole]) for newrole in ROLES} for oldrole in ROLES}
    records=[{k:v for k,v in row.items() if k!='image'} for row in rows];s.save(out/'group_identity.json',records,exclusive=True)
    m=dict(identity(),dataset=cfg['dataset'],classes=cfg['classes'],class_counts=[counts[c] for c in cfg['classes']],category_counts=counts,unique_groups=groups,total_records=len(rows),split_ids=ids,supports=supports,cache_sha256=s.sha(cache),archive_sha256=cfg['archive_sha256'],profile_sha256=s.sha(PROFILE),group_identity_sha256=s.sha(out/'group_identity.json'),old_role_new_role_counts=overlap,source='https://data.caltech.edu/records/mzrjq-6wc02',doi='10.22002/D1.20086',license='CC BY4.0 primary CaltechDATA record; source archive not republished',test_informed_development=True,subject_independence_proven=False,model_forwards=0,optimizer_steps=0)
    s.save(out/'data_manifest.json',m,exclusive=True);s.save(out/'completed.json',dict(passed=True,manifest_sha256=s.sha(out/'data_manifest.json'),cache_sha256=s.sha(cache)),exclusive=True)
    print(json.dumps(dict(passed=True,classes=cfg['classes'],n={k:len(v) for k,v in ids.items()},supports=supports,unique_groups=groups,old_role_new_role_counts=overlap)),flush=True)

def manifest():
    cfg=profile();out=Path(cfg['data_directory']);r=s.read(out/'completed.json');m=s.read(out/'data_manifest.json');assert r['passed'] and r['manifest_sha256']==s.sha(out/'data_manifest.json') and r['cache_sha256']==m['cache_sha256'] and m['profile_sha256']==s.sha(PROFILE) and m['sources']==sources();return m

if __name__=='__main__':
    try:prepare()
    except BaseException as e:
        out=Path(profile()['data_directory']);mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
