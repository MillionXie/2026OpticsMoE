"""Pair and align authors' data, split geographic groups, and verify every image."""
import hashlib,io,json,math,os,sys,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'data_deps'))
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject,Resampling
from affine import Affine
from pyproj import Transformer
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold
from download_archives import SOURCES,digest
PLAN=json.loads((ROOT/'PLAN_TEMPLATE.json').read_text());DATA=Path(PLAN['data_root'])
CLASSES=['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']
def signature(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)
def lookup(archive,suffix):
 result={}
 for info in archive.infolist():
  p=Path(info.filename)
  if p.suffix.lower()==suffix and not info.filename.startswith('__MACOSX/'):
   assert p.stem not in result,p.stem
   result[p.stem]=info
 return result
def main():
 assert not (ROOT/'SOURCE_MANIFEST.json').exists(),'Dataset preparation is forbidden after source sealing'
 started=time.time();archives={};checksums={}
 for name,(url,filename,size,alg,expected) in SOURCES.items():
  path=DATA/'archives'/filename
  atomic(ROOT/'data_progress.json',dict(state='checking_archive',archive=name,updated_at=time.time()))
  assert path.stat().st_size==size
  assert digest(path,alg)==expected
  checksums[name]=dict(url=url,bytes=size,algorithm=alg,checksum=expected,sha256=digest(path,'sha256'))
  archives[name]=zipfile.ZipFile(path)
 maps={name:lookup(z,'.jpg' if name=='rgb' else '.tif') for name,z in archives.items()}
 assert all(len(m)==27000 for m in maps.values()),{k:len(v) for k,v in maps.items()}
 assert set(maps['rgb'])==set(maps['ms'])==set(maps['sar'])
 pairs=[];manifest={};excluded=[];last=0;projections={}
 for index,pid in enumerate(sorted(maps['rgb'])):
  label_name=pid.rsplit('_',1)[0];label=CLASSES.index(label_name)
  rgb_bytes=archives['rgb'].read(maps['rgb'][pid]);sar_bytes=archives['sar'].read(maps['sar'][pid]);ms_bytes=archives['ms'].read(maps['ms'][pid])
  with Image.open(io.BytesIO(rgb_bytes)) as im:
   assert im.mode=='RGB' and im.size==(64,64),pid
   rgb=np.array(im)[4:60,4:60]
  with MemoryFile(ms_bytes) as m,MemoryFile(sar_bytes) as s,m.open() as ms,s.open() as sar:
   assert ms.width==ms.height==64 and ms.count==13 and sar.count==2
   assert ms.crs and sar.crs
   key=str(ms.crs)
   if key not in projections:projections[key]=Transformer.from_crs(ms.crs,'EPSG:3035',always_xy=True)
   cx,cy=ms.transform*(32,32);x,y=projections[key].transform(cx,cy)
   assert np.isfinite([x,y]).all()
   out=np.full((2,56,56),np.nan,dtype=np.float32)
   raw=sar.read().astype(np.float32)
   reproject(raw,out,src_transform=sar.transform,src_crs=sar.crs,src_nodata=sar.nodata,dst_transform=ms.transform*Affine.translation(4,4),dst_crs=ms.crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
   coverage=float(np.isfinite(out).mean())
   if coverage<.99:
    excluded.append(dict(pair_id=pid,reason='less_than_99percent_SAR_coverage',coverage=coverage));continue
   if all(np.nanstd(ch)<1e-6 for ch in out):
    excluded.append(dict(pair_id=pid,reason='constant_SAR_both_polarizations'));continue
   channels=[]
   for ch,mean,std in zip(out,[-12.59,-20.26],[5.26,5.91]):
    finite=np.isfinite(ch);ch=np.where(finite,ch,np.nanmedian(ch))
    lo,hi=np.quantile(ch,[.01,.99]);ch=np.clip(ch,lo,hi)
    channels.append(np.clip((ch-(mean-2*std))/(4*std),0,1))
   encoded=np.round(np.stack([channels[0],channels[1],(channels[0]+channels[1])/2],axis=2)*255).astype(np.uint8)
   meta=dict(pair_id=pid,label=label,label_name=label_name,x_m=x,y_m=y,source_crs=key,source_bounds=list(ms.bounds),sar_coverage=coverage,sar_source_shape=list(raw.shape),sar_source_range=[float(np.nanmin(raw)),float(np.nanmax(raw))])
  for domain,array in [('A',rgb),('B',encoded)]:
   name=f'images/{domain}/{label_name}/{pid}.png';dest=DATA/name;dest.parent.mkdir(parents=True,exist_ok=True)
   Image.fromarray(array).save(dest)
   manifest[name]=dict(sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest(),source_member=maps['rgb' if domain=='A' else 'sar'][pid].filename,source_crc32=maps['rgb' if domain=='A' else 'sar'][pid].CRC)
   meta[domain+'_path']=name
  pairs.append(meta)
  if time.time()-last>15:
   status=dict(state='aligning_verifying',pairs=len(pairs),source_pairs=index+1,total_source_pairs=27000,excluded=len(excluded),elapsed_sec=time.time()-started,updated_at=time.time())
   atomic(ROOT/'data_progress.json',status);print(json.dumps(status),flush=True);last=time.time()
 for z in archives.values():z.close()
 # Reject label-conflicting identical pixels before defining any model split.
 pixel_map={};bad=set()
 for i,p in enumerate(pairs):
  for d in ('A','B'):
   h=manifest[p[d+'_path']]['pixel_sha256'];pixel_map.setdefault(h,[]).append(i)
 for ids in pixel_map.values():
  if len({pairs[i]['label'] for i in ids})>1:bad.update(ids)
 if bad:
  excluded += [dict(pair_id=pairs[i]['pair_id'],reason='identical_pixels_conflicting_labels') for i in sorted(bad)]
  pairs=[p for i,p in enumerate(pairs) if i not in bad]
 assert len(pairs)>=26000,('Unexpectedly many invalid pairs',len(pairs),excluded[:10])
 # Union 50-km regions, close patches across region boundaries, and exact duplicates.
 n=len(pairs);parent=list(range(n))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 regions={};pixels={};coords=np.array([[p['x_m'],p['y_m']] for p in pairs])
 for i,p in enumerate(pairs):
  grid=tuple(np.floor(coords[i]/50000).astype(int));p['grid_id']=f'{grid[0]}_{grid[1]}'
  if grid in regions:union(i,regions[grid])
  else:regions[grid]=i
  for domain in ('A','B'):
   h=manifest[p[domain+'_path']]['pixel_sha256']
   if h in pixels:union(i,pixels[h])
   else:pixels[h]=i
 close=cKDTree(coords).query_pairs(3000,output_type='ndarray')
 for i,j in close:union(int(i),int(j))
 groups=np.array([find(i) for i in range(n)]);labels=np.array([p['label'] for p in pairs])
 assert len(np.unique(groups))>=20
 folds=np.full(n,-1);splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
 for fold,(_,idx) in enumerate(splitter.split(np.zeros(n),labels,groups)):folds[idx]=fold
 partitions=np.where(folds==0,'test',np.where(folds==1,'validation','train'))
 records=[]
 for i,p in enumerate(pairs):
  p.update(spatial_group=f'g{groups[i]}',fold=int(folds[i]),split=str(partitions[i]))
  for domain in ('A','B'):
   records.append(dict(path=p[domain+'_path'],pair_id=p['pair_id'],label=p['label'],label_name=p['label_name'],domain=domain,split=p['split'],spatial_group=p['spatial_group'],grid_id=p['grid_id']))
 records.sort(key=lambda r:(r['domain'],r['pair_id']))
 counts={d:{part:sum(r['domain']==d and r['split']==part for r in records) for part in ('train','validation','test')} for d in ('A','B')}
 class_counts={part:np.bincount(labels[partitions==part],minlength=10).tolist() for part in ('train','validation','test')}
 assert all(min(v)>=50 for v in class_counts.values()),class_counts
 group_sets={part:set(groups[partitions==part].tolist()) for part in ('train','validation','test')}
 distances={}
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  assert not group_sets[a]&group_sets[b]
  distances[a+'_'+b]=float(cKDTree(coords[partitions==a]).query(coords[partitions==b],k=1)[0].min())
 assert min(distances.values())>=3000
 files={r['path']:manifest[r['path']] for r in records};seen={};duplicate_pairs=[]
 for r in records:
  h=files[r['path']]['pixel_sha256']
  if h in seen:
   prev=seen[h];assert prev['split']==r['split'] and prev['label']==r['label'];duplicate_pairs.append([prev['path'],r['path']])
  else:seen[h]=r
 PLAN.update(status='data_verified_ready_for_seal',counts=counts,split_records_sha256=signature(records),classes=CLASSES,source_pairs=27000,retained_pairs=n,excluded_pairs=len(excluded))
 atomic(ROOT/'PLAN.json',PLAN);atomic(ROOT/'SPLIT.json',dict(records=records,policy=PLAN['split_policy']))
 atomic(ROOT/'SPLIT_AUDIT.json',dict(class_names=CLASSES,counts=counts,class_counts_per_domain=class_counts,spatial_groups=len(np.unique(groups)),groups_per_split={k:len(v) for k,v in group_sets.items()},cross_split_minimum_center_distances_m=distances,excluded_pairs=excluded,pairs=pairs))
 atomic(DATA/'IMAGE_MANIFEST.json',files)
 prep=dict(archives=checksums,encoding=PLAN['input_encoding'],sar_band_order=['VV','VH'],sar_db_statistics={'mean':[-12.59,-20.26],'std':[5.26,5.91]},sar_reference='https://github.com/zhu-xlab/FGMAE/blob/main/src/transfer_classification/datasets/EuroSat/eurosat_dataset_s1.py',calibration_uses_test_data=False,spatial_alignment='Both modalities use the optical center56x56 geographic region; SAR bilinear reproject to exactly that grid. Fixed crop removes mismatched source borders; no change to model optics.',rasterio_version=rasterio.__version__)
 atomic(ROOT/'DATA_PREPROCESSING.json',prep)
 result=dict(passed=True,images=len(records),pairs=n,source_pairs=27000,split_sha256=signature(records),paired_split_consistent=True,spatial_groups_disjoint=True,cross_split_minimum_center_distance_m=min(distances.values()),cross_split_or_label_duplicates=[],within_partition_duplicates=duplicate_pairs,excluded_pairs=excluded,archive_checksums=checksums,elapsed_sec=time.time()-started)
 atomic(ROOT/'DATA_CHECKS.json',result)
 # Preview fixed training pairs only. Test pixels never influence preprocessing parameters.
 canvas=Image.new('RGB',(10*150,2*170+36),'white');draw=ImageDraw.Draw(canvas)
 for c,name in enumerate(CLASSES):
  p=next(p for p in pairs if p['label']==c and p['split']=='train')
  for j,domain in enumerate(('A','B')):
   with Image.open(DATA/p[domain+'_path']) as im:canvas.paste(im.resize((144,144)),(c*150,j*170+25))
   draw.text((c*150+3,j*170+8),domain+' '+name[:16],fill='black')
 canvas.save(ROOT/'DATA_PREVIEW.png')
 atomic(ROOT/'data_progress.json',dict(state='complete',images=len(records),pairs=n,counts=counts,elapsed_sec=time.time()-started,updated_at=time.time()))
 print(json.dumps(dict(state='complete',counts=counts,class_counts=class_counts,groups=len(np.unique(groups)),minimum_center_distances_m=distances,excluded=len(excluded)),indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:atomic(ROOT/'data_progress.json',dict(state='failed',error=repr(e),traceback=traceback.format_exc()));raise
