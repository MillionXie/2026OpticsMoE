"""Auditable fixed subsets for recovery experiments; no model-dependent selection."""
import argparse, csv, hashlib, json, tarfile, wave
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from LightGenV2.tasks.t09_multimodal_matching.model import normalize_power
from LightGenV2.tasks.t11_multimodal_nature.sonyc_prepare import EVENTS

def save(p,obj):
 p.write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(x): return hashlib.sha256(('17:'+x).encode()).hexdigest()
def write_split(out,split,fields,labels,records):
 assert len(fields)==len(labels)==len(records) and len(fields)>0
 np.savez_compressed(out/(split+'.npz'),fields=np.stack(fields).astype(np.float16),labels=np.array(labels,dtype=np.int64))
 save(out/(split+'_records.json'),records)
 print(json.dumps({'split':split,'n':len(labels),'counts':dict(Counter(map(int,labels)))}),flush=True)

def audio(a):
 root=a.root; ann=root/'metadata/annotations.csv'; by=defaultdict(list)
 files={p.name:p for p in (root/'audio0').rglob('*.wav')}
 for r in csv.DictReader(ann.open()):
  if r['audio_filename'] in files: by[r['audio_filename']].append(r)
 names=sorted(by); words=['does','the','recording','contain']+[e.split('_')[1] for e in EVENTS]
 vocab={w:i+2 for i,w in enumerate(words)}
 bank_hz=torch.linspace(0,8000,257)
 edges=700*(10**(torch.linspace(2595*np.log10(1+20/700),2595*np.log10(1+8000/700),66)/2595)-1)
 bank=torch.minimum((bank_hz[None]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),(edges[2:,None]-bank_hz[None])/(edges[2:,None]-edges[1:-1,None])).clamp_min(0)
 splits={s:([],[],[]) for s in ['train','val','test']}; excluded=[]
 for j,name in enumerate(names):
  rows=by[name]; part={'validate':'val'}.get(rows[0]['split'],rows[0]['split']); assert len({r['split'] for r in rows})==1
  with wave.open(str(files[name])) as w:
   assert w.getnchannels()==1 and w.getsampwidth()==2
   rate=w.getframerate(); x=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
  x=resample_poly(x,16000,rate); x=np.pad(x[:160000],(0,max(0,160000-len(x))))
  st=torch.stft(torch.tensor(x),n_fft=512,hop_length=160,win_length=400,window=torch.hann_window(400),return_complex=True).abs().square()
  db=10*torch.log10((bank@st).clamp_min(1e-10)); spec=((db-db.max()).clamp(-80,0)+80)/80
  sensor=F.interpolate(spec[None,None],(224,112),mode='bilinear',align_corners=False)[0,0]; sensor=normalize_power(sensor,.5)
  for event in EVENTS:
   verified=[int(r[event]) for r in rows if r['annotator_id']=='0' and r[event] in ('0','1')]
   if verified:
    assert len(set(verified))==1; y=verified[0]; source='verified_annotator_0'
   else:
    assert part!='test', 'Test requires verified consensus'
    votes=[int(r[event]) for r in rows if int(r['annotator_id'])>0 and r[event] in ('0','1')]
    if not votes or 2*sum(votes)==len(votes): excluded.append([name,event,'missing_or_tied']); continue
    y=int(2*sum(votes)>len(votes)); source='crowd_majority'
   code=torch.zeros(32,64)
   for k,w in enumerate(['does','the','recording','contain',event.split('_')[1]]): code[k,vocab[w]]=1
   text=F.interpolate(code[None,None],(224,112),mode='nearest')[0,0]; text=normalize_power(text,.5)
   f,l,r=splits[part]; f.append(torch.cat([sensor,text],-1).numpy()); l.append(y); r.append(dict(audio=name,event=event,label=y,source=source,sensor=rows[0]['sensor_id']))
  if j%100==0: print('audio',j,len(names),flush=True)
 for s,(f,l,r) in splits.items(): write_split(a.out,s,f,l,r)
 sets=[{r['audio'] for r in splits[s][2]} for s in splits]
 assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
 save(a.out/'protocol.json',dict(task='sonyc',classes=2,events=EVENTS,source_sha=sha(ann),scope='audio-0 subset, exploratory; only 27 test recordings',label_policy='verified annotator 0 preferred; otherwise crowd majority; ties excluded; no individual expert annotations',audio='anti-aliased 16kHz, full 10 sec logmel 64x1001, bilinear 224x112',text='fixed word-position onehot 32x64, nearest 224x112',power=[.5,.5],excluded=excluded,vocab=vocab))

def sen(a):
 import rasterio
 root=a.root; meta=root/'metadata'; labfile=meta/'single_label_IGBPsimple_ClsNum.txt'
 labels={line.split(':')[0].strip():int(line.split(':')[1]) for line in labfile.read_text().splitlines() if ':' in line}
 assert set(labels.values())<=set(range(1,11)), sorted(set(labels.values()))
 pool={}
 for split,fn in [('train','train_list.txt'),('test','test_list.txt')]:
  d=defaultdict(list)
  for line in (meta/fn).read_text().splitlines():
   n=Path(line.strip()).name
   if n.startswith('ROIs1158_spring_') and n in labels: d[n.rsplit('_p',1)[0]].append(n)
  pool[split]=d
 train_scenes=sorted(pool['train'],key=rank); test_scenes=sorted(pool['test'],key=rank)
 selected={'val':train_scenes[:4],'train':train_scenes[4:20],'test':test_scenes[:8]}
 records=[]
 for split,scenes in selected.items():
  for scene in scenes:
   d=pool['test' if split=='test' else 'train']
   for name in sorted(d[scene],key=rank)[:128]: records.append(dict(name=name,scene=scene,split=split,label=labels[name]-1))
 save(a.out/'selection.json',dict(scenes=selected,records=records,label_sha=sha(labfile),scope='spring scene-disjoint fixed subset; official held-out test scenes',selection='SHA256(seed17, scene/name); at most 128 patches/scene; no model or test score selection'))
 cache=a.out/'paired_tiffs'; cache.mkdir(exist_ok=True)
 for modality in ['s1','s2']:
  wanted={r['name'].replace('_s2_',f'_{modality}_') for r in records}; found=set()
  with tarfile.open(root/'raw_spring'/f'ROIs1158_spring_{modality}.tar.gz','r|gz') as tar:
   for member in tar:
    name=Path(member.name).name
    if name in wanted and member.isfile():
     raw=tar.extractfile(member).read(); (cache/name).write_bytes(raw); found.add(name)
     if len(found)%500==0: print('extract',modality,len(found),len(wanted),flush=True)
     if found==wanted: break
  assert found==wanted, (modality,len(wanted-found))
  print('extracted',modality,len(found),flush=True)
 result={s:([],[],[]) for s in selected}; excluded=[]
 for row in records:
  n=row['name']; paths=[cache/n.replace('_s2_','_s1_'),cache/n]
  arrays=[]
  for path in paths:
   with rasterio.open(path) as src: arrays.append(src.read().astype(np.float32))
  sar,opt=arrays
  if not np.isfinite(sar).all() or not np.isfinite(opt).all(): excluded.append(dict(row,reason='nonfinite input')); continue
  assert sar.shape[0]==2 and opt.shape[0]==13
  sar=np.clip((sar+25)/25,0,1); opt=np.clip(opt[[3,2,1,7]]/10000,0,1)
  st=F.interpolate(torch.tensor(sar)[None],(112,112),mode='bilinear',align_corners=False)[0]
  ot=F.interpolate(torch.tensor(opt)[None],(112,56),mode='bilinear',align_corners=False)[0]
  left=torch.cat([st[0],st[1]],0); right=torch.cat([torch.cat([ot[0],ot[1]],1),torch.cat([ot[2],ot[3]],1)],0)
  if left.square().sum()==0 or right.square().sum()==0: excluded.append(dict(row,reason='zero modality power')); continue
  field=torch.cat([normalize_power(left,.5),normalize_power(right,.5)],1).numpy()
  f,l,r=result[row['split']]; f.append(field); l.append(row['label']); r.append(row)
 for s,(f,l,r) in result.items(): write_split(a.out,s,f,l,r)
 save(a.out/'protocol.json',dict(task='sen12ms',classes=10,scope='spring scene-disjoint fixed subset, not full benchmark',input='SAR VV,VH clipped [-25,0]dB; S2 B4,B3,B2,B8 reflectance clipped [0,10000]; fixed packing, no CNN',power=[.5,.5],excluded=excluded,scenes=selected,label_sha=sha(labfile)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['sonyc','sen12ms'],required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 (audio if a.task=='sonyc' else sen)(a)
 save(a.out/'manifest.json',{p.name:sha(p) for p in a.out.glob('*') if p.is_file()})
if __name__=='__main__':main()
