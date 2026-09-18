"""Materialize a reproducible SONYC audio/text matching subset for the optical trainer."""
import argparse,json,wave
from pathlib import Path
import numpy as np
from LightGenV2.tasks.t09_multimodal_matching.audio_prepare import logmel
from LightGenV2.tasks.t09_multimodal_matching.prepare import save,tokens,digest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--audio-dir',type=Path,required=True);ap.add_argument('--query-index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
 q=json.loads(a.query_index.read_text())['rows']; by={}
 for r in q: by.setdefault(r['audio_filename'],[]).append(r)
 records={k:sorted({r['audio_filename'] for r in q if r['split']==k}) for k in ['train','val','test']}
 allfiles={p.name:p for p in a.audio_dir.rglob('*.wav')}; assert all(x in allfiles for xs in records.values() for x in xs)
 vocab={'<pad>':0,'<unk>':1}; rows_by={}; imgs_by={}
 for split,names in records.items():
  images=[]; rows=[]
  for i,name in enumerate(names):
   with wave.open(str(allfiles[name]),'rb') as w:
    assert w.getnchannels()==1 and w.getsampwidth()==2
    rate=w.getframerate()
    samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').copy()
    assert rate in (16000,48000)
    if rate==48000: samples=samples[::3]
    samples=samples[:160000]
    samples=np.pad(samples,(0,max(0,160000-len(samples))))
    chunks=np.array_split(samples,10)
    image=np.mean([logmel(c[:16000]) for c in chunks],axis=0).astype(np.uint8)
   images.append(np.repeat(image[:,:,None],3,axis=2))
   for base in by[name]:
    if base['split']!=split: continue
    words=['does','the','recording','contain',base['event'].split('_')[1], '?']
    rows.append(dict(image_local=i,image_id=name,speaker=base['sensor_id'],question=' '.join(words),label=base['label'],event=base['event']))
    if split=='train':
     for w in tokens(rows[-1]['question']): vocab.setdefault(w,len(vocab))
  rows_by[split]=rows; imgs_by[split]=images
  if split!='test': np.savez_compressed(a.out/f'{split}_images.npz',images=np.stack(images)); save(a.out/f'{split}_questions.json',rows)
  else: save(a.out/'test_questions.json',rows)
  save(a.out/f'{split}_audio_records.json',[{'source_member':n,'audio_filename':n} for n in names])
 save(a.out/'vocab.json',vocab)
 payload={'schema_version':1,'source_query_index_sha256':digest(a.query_index.read_bytes()),'counts':{k:{'audio':len(records[k]),'queries':len(rows_by[k])} for k in records},'manifest_files':{p.name:digest(p.read_bytes()) for p in a.out.iterdir() if p.is_file()}}
 save(a.out/'manifest.json',payload); print(json.dumps(payload,ensure_ascii=False))
if __name__=='__main__':main()

