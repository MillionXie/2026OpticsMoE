"""Build an auditable SONYC-UST query index from official annotations."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
EVENTS=['1_engine_presence','2_machinery-impact_presence','3_non-machinery-impact_presence','4_powered-saw_presence','5_alert-signal_presence','6_music_presence','7_human-voice_presence','8_dog_presence']
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--audio-dir',type=Path);a=ap.parse_args(); ann=a.root/'annotations.csv'; assert ann.is_file(),ann
 audio={p.name for p in a.audio_dir.rglob('*.wav')} if a.audio_dir else None
 rows=[]; counts={}; by_event={e:0 for e in EVENTS}
 with ann.open(newline='',encoding='utf8') as f:
  for r in csv.DictReader(f):
   if audio is not None and r['audio_filename'] not in audio: continue
   split='val' if r['split']=='validate' else r['split']
   for e in EVENTS:
    v=r.get(e,'-1')
    if v not in ('0','1'): continue
    rows.append({'split':split,'audio_filename':r['audio_filename'],'sensor_id':r['sensor_id'],'event':e,'label':int(v),'unknown_excluded':True})
    counts[split]=counts.get(split,0)+1; by_event[e]+=int(v)
 out={'schema_version':1,'source_annotation':str(ann),'annotation_sha256':sha(ann),'audio_dir':str(a.audio_dir) if a.audio_dir else None,'audio_files':len(audio) if audio is not None else None,'events':EVENTS,'query_rows':len(rows),'split_query_counts':counts,'positive_query_counts':by_event,'rows':rows}
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); print(json.dumps({k:out[k] for k in ['query_rows','split_query_counts','audio_files']},ensure_ascii=False))
if __name__=='__main__':main()
