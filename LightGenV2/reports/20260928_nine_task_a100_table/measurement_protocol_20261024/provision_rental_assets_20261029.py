"""Bounded original-revision assets; SHA-verified shared weights or HTTPS mirror.

No inference/training. Public shared files are read-only candidates, not trusted
by name. Restore mismatched code from the ORIGINAL fixed Git revision only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

MODELS={'kimi':('moonshotai/Kimi-VL-A3B-Instruct','398eede0903cd983a2bfa0cc634e9ac1d843f375'),
 'deepseek':('deepseek-ai/deepseek-vl2','f363772d1c47f4239dd844015b4bd53beb87951b')}
BASE=Path('/root/data/lgvq_20261029')
KIMI_CODE_REVISIONS={
 'modeling_kimi_vl.py':'2f42821b26eed4aeab4ab594fd220deb1d762700',
 'processing_kimi_vl.py':'deca70bbda38ef23b6d255b3822045a08f026443',
 'tokenization_moonshot.py':'fd5b8b4c516172318b78a0c2dcbce74ef4f108fc'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def matches(p,spec,expected):
 return p.is_file() and p.stat().st_size==spec['size'] and sha(p)==expected
def repair_terminal_kimi(metadata_path,receipt_path,deadline):
 root=BASE/'models/kimi'
 if not (root/'failure.json').exists() or (root/'asset_receipt.json').exists():raise ValueError('Exact failed unsealed asset job required')
 if (BASE/'repo/LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/rental_fullscope_20261029_01').exists():raise ValueError('Never repair assets during/after formal suite')
 jobs=json.loads((BASE/'asset_jobs_started01.json').read_text())
 if any(Path('/proc',str(j['pid'])).exists() for j in jobs if j['model']=='kimi'):raise ValueError('Original asset job must truly exit')
 metadata=json.loads(metadata_path.read_text());old=json.loads(receipt_path.read_text());records=[]
 if metadata['revision']!=MODELS['kimi'][1] or old['revision']!=MODELS['kimi'][1]:raise ValueError('Original revision required')
 specs={s['rfilename']:s for s in metadata['assets']}
 for name,revision in KIMI_CODE_REVISIONS.items():
  if time.time()>=deadline:raise TimeoutError('Repair deadline')
  target=root/name
  if target.exists():
   if matches(target,specs[name],old['files'][name]):continue
   raise ValueError('Preserve different existing asset')
  url=f'https://modelscope.cn/models/moonshotai/Kimi-VL-A3B-Instruct/resolve/{revision}/{name}'
  response=requests.get(url,timeout=(15,45));response.raise_for_status();data=response.content
  if len(data)!=specs[name]['size'] or hashlib.sha256(data).hexdigest()!=old['files'][name]:raise ValueError('Mirror not byte-identical to original')
  with target.open('xb') as f:f.write(data)
  records.append(dict(name=name,source=url,bytes=len(data),sha256=old['files'][name]))
 for name,spec in specs.items():
  if not matches(root/name,spec,old['files'][name]):raise ValueError('Remaining original asset missing/different: '+name)
 repair=dict(status='passed_original_SHA_different_mirror',records=records,original_failure_preserved=True,completed=time.time())
 with (root/'mirror_repair_receipt01.json').open('x') as f:json.dump(repair,f,ensure_ascii=True,indent=2)
 with (root/'asset_receipt.json').open('x') as f:json.dump(dict(old,repair_receipt='mirror_repair_receipt01.json',all_original_SHA_verified=True),f,ensure_ascii=True,indent=2)
 print(json.dumps(repair,ensure_ascii=True))
def provision(model,metadata_path,receipt_path,deadline):
 if not 0<deadline-time.time()<=7200:raise ValueError('Finite at most two-hour asset deadline')
 repo,revision=MODELS[model];metadata=json.loads(metadata_path.read_text());receipt=json.loads(receipt_path.read_text())
 if metadata['revision']!=revision or receipt['revision']!=revision:raise ValueError('Original immutable revision required')
 specs=metadata['assets'];expected=receipt['files']
 if {s['rfilename'] for s in specs}!=set(expected):raise ValueError('Original exact file set required')
 root=BASE/'models'/model;root.mkdir(parents=True,exist_ok=False)
 (root/'official_metadata.json').write_bytes(metadata_path.read_bytes())
 records=[]
 def one(spec):
  name=spec['rfilename']
  if Path(name).name!=name:raise ValueError('Top-level official asset only')
  target=root/name;digest=expected[name]
  shared=Path('/public/huggingface-models')/repo/name
  if matches(shared,spec,digest):
   target.symlink_to(shared);return dict(name=name,sha256=digest,bytes=spec['size'],source='read-only shared SHA verified')
  # An alternative mirror is accepted ONLY by the original fixed SHA/size.
  url=(f'https://modelscope.cn/models/deepseek-ai/deepseek-vl2/resolve/90fc6b481ffa9987faf062e4fcdd2ebd85077d64/{name}' if model=='deepseek'
       else f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}')
  part=root/(name+'.partial')
  for attempt in range(3):
   if time.time()>=deadline:raise TimeoutError('Asset deadline')
   offset=part.stat().st_size if part.exists() else 0
   headers={'Range':f'bytes={offset}-'} if offset else {}
   try:
    with requests.get(url,headers=headers,stream=True,timeout=(20,60)) as r:
     r.raise_for_status()
     if offset and (r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise ValueError('Resume range rejected')
     with part.open('ab' if offset else 'xb') as f:
      for b in r.iter_content(4194304):
       if time.time()>=deadline:raise TimeoutError('Asset deadline')
       if b:f.write(b)
       if f.tell()>spec['size']:raise ValueError('Asset overflow')
    break
   except requests.RequestException:
    if attempt==2:raise
    time.sleep(2*(attempt+1))
  if not matches(part,spec,digest):raise ValueError('Original SHA/size mismatch: '+name)
  part.rename(target)
  return dict(name=name,sha256=digest,bytes=spec['size'],source=url)
 try:
  with ThreadPoolExecutor(max_workers=2) as pool:
   for result in pool.map(one,specs):
    records.append(result)
    (root/'progress.json').write_text(json.dumps(dict(files=records,updated=time.time()),ensure_ascii=True))
  (root/'asset_receipt.json').write_text(json.dumps(dict(status='complete',model=repo,revision=revision,files=expected,transport_records=records,completed=time.time()),ensure_ascii=True,indent=2))
 except BaseException as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),time=time.time()),ensure_ascii=True));raise
 print(json.dumps(dict(status='complete',model=model,files=len(records)),ensure_ascii=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--repair-terminal-kimi',action='store_true')
 a=p.parse_args()
 if a.repair_terminal_kimi:
  if a.model!='kimi':raise ValueError('Only exact three-code Kimi mirror repair')
  repair_terminal_kimi(a.metadata,a.receipt,a.deadline)
 else:provision(a.model,a.metadata,a.receipt,a.deadline)
