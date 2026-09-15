"""Resumable, checksum-verified download of the authors' original archives."""
import argparse,concurrent.futures,hashlib,json,os,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.netdeps'))
import requests
SOURCES={
 'rgb':('https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1','EuroSAT_RGB.zip',94658721,'md5','f46e308c4d50d4bf32fedad2d3d62f3b'),
 'ms':('https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1','EuroSAT_MS.zip',2065402329,'md5','091174add3c8e680a49244acf185b9f0'),
 'sar':('https://huggingface.co/datasets/wangyi111/EuroSAT-SAR/resolve/main/EuroSAT-SAR.zip?download=true','EuroSAT-SAR.zip',922267666,'sha256','3ffad1e8bb3c2b0c2941ed60b978d78f43a77c9b3bddd6fe35c34ab655459c10')}
def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)
def digest(path,algorithm):
 h=hashlib.new(algorithm)
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def download(name,cache,workers):
 url,filename,size,alg,expected=SOURCES[name];dest=cache/filename;state=ROOT/f'download_{name}.json';t=time.time()
 if dest.exists():
  assert dest.stat().st_size==size and digest(dest,alg)==expected
  atomic(state,dict(state='complete',file=str(dest),bytes=size,checksum=expected,algorithm=alg));return
 part=dest.with_suffix('.zip.part');donefile=dest.with_suffix('.parts.json');block=4*1024*1024
 done=json.loads(donefile.read_text()) if donefile.exists() and part.exists() else {}
 if not part.exists():
  with part.open('wb') as f:f.truncate(size)
 assert part.stat().st_size==size
 lock=threading.Lock();local=threading.local();last=0
 def chunk(a):
  nonlocal last
  b=min(size,a+block)-1;key=str(a)
  if key in done:
   with part.open('rb') as f:f.seek(a);payload=f.read(b-a+1)
   if hashlib.sha256(payload).hexdigest()==done[key]:return
  if not hasattr(local,'session'):local.session=requests.Session()
  for attempt in range(8):
   try:
    # Distinct resource query avoids intermediary caches mixing byte ranges.
    r=local.session.get(url+f'&range_start={a}',headers={'Range':f'bytes={a}-{b}'},timeout=(20,70))
    r.raise_for_status()
    assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {a}-{b}/{size}'
    assert len(r.content)==b-a+1
    with part.open('r+b') as f:f.seek(a);f.write(r.content)
    with lock:
     done[key]=hashlib.sha256(r.content).hexdigest()
     if time.time()-last>10 or len(done)==(size+block-1)//block:
      atomic(donefile,done);n=sum(min(block,size-int(k)) for k in done)
      status=dict(state='downloading',archive=name,bytes=n,total=size,elapsed_sec=time.time()-t,updated_at=time.time())
      atomic(state,status);print(json.dumps(status),flush=True);last=time.time()
    return
   except Exception:
    if attempt==7:raise
    time.sleep(min(12,2**attempt))
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(chunk,range(0,size,block)))
  actual=digest(part,alg);assert actual==expected,(name,actual,expected)
  os.replace(part,dest)
  atomic(state,dict(state='complete',file=str(dest),bytes=size,checksum=actual,algorithm=alg,sha256=digest(dest,'sha256'),elapsed_sec=time.time()-t))
 except BaseException as e:atomic(state,dict(state='failed',archive=name,error=repr(e)));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('names',nargs='+',choices=SOURCES);p.add_argument('--cache',required=True);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
 cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
 for name in args.names:download(name,cache,args.workers)
