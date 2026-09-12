"""Pull a SHA-pinned asset ZIP over SSH and install without overwriting different files."""
import argparse,hashlib,json,os,time,zipfile
from pathlib import Path
import paramiko

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=22)
    p.add_argument('--user',required=True);p.add_argument('--remote',required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args();root=Path(__file__).resolve().parent;dest=root/'transfers/inference_assets_direct.zip'
    partial=dest.with_suffix('.partial');dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.WarningPolicy())
        try:
            c.connect(a.host,port=a.port,username=a.user,password=os.environ['SHS_SOURCE_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=20)
            with c.open_sftp() as s:
                mark=[-1];started=time.monotonic()
                def progress(done,total):
                    value=done//(64*1024**2)
                    if value>mark[0]:print(f'{done}/{total}, {done/max(time.monotonic()-started,.001)/1024**2:.1f} MiB/s',flush=True);mark[0]=value
                if partial.exists():raise FileExistsError('Inspect previous incomplete direct download')
                s.get(a.remote,str(partial),callback=progress,max_concurrent_prefetch_requests=32)
            if digest(partial)!=a.sha256:raise ValueError('Downloaded archive SHA mismatch')
            partial.rename(dest)
        finally:c.close()
    if digest(dest)!=a.sha256:raise ValueError('Archive SHA mismatch')
    with zipfile.ZipFile(dest) as z:
        mf=json.loads(z.read('INFERENCE_ASSET_MANIFEST.json'))
        for rel,expected in mf['files'].items():
            target=(root/rel).resolve()
            if not target.is_relative_to(root) or ':' in rel:raise ValueError('Unsafe archive path')
            if target.exists() and digest(target)!=expected:raise FileExistsError(str(target))
        for rel,expected in mf['files'].items():
            target=root/rel
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(rel) as src,target.open('xb') as dst:
                    while b:=src.read(1024**2):dst.write(b)
            if digest(target)!=expected:raise ValueError('Extracted SHA mismatch: '+rel)
        (root/'INFERENCE_ASSET_MANIFEST.json').write_text(json.dumps(mf,indent=2),encoding='utf-8')
    print('Verified inference asset files:',len(mf['files']),flush=True)

if __name__=='__main__':main()
