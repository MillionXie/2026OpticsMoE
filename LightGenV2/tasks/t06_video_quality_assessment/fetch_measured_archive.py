"""Resumable read-only SFTP acquisition transfer; password is environment-only."""
import argparse
import os
from pathlib import Path
import time

from .lab_runtime import sha, write


def main():
    import paramiko
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('host','user','remote','output','sha256'):p.add_argument('--'+key,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--bytes',type=int,required=True)
    a=p.parse_args();dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    receipt=dest.with_suffix('.transfer.json')
    for attempt in range(1,4):
        client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.WarningPolicy())
        try:
            offset=dest.stat().st_size if dest.exists() else 0
            if offset>a.bytes:raise ValueError('Existing destination exceeds source size')
            if offset<a.bytes:
                client.connect(a.host,port=a.port,username=a.user,password=os.environ['SHS_SSH_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=30)
                client.get_transport().set_keepalive(15)
                with client.open_sftp() as s:
                    if s.stat(a.remote).st_size!=a.bytes:raise ValueError('Source size mismatch')
                    with s.open(a.remote,'rb') as source, dest.open('ab') as target:
                        source.settimeout(120);source.seek(offset)
                        source.prefetch(file_size=a.bytes,max_concurrent_requests=32)
                        last=0
                        while offset<a.bytes:
                            chunk=source.read(min(262144,a.bytes-offset))
                            if not chunk:raise EOFError('Incomplete remote source')
                            target.write(chunk);offset+=len(chunk)
                            if time.monotonic()-last>15:
                                target.flush();last=time.monotonic()
                                write(receipt,dict(state='transferring',bytes=offset,total=a.bytes,attempt=attempt))
                                print('TRANSFER',offset,'/',a.bytes,flush=True)
            actual=sha(dest)
            if actual!=a.sha256:raise ValueError('SHA256 mismatch; preserve file for diagnosis')
            write(receipt,dict(state='verified',bytes=a.bytes,sha256=actual,remote=a.remote))
            print('VERIFIED',actual,flush=True);return
        except Exception as e:
            write(receipt,dict(state='failed' if attempt==3 else 'retrying',error=repr(e),attempt=attempt))
            if attempt==3 or isinstance(e,ValueError):raise
            time.sleep(5)
        finally:client.close()


if __name__=='__main__':main()
