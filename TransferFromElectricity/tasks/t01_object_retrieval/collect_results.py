"""Fetch completed lightweight run artifacts with independent remote SHA256 checks.

Passwords are prompted interactively and never written to the workspace.
Source code and best/last checkpoints are never transferred by this command.
"""
import argparse
import getpass
import hashlib
import json
import shlex
import time
from pathlib import Path, PurePosixPath
import paramiko


def remote_state(sftp, remote, require_gpu_audit):
    def read(name):
        with sftp.open(str(remote/name),'rb') as stream:return json.loads(stream.read())
    try:status=read('status.json')['status']
    except (FileNotFoundError,json.JSONDecodeError):return 'waiting',0
    if status=='failed':raise RuntimeError(f'Remote training failed: {remote}')
    try:epoch=len(read('history.json'))
    except (FileNotFoundError,json.JSONDecodeError):epoch=0
    if status=='complete' and require_gpu_audit:
        try:audit=read('gpu_execution.json')
        except (FileNotFoundError,json.JSONDecodeError):return 'waiting_gpu_audit',epoch
        if audit['status']!='complete' or audit['returncode']!=0:
            raise RuntimeError(f'Remote GPU execution failed: {remote}')
    return status,epoch


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--host',required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--user',required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--runs',nargs='+',required=True,help='Paths relative to this task directory')
    parser.add_argument('--wait-seconds',type=int,default=0,help='Poll incomplete runs every 1-60 seconds; zero fails immediately')
    parser.add_argument('--require-gpu-audit',action='store_true')
    args=parser.parse_args()
    if not 0<=args.wait_seconds<=60:raise ValueError('wait-seconds must be between zero and 60')
    task=Path(__file__).resolve().parent
    remote_task=PurePosixPath(args.remote_root)/'TransferFromElectricity/tasks/t01_object_retrieval'
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(args.host,port=args.port,username=args.user,password=getpass.getpass('SSH password: '))
    client.get_transport().set_keepalive(30)
    try:
        sftp=client.open_sftp()
        pending=[]
        for name in dict.fromkeys(args.runs):
            relative=PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='runs':
                raise ValueError('Only task-relative runs paths are supported')
            pending.append(relative)
        previous={}
        while pending:
          for relative in list(pending):
            remote=remote_task/relative
            state=remote_state(sftp,remote,args.require_gpu_audit)
            if state!=previous.get(relative):
                print(f'{relative}: {state[0]}, epoch {state[1]}',flush=True);previous[relative]=state
            if state[0]!='complete':
                if not args.wait_seconds:raise ValueError(f'Incomplete run: {relative}')
                continue
            destination=task/Path(*relative.parts)
            destination.mkdir(parents=True,exist_ok=True)
            receipt=[]
            for filename in sorted(sftp.listdir(str(remote))):
                if Path(filename).suffix not in {'.json','.yaml','.csv'} and filename!='expert_bank.pt':
                    continue
                source=str(remote/filename)
                _,stdout,stderr=client.exec_command('sha256sum -- '+shlex.quote(source))
                checksum=stdout.read().decode().split()[0]
                if stdout.channel.recv_exit_status()!=0:
                    raise RuntimeError(stderr.read().decode())
                with sftp.open(source,'rb') as f: content=f.read()
                if hashlib.sha256(content).hexdigest()!=checksum:
                    raise RuntimeError(f'Transfer checksum mismatch: {source}')
                target=destination/filename
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=checksum:
                    raise FileExistsError(f'Refusing to replace different artifact: {target}')
                target.write_bytes(content)
                receipt.append({'source':source,'file':filename,'sha256':checksum,'bytes':len(content)})
            (destination/'transfer_manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
            print(f'{relative}: {len(receipt)} artifacts verified',flush=True)
            pending.remove(relative)
          if pending:time.sleep(args.wait_seconds)
        sftp.close()
    finally:
        client.close()


if __name__=='__main__':
    main()
