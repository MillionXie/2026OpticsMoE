"""Fetch completed lightweight run artifacts with independent remote SHA256 checks.

Passwords are prompted interactively and never written to the workspace.
Source code and best/last checkpoints are never transferred by this command.
"""
import argparse
import getpass
import hashlib
import json
import shlex
from pathlib import Path, PurePosixPath
import paramiko


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--host',required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--user',required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--runs',nargs='+',required=True,help='Paths relative to this task directory')
    args=parser.parse_args()
    task=Path(__file__).resolve().parent
    remote_task=PurePosixPath(args.remote_root)/'TransferFromElectricity/tasks/t01_object_retrieval'
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(args.host,port=args.port,username=args.user,password=getpass.getpass('SSH password: '))
    try:
        sftp=client.open_sftp()
        for name in args.runs:
            relative=PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='runs':
                raise ValueError('Only task-relative runs paths are supported')
            remote=remote_task/relative
            with sftp.open(str(remote/'status.json'),'rb') as f:
                if json.loads(f.read())['status']!='complete':
                    raise ValueError(f'Incomplete run: {relative}')
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
        sftp.close()
    finally:
        client.close()


if __name__=='__main__':
    main()
