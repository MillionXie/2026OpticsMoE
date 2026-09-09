"""Collect immutable suite/allocation audits using independent remote SHA256."""
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import paramiko


def main():
    local=Path(__file__).resolve().parent/'reports/unseen_v1_20260909'
    local.mkdir(parents=True,exist_ok=True)
    remote='/DATA/DATA1/guest3/2026OpticsMoE/TransferFromElectricity/tasks/t01_object_retrieval/runs'
    files=[('smoke','20260909_unseen_smoke_a_execution.json'),
           ('simulation','20260909_unseen_formal_a_execution.json'),
           ('simulation','20260909_unseen_formal_b_execution.json'),
           ('simulation','20260909_unseen_formal_a_allocation.json')]
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.connect('202.120.62.181',port=24096,username='guest3',password=getpass.getpass('SSH password: '))
    receipts=[]
    try:
        sftp=client.open_sftp()
        for kind,name in files:
            source=f'{remote}/{kind}/{name}'
            with sftp.open(source,'rb') as f: content=f.read()
            parsed=json.loads(content)
            if name.endswith('_execution.json'):
                assert parsed['complete'] and all(r['returncode']==0 for r in parsed['records'])
            _,out,err=client.exec_command('sha256sum -- '+shlex.quote(source))
            checksum=out.read().decode().split()[0]
            if out.channel.recv_exit_status()!=0:raise RuntimeError(err.read().decode())
            assert hashlib.sha256(content).hexdigest()==checksum
            target=local/name
            if target.exists():assert target.read_bytes()==content
            else:target.write_bytes(content)
            receipts.append({'file':name,'source':source,'sha256':checksum,'bytes':len(content)})
        (local/'execution_transfer.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
        print('Four suite/allocation audits independently verified')
    finally:client.close()


if __name__=='__main__':main()
