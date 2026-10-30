"""Direct lab↔compute DATA transport; credentials only from encrypted stdin.

Host key is pinned to the existing authenticated compute connection. Credentials
are never arguments, files, profiles or output. No SDK or model is initialized.
"""
import argparse,base64,json,shlex,sys,time
from pathlib import Path
from .scene_lab_protocol import Ledger,STAGES,put
from .scene_lab_data_stream import completed_prefix,send,receive


def main():
    import paramiko
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--direction',choices=('inputs','outputs'),required=True);p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--server-ledger',required=True);a=p.parse_args()
    credential=json.loads(sys.stdin.readline());required={'host','port','username','password','host_key_type','host_key_base64','server_python','server_root'}
    if set(credential)!=required:raise ValueError('Incomplete private connection identity')
    class PinnedHostKey(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self,client,hostname,key):
            if key.get_name()!=credential['host_key_type']or key.get_base64()!=credential['host_key_base64']:raise paramiko.SSHException('Compute host-key mismatch')
    client=paramiko.SSHClient();client.set_missing_host_key_policy(PinnedHostKey());began=time.monotonic()
    c=json.loads((a.ledger/'contract.json').read_text());ledger=Ledger(a.ledger,c['session'],c['models'],c['requests']);ledger.ready(a.stage)
    def command(mode,start=0):
        args=['data','--mode',mode,'--direction',a.direction,'--stage',a.stage,'--ledger',a.server_ledger,'--wait-seconds','600','--start-index',str(start)]
        bootstrap='import sys,os;os.environ["CUDA_VISIBLE_DEVICES"]="";sys.path.insert(0,'+repr(credential['server_root'])+');from LightGenV2.tasks.t12_text_to_image.scene_lab_data_stream import main;sys.argv='+repr(args)+';main()'
        return shlex.quote(credential['server_python'])+' -u -c '+shlex.quote(bootstrap)
    try:
        client.connect(credential['host'],port=credential['port'],username=credential['username'],password=credential.pop('password'),look_for_keys=False,allow_agent=False,timeout=15,banner_timeout=15,auth_timeout=15)
        if a.direction=='inputs':
            start=completed_prefix(ledger,a.stage,'inputs');print(json.dumps(dict(direct=True,resume_at=start,stage=a.stage,direction=a.direction)),flush=True)
            si,so,se=client.exec_command(command('send',start),timeout=3600);si.close();receive(ledger,a.stage,'inputs',so,start);rc=so.channel.recv_exit_status();error=se.read().decode(errors='replace')[-1500:]
        else:
            si,so,se=client.exec_command(command('prefix'),timeout=120);si.close();prefix=json.loads(so.read());rc=so.channel.recv_exit_status()
            if rc!=0 or prefix['ledger']!=ledger.key:raise RuntimeError('Wrong compute destination prefix')
            start=prefix['next_index'];print(json.dumps(dict(direct=True,resume_at=start,stage=a.stage,direction=a.direction)),flush=True)
            si,so,se=client.exec_command(command('receive',start),timeout=3600);send(ledger,a.stage,'outputs',si,600,start);si.channel.shutdown_write();ack=so.read().decode(errors='replace')[-1500:];rc=so.channel.recv_exit_status();error=se.read().decode(errors='replace')[-1500:];print(ack,flush=True)
        if rc!=0:raise RuntimeError(('Direct data endpoint failed',rc,error))
        report=dict(complete=True,stage=a.stage,direction=a.direction,resumed_at=start,count=len(ledger.requests),elapsed_seconds=time.monotonic()-began,ledger_SHA256=ledger.key,SDK=False,GPU=False,credentials_saved=False,direct_lab_compute=True)
        path=a.ledger.parent/(a.stage+'_'+a.direction+'_direct_transfer_report.json')
        if path.exists():
            if not json.loads(path.read_text())['complete']:raise ValueError('Existing incomplete transfer report')
        else:put(path,report)
        print(json.dumps(report),flush=True)
    finally:client.close();credential.clear()


if __name__=='__main__':main()
