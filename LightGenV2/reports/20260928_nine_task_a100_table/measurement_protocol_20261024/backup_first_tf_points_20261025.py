"""Serial private-transport backup of completed supplement points, no inference."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO/'.codex_tmp/energy_demo'))
from ssh_connection import connect
ROOT='LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/a100_first_tf_seven_20261025_01'
PY='C:/Users/Xml12/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',default=ROOT);p.add_argument('--expected-count',type=int,default=21);args=p.parse_args()
    root=args.suite
    previous=None
    while True:
        client=connect()
        try:
            with client.open_sftp() as ftp:
                with ftp.open('/DATA/DATA1/guest3/2026OpticsMoE/'+root+'/suite_status.json') as f:s=json.load(f)
        finally:client.close()
        marker=(s['status'],s.get('phase'),len(s['completed']))
        if marker!=previous:print(json.dumps(dict(status=s['status'],phase=s.get('phase'),completed=len(s['completed']))),flush=True);previous=marker
        for point in s['completed']:
            relative=point['folder'].split('/2026OpticsMoE/',1)[1];local=REPO/relative
            if not (local/'backup_receipt_complete01.json').exists():
                subprocess.run([sys.executable,str(REPO/'.codex_tmp/energy_demo/backup_completed_data.py'),relative,'backup_receipt_complete01.json'],cwd=REPO,check=True,timeout=240)
            if not (local/'local_first_tf_independent_audit01.json').exists():
                subprocess.run([PY,str(HERE/'audit_first_tf_delivery_20261025.py'),'--point',str(local)],cwd=REPO,check=True,timeout=120)
        if s['status'] in ('complete','stopped'):
            assert not (REPO/root/'backup_receipt_suite01.json').exists()
            subprocess.run([sys.executable,str(REPO/'.codex_tmp/energy_demo/backup_completed_data.py'),root,'backup_receipt_suite01.json'],cwd=REPO,check=True,timeout=240)
            if s['status']!='complete':raise RuntimeError(s.get('error','Stopped supplement'))
            assert len(s['completed'])==args.expected_count;print('All missing points backed up and independently audited.',flush=True);return
        time.sleep(30)
if __name__=='__main__':main()
