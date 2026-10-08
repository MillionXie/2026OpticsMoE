"""Wait for all predeclared models, then replay locked validation and test once."""
import argparse,json,os,subprocess,sys,time,traceback,hashlib
from pathlib import Path
from datetime import datetime,timezone

def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--training-pid',type=int,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=datetime.now(timezone.utc).isoformat()))
    try:
        while True:
            status=json.loads((a.run/'status.json').read_text()) if (a.run/'status.json').exists() else {}
            if status.get('state')=='training_complete_test_not_read':break
            os.kill(a.training_pid,0);time.sleep(15)
        assert not (a.run/'test_results.json').exists();metadata=json.loads((a.run/'metadata.json').read_text());cmd=[sys.executable,*metadata['command']];cmd[cmd.index('--phase')+1]='test';save(a.out/'status.json',dict(state='replaying_and_evaluating',command=cmd));subprocess.run(cmd,check=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('verify_bloodmnist.py')),'--run',str(a.run)],check=True)
        save(a.out/'status.json',dict(state='complete',time=datetime.now(timezone.utc).isoformat()))
    except Exception:
        save(a.out/'status.json',dict(state='failed',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
