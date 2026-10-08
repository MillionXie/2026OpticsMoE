"""After confirmation ends, seal every candidate before any test evaluation."""
import argparse,hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path

def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--reference-runs',type=Path,nargs='*',default=[]);p.add_argument('--controller-run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=datetime.now(timezone.utc).isoformat()))
    while not (a.controller_run/'status.json').exists():time.sleep(10)
    assert read(a.controller_run/'status.json')['state']=='complete','Confirmation controller did not finish normally'
    plan=read(a.controller_run/'confirmation_plan.json');confirmations=[a.runs[0].parent/x['run'] for x in plan['confirmations'] if x['state']=='complete'];lock=a.out/'selection_lock.json';here=Path(__file__).parent
    try:
        subprocess.run([sys.executable,str(here/'select_adrenal_generalization.py'),'--runs',*[str(r) for r in a.runs],'--confirmation-runs',*[str(r) for r in confirmations],'--reference-runs',*[str(r) for r in a.reference_runs],'--out',str(lock)],check=True)
        for root in a.runs+confirmations+a.reference_runs:
            cmd=[sys.executable,str(here/'evaluate_adrenal_generalization.py'),'--run',str(root),'--data',str(a.data),'--selection-lock',str(lock)]
            if root in a.reference_runs:cmd+=['--historical-reference']
            save(a.out/'status.json',dict(state='evaluating_locked_candidates',run=root.name));subprocess.run(cmd,check=True)
        save(a.out/'status.json',dict(state='complete',time=datetime.now(timezone.utc).isoformat(),runs=[r.name for r in a.runs+confirmations],references=[r.name for r in a.reference_runs]))
    except Exception:
        save(a.out/'status.json',dict(state='failed',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
