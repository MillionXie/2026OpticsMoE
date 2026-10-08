"""Budget-aware confirmation of a validation-selected shared configuration."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',required=True);p.add_argument('--reserve-seconds',type=int,default=600);p.add_argument('--seeds',type=int,nargs='+',default=[27,37]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    deadline=datetime.fromisoformat(a.deadline).timestamp();save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=datetime.now(timezone.utc).isoformat()))
    while not all((r/'test_lock.json').exists() for r in a.runs):
        if time.time()>deadline-a.reserve_seconds:save(a.out/'status.json',dict(state='budget_exhausted_waiting_for_candidates'));return
        time.sleep(15)
    lock=a.out/'selection_before_confirmation.json'
    subprocess.run([sys.executable,str(Path(__file__).with_name('select_adrenal_generalization.py')),'--runs',*[str(r) for r in a.runs],'--out',str(lock)],check=True)
    selected=read(lock)['selected_shared_configuration'];root=next(r for r in a.runs if r.name==selected);m=read(root/'metadata.json');results=read(root/'validation_results.json')
    # Conservative full-epoch estimate, so early stopping in seed17 does not
    # imply that another seed must stop equally early.
    estimate=30+1.15*sum(x['seconds']*m['config']['epochs']/x['epochs_completed'] for x in results)
    plan=dict(selected=selected,full_budget_estimate_seconds=estimate,confirmations=[],skipped=[],test_read=False)
    for seed in a.seeds:
        if time.time()+estimate+a.reserve_seconds>deadline:plan['skipped'].append(dict(seed=seed,reason='insufficient time for full paired budget plus verification reserve'));continue
        dest=root.parent/f'adrenal_selected_s{seed}_20260916';cmd=[sys.executable,*m['command']];cmd[cmd.index('--out')+1]=str(dest)
        if '--seeds' in cmd:
            i=cmd.index('--seeds');j=i+1
            while j<len(cmd) and not cmd[j].startswith('--'):j+=1
            cmd[i:j]=['--seeds',str(seed)]
        else:cmd+=['--seeds',str(seed)]
        plan['confirmations'].append(dict(seed=seed,run=dest.name,command=cmd,state='running'));save(a.out/'confirmation_plan.json',plan)
        subprocess.run(cmd,check=True);plan['confirmations'][-1]['state']='complete';save(a.out/'confirmation_plan.json',plan)
    save(a.out/'confirmation_plan.json',plan);save(a.out/'status.json',dict(state='complete',test_read=False))

if __name__=='__main__':main()
