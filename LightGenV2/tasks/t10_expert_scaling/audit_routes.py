"""Export an auditable table of hard expert load and soft router probability."""
import argparse
import csv
import json
import math
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False); rows=[]
    for path in sorted(a.runs.glob('*/result.json')):
        x=json.loads(path.read_text()); v=x.get('val',{}); load=v.get('route_load'); prob=v.get('route_probability')
        if not load: continue
        rows.append(dict(run=path.parent.name,experts=len(load),best_epoch=x.get('best_epoch'),validation_accuracy=v.get('accuracy'),validation_macro_nll=v.get('macro_nll'),hard_load=load,soft_probability=prob,dead_hard_experts=sum(p==0 for p in load),max_hard_load=max(load),min_hard_load=min(load),soft_entropy=-sum(p*math.log(max(p,1e-12)) for p in prob),distinct_selected_sets=v.get('distinct_selected_sets')))
    (a.out/'route_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    cols=['run','experts','best_epoch','validation_accuracy','validation_macro_nll','dead_hard_experts','max_hard_load','min_hard_load','soft_entropy','distinct_selected_sets']
    with (a.out/'route_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows({k:r[k] for k in cols} for r in rows)
    print(f'exported {len(rows)} route summaries')

if __name__=='__main__': main()
