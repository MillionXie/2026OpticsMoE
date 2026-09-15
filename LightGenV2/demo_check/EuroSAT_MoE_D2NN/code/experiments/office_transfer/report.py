import json
from pathlib import Path
import numpy as np
from .data import plan,atomic_json
from .runtime import ROOT

def summarize(mode,names=('A','B_only','M1','M2','D_A','D_B_only','D1')):
    cfg=plan();root=ROOT/'runs'/mode;rows=[];gates=[]
    for seed in cfg[mode]['seeds']:
        results={n:json.loads((root/('seed'+str(seed))/n/'final.json').read_text()) for n in names}
        for n,r in results.items():
            source=results['D_A' if n.startswith('D') else 'A'];single=results['D_B_only' if n.startswith('D') else 'B_only']
            a=r['validation_a']['accuracy'];b=r['validation_b']['accuracy'];transfer=n in ('M1','M2','D1')
            rows.append(dict(seed=seed,arm=n,A=a,B=b,mean=(a+b)/2,
                forgetting=source['validation_a']['accuracy']-a if transfer else None,
                b_gap=single['validation_b']['accuracy']-b if transfer else None,
                b_gain=b-source['validation_b']['accuracy'] if transfer else None,
                train_samples=r['total_train_samples'],seconds=r['total_epoch_seconds'],checkpoint=r['checkpoint']))
        m1=next(r for r in rows if r['seed']==seed and r['arm']=='M1')
        gates.append(results['A']['validation_a']['accuracy']>=cfg['pilot_gate']['single_domain_validation_accuracy'] and results['B_only']['validation_b']['accuracy']>=cfg['pilot_gate']['single_domain_validation_accuracy'] and m1['forgetting']<=cfg['targets']['maximum_forgetting'] and m1['b_gap']<=cfg['targets']['maximum_b_baseline_gap'] and m1['b_gain']>cfg['pilot_gate']['minimum_b_gain'])
    aggregate={arm:{metric:dict(mean=float(np.mean([r[metric] for r in rows if r['arm']==arm])),std=float(np.std([r[metric] for r in rows if r['arm']==arm],ddof=1)) if len(cfg[mode]['seeds'])>1 else None) for metric in ('A','B','mean')} for arm in names}
    result=dict(mode=mode,partition='validation',test_used=False,rows=rows,aggregate=aggregate,pilot_gate_passed=all(gates),targets=cfg['targets'],
        next_action='Report M1 performance and wait for user confirmation; no automatic D1, M2 or full training.',
        budget_note='B-only receives B-only source training; transfer also receives source pretraining and 50% replay. Sample and time budgets are reported, not claimed identical.')
    atomic_json(root/'summary.json',result)
    text=['# Office-Home validation results','', '| seed | arm | A % | B % | mean % | forgetting pp | B gap pp | B gain pp |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append('| '+str(r['seed'])+' | '+r['arm']+' | '+' | '.join('—' if r[k] is None else f'{100*r[k]:.2f}' for k in ('A','B','mean','forgetting','b_gap','b_gain'))+' |')
    (root/'performance.md').write_text('\n'.join(text)+'\n',encoding='utf-8');return result
