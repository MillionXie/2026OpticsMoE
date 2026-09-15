"""Seal every chosen checkpoint before any full-run test forward."""
import hashlib,json,math
import numpy as np
import torch
from .data import prepare,plan,signature,Images,atomic_json
from .runtime import ROOT,setup,build,verify_source,require_scope
from experiments.vision_transfer import model as M,engine as E

def mechanisms(loaded,r,h,s,split,folder):
    final=M.clone(r,h);vb=Images(split,1,'validation');va=Images(split,0,'validation')
    normal={d:E.evaluate(loaded,r,h,ds,s) for d,ds in [('A',va),('B',vb)]};out={'normal':normal}
    before=torch.load(folder/'initial_phases.pt',map_location='cpu',weights_only=False)
    with torch.no_grad():
        for n,p in M.named(r,h).items():
            if M.group_of(n)=='expert_b':p.copy_(before[n].to(p.device))
    out['reset_reserved_B_phase']={d:E.evaluate(loaded,r,h,ds,s) for d,ds in [('A',va),('B',vb)]}
    M.restore(r,h,final);router=r.vision_surrogate.core.optical_branch.core.router;router.uniform_ablation=True
    try:out['uniform_amplitude']={d:E.evaluate(loaded,r,h,ds,s) for d,ds in [('A',va),('B',vb)]}
    finally:router.uniform_ablation=False;M.restore(r,h,final)
    out['B_accuracy_drop_on_phase_reset']=normal['B']['accuracy']-out['reset_reserved_B_phase']['B']['accuracy']
    atomic_json(folder/'mechanisms_validation.json',out);return out

def evaluate(mode='full',mechanism_only=False):
    verify_source();scope=require_scope(mode);split=prepare();cfg=plan();root=ROOT/'runs'/mode
    names=tuple(scope['allowed_stages'])
    checkpoints={str(root/('seed'+str(seed))/n/'selected.pt'):hashlib.sha256((root/('seed'+str(seed))/n/'selected.pt').read_bytes()).hexdigest() for seed in cfg[mode]['seeds'] for n in names}
    seal=dict(checkpoints=checkpoints,plan_sha256=signature(cfg),split_sha256=split['split_sha256'])
    if not mechanism_only:
        if mode!='full':raise RuntimeError('Pilot test remains sealed')
        dest=root/'test_seal.json'
        if dest.exists() and json.loads(dest.read_text())!=seal:raise RuntimeError('Checkpoint seal changed')
        atomic_json(dest,seal)
    rows=[]
    for seed in cfg[mode]['seeds']:
        for name in (tuple(n for n in names if n in ('M1','M2')) if mechanism_only else names):
            folder=root/('seed'+str(seed))/name;arch='d2nn' if name.startswith('D') else 'moe'
            loaded,s=setup(seed,folder);r,h=build(loaded,s,arch)
            M.restore(r,h,torch.load(folder/'selected.pt',map_location='cpu',weights_only=False));digest=M.digest(r,h)
            if name in ('M1','M2'):mechanisms(loaded,r,h,s,split,folder)
            if not mechanism_only:
                result={domain:E.evaluate(loaded,r,h,Images(split,d,'test'),s,predictions_path=folder/('test_'+domain+'.npz')) for d,domain in enumerate(('A','B'))}
                assert digest==M.digest(r,h),'Evaluation mutated the fixed model'
                # Interleave A/B and reverse batch order without passing domain IDs.
                ims=[Images(split,d,'test')[i][0] for i in range(5) for d in (0,1)]
                with torch.no_grad(),E.autocast(loaded,s):
                    one=M.predict(loaded,r,h,E.base._prepare(loaded,ims,s)).float();two=M.predict(loaded,r,h,E.base._prepare(loaded,list(reversed(ims)),s)).float().flip(0)
                torch.testing.assert_close(one,two,atol=.02,rtol=.002);assert torch.equal(one.argmax(1),two.argmax(1))
                atomic_json(folder/'test.json',dict(**result,checkpoint_sha256=checkpoints[str(folder/'selected.pt')],mixed_order_invariant=True))
                rows.append(dict(seed=seed,arm=name,A=result['A']['accuracy'],B=result['B']['accuracy'],macro_f1_A=result['A']['macro_f1'],macro_f1_B=result['B']['macro_f1']))
            r.close();del r,h,loaded;torch.cuda.empty_cache()
    if not mechanism_only:
        for row in rows:
            source=next(r for r in rows if r['seed']==row['seed'] and r['arm']==('D_A' if row['arm'].startswith('D') else 'A'))
            single=next(r for r in rows if r['seed']==row['seed'] and r['arm']==('D_B_only' if row['arm'].startswith('D') else 'B_only'))
            row.update(mean=(row['A']+row['B'])/2,forgetting=source['A']-row['A'],b_gap=single['B']-row['B'],b_gain=row['B']-source['B'])
        aggregate={name:{metric:dict(mean=float(np.mean([r[metric] for r in rows if r['arm']==name])),std=float(np.std([r[metric] for r in rows if r['arm']==name],ddof=1))) for metric in ('A','B','mean','forgetting','b_gap','b_gain')} for name in names}
        atomic_json(root/'test_summary.json',dict(partition='test',rows=rows,aggregate=aggregate,seal=seal))
        lines=['# Office-Home test: three seeds','', '| arm | A % (mean ± sd) | B % (mean ± sd) | equal-domain mean % |','|---|---:|---:|---:|']
        for name in names:lines.append('| '+name+' | '+' | '.join(f"{aggregate[name][k]['mean']*100:.2f} ± {aggregate[name][k]['std']*100:.2f}" for k in ('A','B','mean'))+' |')
        (root/'test_performance.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--pilot-mechanisms',action='store_true');a=p.parse_args()
    evaluate('pilot' if a.pilot_mechanisms else 'full',a.pilot_mechanisms)
