"""Verify the retrospective fixed-epoch diagnostic from per-sample predictions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from verify_adrenal_regularized import read, rows, sha, verify_predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    sim=a.task/'runs/simulation'
    parent=sim/'adrenal_regularized_s17_20260916'
    baseline=read(sim/'adrenal_depth_audit_20260916/results.json')
    baseline={r['variant']:r for r in baseline}
    regularized={r['variant']:r for r in read(parent/'validation_results.json')}
    selected={r['variant']:r for r in read(parent/'independent_verification.json')['results']}
    checked={}
    for name in ['original','regularized']:
        root=sim/f'adrenal_{name}_fixed50_audit_20260916'
        lock=read(root/'checkpoint_lock.json')['checkpoints']
        records=read(root/'results.json')
        assert len(records)==12 and len(lock)==12
        checked[name]={}
        for r in records:
            v=r['variant'];dest=root/v
            assert r['epoch']==50 and r['weights_unchanged'] and r['seed']==17
            expected_sha=(baseline[v]['checkpoint_sha256']['last'] if name=='original'
                          else regularized[v]['last_checkpoint_sha256'])
            assert r['checkpoint_sha256']==expected_sha and expected_sha in lock.values()
            vr,vy,vp=verify_predictions(dest/'validation_predictions.csv',r['val'],[76,22])
            tr,ty,tp=verify_predictions(dest/'test_predictions.csv',r['test'],[229,69])
            assert [(x['sample_id'],x['label_true']) for x in tr] == [
                (x['sample_id'],x['label_true']) for x in rows(parent/'runs'/v/'seed17'/'test_predictions.csv')]
            score=np.unique(vp)
            candidates=np.unique(np.r_[0.,.5,1.,(score[:-1]+score[1:])/2])
            def criterion(t):
                pred=vp>t
                return .5*(pred[vy==1].mean()+(~pred[vy==0]).mean()),(pred==vy).mean(),-abs(t-.5),-t
            threshold=max(candidates,key=criterion)
            assert threshold==read(dest/'thresholds.json')['policies']['val_balanced']['threshold']
            verify_predictions(dest/'test_predictions.csv',r['val_threshold_test'],[229,69],threshold)
            expected_val=selected[v]['old_last_val' if name=='original' else 'last_val']
            assert abs(r['val']['auroc']-expected_val)<1e-12
            checked[name][v]=r
    combined=[]
    for v,old in checked['original'].items():
        new=checked['regularized'][v];s=selected[v]
        combined.append(dict(variant=v,old_selected_test=s['old_test'],old_epoch50_test=old['test']['auroc'],
                             new_selected_test=s['test'],new_epoch50_test=new['test']['auroc'],
                             new_selected_val=s['val'],new_epoch50_val=new['val']['auroc'],
                             selected_epoch=s['selected_epoch']))
    a.out.mkdir(parents=True,exist_ok=True)
    report=dict(passed=True,models_per_protocol=12,total_fixed_checkpoints=24,results=combined,
                checkpoint_identity='SHA256 verified against files on server; identities cross-checked locally against prior audits; last checkpoint binaries are retained on server',
                sample_metrics='AUROC and confusion matrices recomputed independently on CPU',
                verifier_sha256=sha(Path(__file__)))
    (a.out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (a.out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    styles=[('old_selected_test','Original / val-selected','#8c9199','--'),
            ('old_epoch50_test','Original / epoch 50','#8c9199','-'),
            ('new_selected_test','Regularized / val-selected','#2576a1','--'),
            ('new_epoch50_test','Regularized / epoch 50','#2576a1','-')]
    for ax,(arch,activation) in zip(axes.flat,[('moe','relu_softsign'),('d2nn','relu_softsign'),('moe','off'),('d2nn','off')]):
        group=sorted([r for r in combined if r['variant'].startswith(arch+'_') and r['variant'].endswith('_'+activation)],
                     key=lambda r:int(r['variant'].split('_')[1][1:]))
        for key,label,color,style in styles:
            ax.plot([2,4,6],[r[key] for r in group],marker='o',linestyle=style,color=color,label=label)
        ax.set_title(arch.upper()+' / '+('Softsign OEO' if activation!='off' else 'OEO off'))
        ax.set_xticks([2,4,6]);ax.set_ylim(.5,.8);ax.grid(alpha=.2);ax.set_xlabel('Depth');ax.set_ylabel('Test AUROC')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2)
    fig.suptitle('Checkpoint selection diagnostic — seed 17, retrospective')
    fig.tight_layout(rect=(0,.1,1,.96))
    fig.savefig(a.out/'selection_policy_comparison.png',dpi=170)
    fig.savefig(a.out/'selection_policy_comparison.pdf')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
