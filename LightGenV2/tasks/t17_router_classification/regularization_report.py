"""Summarize validation-selected MoE candidates without additional inference."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-prefix',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--candidates',nargs='+');args=p.parse_args()
    task=Path(__file__).resolve().parent
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    summary={};lines=['# MoE regularization candidates','',
        'Selection uses validation only: mean of accuracy and macro recall. '
        'Only the winning candidate per architecture receives one exploratory test.','',
        '| Candidate | Weights | Fine-tune epoch | Val accuracy | Val macro | Train macro | Gap | Test accuracy | Test macro |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    names=('optical_early','optical_late','electronic')
    if (task/'runs'/'simulation'/f'{args.run_prefix}_optical').exists():names=('optical','electronic')
    if args.candidates:names=args.candidates
    for name in names:
        run=task/'runs'/'simulation'/f'{args.run_prefix}_{name}'
        d=json.loads((run/'selected_diagnostics.json').read_text())
        cfg=json.loads((run/'config.json').read_text());hist=json.loads((run/'metrics.json').read_text())
        initial=json.loads((run/'initial_diagnostics.json').read_text())
        result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else None
        v=d['validation'];t=d['training'];gap=100*(t['balanced_accuracy']-v['balanced_accuracy'])
        summary[name]=dict(selected=d,config=cfg,history=hist,initial=initial,result=result,
                          macro_gap_pp=gap)
        test_acc=f'{result["accuracy"]*100:.2f}%' if result else 'not tested'
        test_macro=f'{result["balanced_accuracy"]*100:.2f}%' if result else 'not tested'
        lines.append(f'| {name} | {d["weights_kind"]} | {d["fine_tune_epoch"]} | '
            f'{v["accuracy"]*100:.2f}% | {v["balanced_accuracy"]*100:.2f}% | '
            f'{t["balanced_accuracy"]*100:.2f}% | {gap:.2f} pp | {test_acc} | {test_macro} |')
        axes[0].plot([r['fine_tune_epoch'] for r in hist],
                     [r['selection_score']*100 for r in hist],label=name)
        axes[1].scatter(t['balanced_accuracy']*100,v['balanced_accuracy']*100,label=name,s=65)
        if result and result.get('router'):
            summary[name]['routing']=result['router']
    axes[0].set(xlabel='Fine-tune epoch',ylabel='Validation selection score (%)')
    axes[1].plot([65,100],[65,100],'--',color='gray')
    axes[1].set(xlabel='Selected clean train macro recall (%)',ylabel='Selected validation macro recall (%)')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/'generalization_curves.png',dpi=170);plt.close(fig)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines.extend(['',f'Run prefix: {args.run_prefix}',
        'Inference geometry, Top-2, two OEO stages and single Linear are unchanged. '
        'D4 augmentation and regularization are training-only. '
        'Baseline tests were previously seen; new tests are exploratory.'])
    (args.out/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
