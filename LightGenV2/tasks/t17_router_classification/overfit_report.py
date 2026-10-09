"""Compare fixed-checkpoint full training/validation diagnostics without inference."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-prefix',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    task=Path(__file__).resolve().parent
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    analysis={}
    lines=['# Sixty-epoch continuation: generalization diagnostics','',
        'All metrics below use fixed weights on full train/validation splits. '
        'Online batch training loss is not used to estimate the gap.','',
        '| Model | Selected epoch | Selected train / val macro recall | '
        'Epoch60 train / val macro recall | Epoch60 gap | Validation peak to epoch60 drop |',
        '|---|---:|---:|---:|---:|---:|']
    for row,variant in enumerate(('optical','electronic','d2nn')):
        run=task/'runs'/'simulation'/f'{args.run_prefix}_{variant}'
        hist=json.loads((run/'metrics.json').read_text())
        start=json.loads((run/'resume_diagnostics.json').read_text())
        selected=json.loads((run/'selected_diagnostics.json').read_text())
        config=json.loads((run/'config.json').read_text())
        if hist[-1]['epoch']!=60 or 'training' not in hist[-1]:
            raise ValueError('run not completed through epoch60 with training diagnostics')
        points={x['epoch']:x for x in start.values()}
        points.update({x['epoch']:x for x in hist if 'training' in x})
        points=sorted(points.values(),key=lambda x:x['epoch'])
        valpoints={x['epoch']:x['validation'] for x in hist}
        valpoints.update({x['epoch']:x['validation'] for x in start.values()})
        valpoints=sorted(valpoints.items())
        last=hist[-1]
        train=last['training'];val=last['validation']
        peak=max(x['validation']['balanced_accuracy'] for x in hist)
        analysis[variant]=dict(selected=selected,epoch60=dict(training=train,validation=val),
            original_best=start['best'],original_last=start['last'],
            epoch60_macro_gap_pp=100*(train['balanced_accuracy']-val['balanced_accuracy']),
            epoch60_loss_gap=val['cross_entropy']-train['cross_entropy'],
            validation_peak_to_epoch60_drop_pp=100*(peak-val['balanced_accuracy']),
            best_validation_gain_pp=100*(peak-start['best']['validation']['balanced_accuracy']),
            train_macro_change_from_original_last_pp=100*(train['balanced_accuracy']-
                start['last']['training']['balanced_accuracy']),
            val_macro_change_from_original_last_pp=100*(val['balanced_accuracy']-
                start['last']['validation']['balanced_accuracy']),
            resume_source=config['resume_source'])
        item=analysis[variant]
        lines.append(f'| {variant} | {selected["epoch"]} | '
            f'{selected["training"]["balanced_accuracy"]*100:.2f}% / '
            f'{selected["validation"]["balanced_accuracy"]*100:.2f}% | '
            f'{train["balanced_accuracy"]*100:.2f}% / {val["balanced_accuracy"]*100:.2f}% | '
            f'{item["epoch60_macro_gap_pp"]:.2f} pp | '
            f'{item["validation_peak_to_epoch60_drop_pp"]:.2f} pp |')
        for col,key,label in ((0,'balanced_accuracy','Macro recall (%)'),
                              (1,'cross_entropy','Cross entropy')):
            ax=axes[row,col];factor=100 if col==0 else 1
            ax.plot([x['epoch'] for x in points],
                    [x['training'][key]*factor for x in points],'o-',label='Full train (sampled epochs)')
            usable=[(e,v) for e,v in valpoints if key in v]
            ax.plot([e for e,v in usable],[v[key]*factor for e,v in usable],label='Full validation')
            ax.axvline(selected['epoch'],color='gray',linestyle=':',label='Selected checkpoint')
            ax.set(title=variant,xlabel='Cumulative epoch',ylabel=label)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/'generalization_curves.png',dpi=160);plt.close(fig)
    (args.out/'generalization.json').write_text(json.dumps(analysis,indent=2)+'\n')
    lines.extend(['','Training and validation gaps are descriptive evidence. '
        'A growing gap plus declining validation after the peak supports late overfitting; '
        'a gap alone does not distinguish overfitting from distribution differences.',
        'Validation oscillations also need to be considered; the current experiment keeps '
        'the original constant learning rates.','',f'Run prefix: {args.run_prefix}'])
    (args.out/'generalization.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:9]))

if __name__=='__main__':main()
