"""Summarize saved results and diagnostics without rerunning any model."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-prefix',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    task=Path(__file__).resolve().parent
    args.out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results={};fig,axes=plt.subplots(1,3,figsize=(15,4))
    rows=['| Architecture | Accuracy | Macro recall | Selected epoch |',
          '|---|---:|---:|---:|']
    for ax,variant in zip(axes,('optical','electronic','d2nn')):
        run=task/'runs'/'simulation'/f'{args.run_prefix}_{variant}'
        result=json.loads((run/'result.json').read_text())
        config=json.loads((run/'config.json').read_text())
        history=json.loads((run/'metrics.json').read_text())
        results[variant]=dict(result=result,config=config,history=history)
        rows.append(f'| {variant} | {result["accuracy"]*100:.2f}% | '
                    f'{result["balanced_accuracy"]*100:.2f}% | {result["selected_epoch"]} |')
        import numpy as np
        cm=np.asarray(result['confusion_matrix'])
        recall=cm/cm.sum(1,keepdims=True)
        im=ax.imshow(recall,vmin=0,vmax=1,cmap='Blues')
        ax.set(title=variant,xlabel='Predicted class',ylabel='True class',
               xticks=range(9),yticks=range(9))
        for r in range(9):
            for c in range(9):
                ax.text(c,r,str(cm[r,c]),ha='center',va='center',fontsize=7,
                        color='white' if recall[r,c]>.6 else 'black')
    fig.tight_layout();fig.savefig(args.out/'confusion_matrices.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for variant,item in results.items():
        hist=item['history'];axes[0].plot([r['epoch'] for r in hist],
            [r['validation']['balanced_accuracy']*100 for r in hist],label=variant)
        route=item['result'].get('router')
        if route:axes[1].plot(range(1,len(route['mean_power'])+1),route['mean_power'],'o-',label=variant)
    axes[0].set(xlabel='Epoch',ylabel='Validation macro recall (%)')
    slots=max(len(item['result'].get('router',{}).get('mean_power',[])) for item in results.values())
    axes[1].set(xlabel='Expert slot (row-major for four slots)',ylabel='Test mean routed power',xticks=range(1,slots+1))
    for ax in axes: ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.out/'training_and_routing.png',dpi=180);plt.close(fig)
    (args.out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.out/'results.md').write_text('# CRC9 three-way classification\n\n'+ '\n'.join(rows)+
        '\n\nFull fixed image split; one validation-selected checkpoint evaluation per model.\n'+
        'Run prefix: '+args.run_prefix+'\n')
    print('\n'.join(rows))

if __name__=='__main__':main()
