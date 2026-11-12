"""Plot the published aggregate global-ablation table."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .plot_style import save_figure
else:
    from plot_style import save_figure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.summary.open(encoding='utf-8-sig') as stream:
        rows={(r['kind'],r['stage']):r for r in csv.DictReader(stream)}
    fig,ax=plt.subplots(figsize=(7.2,3.5),layout='constrained')
    x=np.arange(3)
    styles = [
        ('bypass', 'Block bypass', '#86A2E1'),
        ('random_phase', 'Uniform phase', '#F5BA92'),
        ('random_raw_phase', 'Default raw phase', '#E2E2E2'),
        ('random_global', 'Random block + uniform phase', '#FFEDB8'),
        ('random_global_default', 'Default random block', '#8CB9A8'),
    ]
    styles = [style for style in styles if all((style[0], stage) in rows for stage in ('vision','joint','both'))]
    offsets = (np.arange(len(styles)) - (len(styles)-1)/2)*.14
    for offset, (kind, label, color) in zip(offsets, styles):
        values=[float(rows[kind,stage]['mean_srcc']) for stage in ('vision','joint','both')]
        errors=[float(rows[kind,stage]['std_srcc']) for stage in ('vision','joint','both')]
        ax.bar(x+offset,values,width=.13,yerr=errors,capsize=2,label=label,color=color,edgecolor='#777777',linewidth=.4)
    reference=float(rows['learned','both']['mean_srcc'])
    ax.axhline(reference,color='#ad2727',linestyle='--',label='Learned model')
    ax.axhline(0,color='#777777',linewidth=.5)
    ax.set(xticks=x,xticklabels=['Vision global','Joint global','Both globals'],ylabel='Development SRCC',ylim=(-1,1))
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3)
    ax.set_title('Global-stage forward ablations')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_figure(fig,args.output)
    plt.close(fig)


if __name__=='__main__':
    main()
