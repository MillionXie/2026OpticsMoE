"""Plot measured panel-B histories without treating test results as a time series."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    task = Path(__file__).resolve().parent
    names = [('panel_b_depth_s17_20261031/worker1/lg4','LG: 4 main layers'),
             ('panel_b_depth_s17_20261031/worker1/lg6','LG: 6 main layers'),
             ('panel_b_d2nn60_s17_20261031/d2nn3','D2NN + OEO: 3 layers'),
             ('panel_b_d2nn60_s17_20261031/d2nn5','D2NN + OEO: 5 layers'),
             ('panel_b_d2nn60_s17_20261031/d2nn7','D2NN + OEO: 7 layers')]
    names = [(task/'runs/simulation'/name,title) for name,title in names
             if (task/'runs/simulation'/name/'result.json').exists()]
    fig, grid = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    axes = grid.flatten()
    for ax, (path, title) in zip(axes, names):
        history = json.loads((path/'history.json').read_text())
        result = json.loads((path/'result.json').read_text())
        for metric_names, label, color in [(['original_TRAIN', 'training'], 'TRAIN', '#4477aa'),
                                     (['validation'], 'VAL', '#cc6677')]:
            pairs = [(r['epoch'], 100*r[k]['accuracy']) for r in history for k in metric_names if k in r]
            ax.plot(*zip(*pairs), label=label, color=color, lw=1.5)
        selected = result['selected_epoch']
        ax.scatter([selected], [100*result['train']['accuracy']], marker='D', color='#4477aa', s=35,
                   label='Selected checkpoint TRAIN', zorder=4)
        dev = result.get('original_test_dev', result.get('test'))['accuracy']*100
        ax.scatter([selected], [dev], marker='*', color='#228833', s=100,
                   label='Locked checkpoint TEST-dev', zorder=4)
        ax.axvline(selected, color='#888888', lw=.8, ls=':')
        ax.set(title=title, xlabel='Executed epoch', ylim=(0, 100))
        ax.grid(alpha=.15)
    for ax in axes[len(names):]:ax.set_visible(False)
    axes[0].set_ylabel('Accuracy (%)');axes[3].set_ylabel('Accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle('Seed 17; original TEST exposed during development', fontsize=11)
    fig.tight_layout(rect=(0, .12, 1, .94))
    out = task/'reports/panel_b_depth_s17_20261031'
    fig.savefig(out/'convergence.png', dpi=160)
    fig.savefig(out/'convergence.svg')
    svg=out/'convergence.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    main()
