"""Plot measured panel-B histories without treating test results as a time series."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    task = Path(__file__).resolve().parent
    paths = [task/'runs/simulation'/name for name in
             ['panel_b_depth_s17_20261031/worker1/lg4',
              'panel_b_d2nn60_s17_20261031/d2nn3']]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, path, title in zip(axes, paths, ['LG: 4 main layers', 'D2NN + OEO: 3 layers']):
        history = json.loads((path/'history.json').read_text())
        result = json.loads((path/'result.json').read_text())
        for names, label, color in [(['original_TRAIN', 'training'], 'TRAIN', '#4477aa'),
                                     (['validation'], 'VAL', '#cc6677')]:
            pairs = [(r['epoch'], 100*r[k]['accuracy']) for r in history for k in names if k in r]
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
    axes[0].set_ylabel('Accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle('Seed 17; original TEST exposed during development', fontsize=11)
    fig.tight_layout(rect=(0, .18, 1, .94))
    out = task/'reports/panel_b_depth_s17_20261031'
    fig.savefig(out/'convergence.png', dpi=160)
    fig.savefig(out/'convergence.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
