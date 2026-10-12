"""Plot macro recall and test confusion matrices from the completed CPU audit."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    out = args.report
    audit = json.loads((out / 'final_audit.json').read_text())
    lock = json.loads((out / 'global_selection_lock.json').read_text())
    assert audit['cpu_checks_passed'] and audit['total_test_evaluations'] == 8
    assert not audit['optical_inference_performed']
    assert all(row['test_evaluations'] == 1 and not row['test_used_for_selection']
               for row in audit['summaries'])
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    classes = lock['classes']
    summaries = audit['summaries']

    def save(fig, name):
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(out / f'{name}.{ext}', dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    positions = np.arange(len(summaries))
    for index, (split, color) in enumerate([('train', '#2365a5'),
                                           ('val', '#ce6c23'), ('test', '#278764')]):
        ax.bar(positions + (index - 1) * .25,
               [100 * row[f'{split}_macro_recall'] for row in summaries],
               width=.24, color=color, label=split)
    ax.set_xticks(positions, [f"L{row['depth']}\nrho={row['rho']}" for row in summaries])
    ax.set(ylabel='Macro recall (%)', ylim=(0, 105))
    ax.axhline(70, color='grey', linestyle=':', linewidth=1)
    ax.legend(); ax.grid(axis='y', alpha=.15)
    fig.suptitle('Caltech101 count-conditioned20 subset | common100 | seed17\n'
                 'Test class supports unequal; macro recall weights each class equally')
    save(fig, 'macro_summary')

    for depth in (2, 4, 8, 16):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        for ax, rho in zip(axes, (0., .3)):
            key = f'L{depth}_' + ('rho0' if rho == 0 else 'rho03')
            metric = next(row for row in audit['arms'] if row['key'] == key)['metrics']['test']
            counts = np.array(metric['confusion_matrix'])
            support = np.array(metric['support'])
            assert counts.shape == (len(classes), len(classes))
            assert counts.sum(axis=1).tolist() == support.tolist() and (support > 0).all()
            im = ax.imshow(counts / support[:, None], vmin=0, vmax=1, cmap='Blues')
            ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
            ax.set_yticks(range(len(classes)), classes, fontsize=7)
            for y in range(len(classes)):
                for x in range(len(classes)):
                    if counts[y, x]:
                        ax.text(x, y, str(counts[y, x]), ha='center', va='center', fontsize=5.5,
                                color='white' if counts[y, x] / support[y] > .5 else 'black')
            ax.set(xlabel='Predicted class', ylabel='True class',
                   title=f"L{depth} rho={rho} | test macro recall {100 * metric['balanced_accuracy']:.2f}%")
        fig.colorbar(im, ax=axes, shrink=.8, label='Fraction within true class')
        fig.suptitle('Caltech101 count-conditioned20 subset | test once after all8 EMA locks\n'
                     'Cells: saved test counts; color: row-normalized recall | seed17, no error bars')
        save(fig, f'test_confusion_L{depth}')


if __name__ == '__main__':
    main()
