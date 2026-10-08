"""CPU-only, independent verification and paired reporting of the new protocol."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def auc(y, scores):
    a, b = scores[y == 1, None], scores[y == 0][None, :]
    return float(((a > b) + .5 * (a == b)).mean())


def verify_predictions(path, expected, support, threshold=.5):
    records = rows(path)
    assert len({r['sample_id'] for r in records}) == sum(support)
    y = np.array([int(r['label_true']) for r in records])
    p = np.array([[float(r['score0']), float(r['score1'])] for r in records])
    assert np.bincount(y).tolist() == support
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-6)
    prediction = p[:, 1] > threshold
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y, prediction.astype(int)), 1)
    assert cm.tolist() == expected['confusion_matrix']
    assert abs(auc(y, p[:, 1]) - expected['auroc']) < 1e-12
    assert abs((prediction == y).mean() - expected['accuracy']) < 1e-12
    return records, y, p[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--task', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root, task = args.run, args.task
    metadata, lock = read(root/'metadata.json'), read(root/'test_lock.json')
    assert metadata['seeds'] == [17], 'This verifier compares against the locally reproduced seed17 baseline only.'
    valresults = read(root/'validation_results.json')
    have_test = (root/'test_results.json').exists()
    testresults = {(r['variant'], r['seed']): r for r in read(root/'test_results.json')} if have_test else {}
    original = {r['variant']: r for r in read(task/'runs/simulation/adrenal_depth_audit_20260916/results.json')}
    for rel, digest in lock['files'].items():
        assert sha(root/rel) == digest, rel
    orders, transforms, report = {}, {}, []
    for item in valresults:
        variant, seed = item['variant'], item['seed']
        dest = root/'runs'/variant/('seed'+str(seed))
        assert item['checkpoint_sha256'] == sha(dest/'best_checkpoint.pt')
        assert item['threshold_sha256'] == sha(dest/'thresholds.json')
        assert item['updates'] == metadata['protocol']['epochs'] * 149
        assert all(item['changed_phase_planes'].values())
        orders.setdefault(seed, set()).add(tuple(item['order_sha256']))
        transforms.setdefault(seed, set()).add(tuple(item['transform_sha256']))
        history = rows(dest/'history.csv')
        assert [int(r['epoch']) for r in history] == list(range(1, metadata['protocol']['epochs']+1))
        best, best_auc, best_mse = None, -1., float('inf')
        for r in history:
            a, m = float(r['val_auroc']), float(r['val_detector_plane_mse'])
            if a > best_auc + 1e-6 or (abs(a-best_auc) <= 1e-6 and m < best_mse):
                best, best_auc, best_mse = int(r['epoch']), a, m
        assert best == item['selected_epoch'] and abs(best_auc-item['val']['auroc']) < 1e-12
        verify_predictions(dest/'selected_train_predictions.csv', item['train'], [929,259])
        records, yv, pv = verify_predictions(dest/'selected_val_predictions.csv', item['val'], [76,22])
        verify_predictions(dest/'last_train_predictions.csv', item['last_train'], [929,259])
        thresholds = read(dest/'thresholds.json')
        scores = np.unique(pv)
        candidates = np.unique(np.r_[0., .5, 1., (scores[:-1]+scores[1:])/2])
        def criterion(t):
            pred = pv > t
            balanced = .5 * (pred[yv==1].mean() + (~pred[yv==0]).mean())
            return balanced, (pred==yv).mean(), -abs(t-.5), -t
        t = max(candidates, key=criterion)
        assert t == thresholds['policies']['val_balanced']['threshold']
        old = original[variant]
        depth = int(variant.split('_')[1][1:])
        oldrun = task/'runs/simulation'/f'adrenal_L{depth}_s17_202609{15 if depth==2 else 16}'/'runs'/variant/'seed17'
        assert read(dest/'initialization.json') == read(oldrun/'initialization.json')
        old_completed = read(oldrun/'completed.json')
        assert item['order_sha256'] == old_completed['order_sha256']
        assert item['parameters'] == old_completed['parameters']
        result = dict(variant=variant, seed=seed, selected_epoch=best,
                      old_train=old['metrics']['best_train']['auroc'], train=item['train']['auroc'],
                      old_val=old['metrics']['best_validation']['auroc'], val=item['val']['auroc'],
                      old_last_train=old['metrics']['last_train']['auroc'], last_train=item['last_train']['auroc'],
                      old_last_val=old['metrics']['last_validation']['auroc'], last_val=float(history[-1]['val_auroc']),
                      threshold=float(t))
        if have_test:
            test = testresults[(variant,seed)]
            rec, yt, pt = verify_predictions(dest/'test_predictions.csv', test['fixed'], [229,69])
            verify_predictions(dest/'test_predictions.csv', test['val_threshold'], [229,69], t)
            oldmetrics = read(oldrun/'test_metrics.json')
            oldrecords, oy, op = verify_predictions(oldrun/'test_predictions.csv', oldmetrics, [229,69])
            assert [r['sample_id'] for r in rec] == [r['sample_id'] for r in oldrecords]
            assert np.array_equal(yt, oy)
            # Paired, class-stratified sample bootstrap. This quantifies this
            # test set's sampling uncertainty, NOT training-seed uncertainty.
            rng = np.random.default_rng(20260916)
            neg, pos = np.flatnonzero(yt==0), np.flatnonzero(yt==1)
            differences = []
            for _ in range(2000):
                idx = np.r_[rng.choice(neg,len(neg)), rng.choice(pos,len(pos))]
                differences.append(auc(yt[idx], pt[idx])-auc(yt[idx], op[idx]))
            lo, hi = np.quantile(differences,[.025,.975])
            result.update(old_test=oldmetrics['auroc'], test=test['fixed']['auroc'],
                          test_delta_ci_low=float(lo), test_delta_ci_high=float(hi),
                          fixed_accuracy=test['fixed']['accuracy'],
                          val_threshold_accuracy=test['val_threshold']['accuracy'],
                          val_threshold_balanced_accuracy=test['val_threshold']['balanced_accuracy'],
                          val_threshold_recall=test['val_threshold']['positive_recall'])
        report.append(result)
    assert all(len(v)==1 for v in orders.values()) and all(len(v)==1 for v in transforms.values())
    result = dict(passed=True, models=len(report), test_verified=have_test, identical_paired_orders=True,
                  identical_paired_augmentation=True, selection_verified=True, thresholds_verified=True,
                  verifier_sha256=sha(Path(__file__)), data_sha256=metadata['data_sha256'], results=report)
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (root/'paired_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
    print(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,(arch,activation) in zip(axes.flat,[('moe','relu_softsign'),('d2nn','relu_softsign'),('moe','off'),('d2nn','off')]):
        group=sorted([r for r in report if r['variant'].startswith(arch+'_') and r['variant'].endswith('_'+activation)],
                     key=lambda r:int(r['variant'].split('_')[1][1:]))
        # Plot each seed separately: never silently average incomplete runs.
        for seed in sorted({r['seed'] for r in group}):
            subset=[r for r in group if r['seed']==seed]
            d=[int(r['variant'].split('_')[1][1:]) for r in subset]
            for key,color in [('train','#b35c16'),('val','#28779d')]+([('test','#29854b')] if have_test else []):
                ax.plot(d,[r['old_'+key] for r in subset],'o--',color=color,alpha=.6,label='Original '+key+f' s17')
                ax.plot(d,[r[key] for r in subset],'o-',color=color,label='Regularized '+key+f' s{seed}')
        ax.set_title(arch.upper()+' / '+('Softsign OEO' if activation!='off' else 'OEO off'))
        ax.set_xticks([2,4,6]);ax.set_ylim(.5,1);ax.grid(alpha=.2);ax.set_xlabel('Depth');ax.set_ylabel('Selected-checkpoint AUROC')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Paired regularization experiment — official split, seed 17')
    fig.tight_layout(rect=(0,.09,1,.95))
    fig.savefig(root/'regularization_comparison.png',dpi=170)
    fig.savefig(root/'regularization_comparison.pdf')


if __name__=='__main__':
    main()
