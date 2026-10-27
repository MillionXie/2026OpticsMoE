"""Audit downloaded seed17 Mixup artifacts without running model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictions(path, expected_samples):
    with np.load(path, allow_pickle=False) as z:
        ids, labels, scores = z['ids'].astype(str), z['labels'], z['scores']
        assert scores.shape == (expected_samples, 8) and np.isfinite(scores).all() and (scores >= 0).all()
        assert len(set(ids)) == expected_samples and np.allclose(scores.sum(1), 1, atol=1e-5)
        assert np.all((labels >= 0) & (labels < 8))
        pred = scores.argmax(1)
        if 'predictions' in z:
            assert np.array_equal(pred, z['predictions'])
        cm = np.zeros((8, 8), dtype=int)
        np.add.at(cm, (labels, pred), 1)
        extra = {k: float(z[k]) for k in ['round', 'power_fraction'] if k in z}
    return ids, labels, pred, cm, extra


def summarize(snapshot, report):
    manifest = read(snapshot / 'artifact_manifest.json')
    for entry in manifest['files']:
        path = (snapshot / entry['path']).resolve()
        assert path.is_relative_to(snapshot.resolve())
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']
    server = read(snapshot / 'final_independent_server_audit.json')
    assert server['all_own_pids_exited'] and server['release_verified']
    plan = read(snapshot / 'plan.json')
    locked = read(snapshot / 'all_candidates_lock.json')
    declared = read(snapshot / 'summary.json')
    reference = snapshot / 'reference'
    original_test = read(reference / 'test_result.json')
    assert digest(reference / 'test_predictions.npz') == original_test['predictions_sha256']
    old_ids, old_y, old_pred, old_cm, _ = predictions(reference / 'test_predictions.npz', 752)
    assert np.array_equal(old_cm, original_test['test']['confusion_matrix'])
    rows, histories = [], {}
    for name in ['mixup_s17', 'power_curriculum_mixup_s17']:
        directory = snapshot / name
        result, metadata = read(directory / 'result.json'), read(directory / 'metadata.json')
        history = read(directory / 'history.json')
        assert result['rounds'] == len(history) <= 32
        assert [r['round'] for r in history] == list(range(1,len(history)+1))
        if name == 'mixup_s17':
            assert all(r['full49'] and r['power_fraction'] == 1 for r in history)
        else:
            assert history[0]['power_fraction'] == 0
            assert all(r['full49'] == (r['round'] >= 16) for r in history)
            assert all(0 <= r['power_fraction'] <= 1 for r in history)
            assert all(a['power_fraction'] <= b['power_fraction'] for a,b in zip(history,history[1:]))
        assert result['optimized_images'] == len(history) * 1024
        assert result['selected_power_fraction'] == 1
        assert result['full49_rounds'] == sum(x['full49'] for x in history) >= 12
        assert len(history) == 32 or history[-1]['patience'] >= 8
        assert all(len(x['gradients']) == 5 and all(
            g['finite'] and np.isfinite(g['norm']) and g['norm'] > 0
            for g in x['gradients'].values()) for x in history)
        assert all(np.isfinite(x['mixup_loss']) and np.isfinite(x['gradient_norm']) for x in history)
        assert result['checkpoint_sha256'] == server['methods'][name]['best_sha256']
        assert metadata['source_checkpoint_sha256'] == server['methods'][name]['source_sha256']
        assert metadata['git_commit'] == plan['source_commit'] == result['source_commit']
        route, split_ids, split_labels = {}, {}, {}
        for split, count in [('train', 3496), ('val', 752)]:
            cm = np.asarray(result[split]['confusion_matrix'])
            assert cm.shape == (8, 8) and cm.sum() == count
            assert abs(cm.trace() / count - result[split]['accuracy']) < 1e-7
            with np.load(directory / f'selected_{split}_routes.npz', allow_pickle=False) as z:
                ids, labels, p, mask = z['ids'].astype(str), z['labels'], z['route_probabilities'].astype(float), z['route_mask']
                assert len(set(ids)) == count and p.shape == (count, 49)
                assert np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(1), 1, atol=1e-6)
                assert np.all(mask == 1)
                split_ids[split], split_labels[split] = ids, labels
                route[split] = dict(mean_sample_entropy_neff=float(np.exp(-(p * np.log(p)).sum(1)).mean()),
                    mean_sample_participation_ratio=float((1 / (p * p).sum(1)).mean()),
                    mean_top24_power=float(np.sort(p, axis=1)[:, -24:].sum(1).mean()),
                    hard_all_one_is_fullk_definition=True)
        assert set(split_ids['train']).isdisjoint(split_ids['val'])
        assert set(split_ids['train']).isdisjoint(old_ids) and set(split_ids['val']).isdisjoint(old_ids)
        sampling = [json.loads(line) for line in (directory / 'sampling_history.jsonl').read_text().splitlines()]
        assert len(sampling) == len(history)
        for step, sample in enumerate(sampling, 1):
            ix = np.asarray(sample['indices'])
            assert sample['round'] == step and len(ix) == len(set(ix)) == 1024
            assert np.bincount(split_labels['train'][ix], minlength=8).tolist() == [128] * 8
            assert split_ids['train'][ix].tolist() == sample['ids']
        assert len(set(i for s in sampling[:4] for i in s['indices'])) == 3496
        for stem, selected in [('best_val_predictions', True), ('latest_val_predictions', False)]:
            ids, y, pred, cm, extra = predictions(directory / f'{stem}.npz', 752)
            assert np.array_equal(ids, split_ids['val']) and np.array_equal(y, split_labels['val'])
            step = int(extra['round'])
            expected = result['val'] if selected else history[-1]['val']
            assert np.array_equal(cm, expected['confusion_matrix'])
            if selected:
                assert step == result['selected_round'] and extra['power_fraction'] == 1
            else:
                assert step == len(history)
        for path in directory.glob('train_scores_round*.npz'):
            step = int(path.stem.replace('train_scores_round', ''))
            ids, y, pred, cm, extra = predictions(path, 3496)
            assert np.array_equal(ids, split_ids['train']) and np.array_equal(y, split_labels['train'])
            assert np.array_equal(cm, history[step - 1]['train_full']['confusion_matrix'])
        test = read(directory / 'test_result.json')
        assert test['checkpoint_sha256'] == result['checkpoint_sha256']
        assert digest(directory / 'test_predictions.npz') == test['predictions_sha256']
        ids, y, pred, cm, _ = predictions(directory / 'test_predictions.npz', 752)
        assert np.array_equal(ids, old_ids) and np.array_equal(y, old_y)
        assert np.array_equal(cm, test['test']['confusion_matrix'])
        correct = int((pred == y).sum())
        assert abs(correct / 752 - test['test']['accuracy']) < 1e-7
        old_right, new_right = old_pred == y, pred == y
        if result['selected_round'] == 0:
            receipt = read(directory / 'test_reuse_receipt.json')
            assert receipt['identical_checkpoint'] and not receipt['gpu_forward_executed']
            assert result['checkpoint_sha256'] == original_test['checkpoint_sha256']
            assert np.array_equal(pred, old_pred)
        else:
            assert test['power_fraction'] == 1 and test['source_commit'] == plan['source_commit']
        times = np.diff([0] + [x['elapsed_seconds'] for x in history])
        rows.append(dict(recipe=result['recipe'],seed=17,rounds=len(history),full49_rounds=result['full49_rounds'],
            selected_round=result['selected_round'],selected_power_fraction=1,optimized_images=result['optimized_images'],
            full_train_epoch_equivalent=result['optimized_images']/3496,source_run=metadata['source_run'],
            source_selected_epoch=metadata['source_selected_epoch'],source_sha256=metadata['source_checkpoint_sha256'],
            checkpoint_sha256=result['checkpoint_sha256'],last_checkpoint_sha256=server['methods'][name]['last_sha256'],
            train_accuracy=result['train']['accuracy'],validation_accuracy=result['val']['accuracy'],
            validation_nll=result['val']['macro_nll'],test_accuracy=correct/752,test_correct=correct,
            test_confusion_matrix=cm.tolist(),test_predictions_sha256=test['predictions_sha256'],
            test_reused=result['selected_round']==0,round_wall_median_seconds=float(np.median(times)),
            history_elapsed_seconds=history[-1]['elapsed_seconds'],route=route,
            sampling_all_rounds_verified=True,gradients_all_finite_nonzero=True,
            paired_with_original120=dict(old_correct=int(old_right.sum()),new_correct=correct,
                corrected=int((~old_right & new_right).sum()),new_wrong=int((old_right & ~new_right).sum()),
                both_wrong=int((~old_right & ~new_right).sum()),class_correct_delta=(cm.diagonal()-old_cm.diagonal()).tolist())))
        histories[name] = history
    assert len(locked['candidates']) == 2
    selected = max(rows,key=lambda r:(r['test_accuracy'],r['validation_accuracy'],-r['validation_nll']))
    assert selected['recipe'] == declared['selected']['recipe'] and selected['test_correct'] == declared['test_correct']
    assert declared['pilot_target_met'] == (selected['test_correct'] >= 616)
    summary = dict(dataset='Kather2016',seed=17,samples=dict(train=3496,val=752,test=752),
        role='User-authorized TEST-feedback method development; no independent confirmatory claim.',
        checkpoint_selection='VAL accuracy; NLL tie-break; only original full49 inference eligible',
        recipe_selection='Authorized observed TEST development score, then VAL tie-break',
        source_commit=plan['source_commit'],host_head=plan['host_head'],rows=rows,selected_recipe=selected['recipe'],
        target=dict(correct=616,total=752,accuracy=616/752,comparison='Previous three-seed k24 mean80.90% plus1pp',
            same_seed_k24_correct=604,same_seed_plus1pp_min_correct=612),
        pilot_target_met=selected['test_correct']>=616,three_seed_mean_target_established=False,
        outcome='Single-seed development target reached; three-seed target unestablished.' if selected['test_correct']>=616 else 'Target not reached; this finite phase is complete, the authorized optimization objective remains unmet.',
        optimized_images_total=sum(r['optimized_images'] for r in rows),
        source_weights_preserved=True,all_downloaded_sha_verified=True,manifest_sha256=digest(snapshot/'artifact_manifest.json'),
        artifact_count=len(manifest['files']),server_weight_and_release_audit=server,
        limitations=['Only seed17; no three-seed mean or SD.',
            'Partial-power scores are transition diagnostics, never full49 target results.',
            'Each round has1024 training optimizer inputs; it is not a3496-image full epoch.',
            'Round wall time includes periodic fullTRAIN evaluation and checkpoint overhead; startup/final exports are separate.',
            'Mixup changes training targets; raw objective is not comparable to hard-label loss.',
            'All older trials and their additional development costs remain preserved.'])
    report.mkdir(parents=True,exist_ok=True)
    (report/'final_summary_20261027.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for name,h in histories.items():
        label='Mixup' if name=='mixup_s17' else 'k24 to full49 + Mixup'
        steps=[r['round'] for r in h]
        line=axes[0,0].plot(steps,[r['val']['accuracy']*100 for r in h],label=label)[0]
        color=line.get_color()
        full=[r for r in h if r['full49']]
        axes[0,0].scatter([r['round'] for r in full],[r['val']['accuracy']*100 for r in full],color=color,s=12)
        axes[0,1].plot(steps,[r['val']['macro_nll'] for r in h],color=color,label=label+' VAL')
        axes[0,1].plot(steps,[r['train_probe']['macro_nll'] for r in h],color=color,ls='--',alpha=.6,label=label+' TRAIN probe')
        axes[1,0].plot(steps,np.diff([0]+[r['elapsed_seconds'] for r in h]),color=color,label=label)
    axes[0,0].axvline(16,color='gray',ls=':',label='Curriculum full49 starts')
    axes[0,0].axhline(78.324467,color='black',ls=':',label='Original full49 VAL')
    axes[0,0].set(title='VAL: markers denote eligible full49 rounds',xlabel='Round (1024 optimizer inputs)',ylabel='Accuracy (%)')
    axes[0,1].set(title='Clean TRAIN probe / VAL NLL',xlabel='Round',ylabel='NLL')
    axes[1,0].set(title='Round walltime includes periodic fullTRAIN scoring',xlabel='Round',ylabel='Seconds')
    x=np.arange(8)
    axes[1,1].bar(x-.25,old_cm.diagonal()/94*100,.25,label='Original120')
    for i,row in enumerate(rows):
        label='Curriculum fallback to Original120' if row['test_reused'] else row['recipe']
        axes[1,1].bar(x+i*.25,np.asarray(row['test_confusion_matrix']).diagonal()/94*100,.25,label=label)
    axes[1,1].set_xticks(x,['TUM','STR','CMP','LYM','DEB','MUC','ADI','EMP'])
    axes[1,1].set(title='Observed development TEST, locked full49 models',ylabel='Recall (%)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Seed17 finite Mixup / power-continuation development')
    fig.tight_layout();fig.savefig(report/'final_convergence_20261027.png',dpi=160);plt.close(fig)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    s=summarize(args.snapshot,args.report)
    print(json.dumps(dict(rows=[{k:r[k] for k in ['recipe','rounds','selected_round','train_accuracy','validation_accuracy','test_correct','test_accuracy','paired_with_original120']} for r in s['rows']],pilot_target_met=s['pilot_target_met'])))
