"""Measure missing epoch accuracies by instrumenting the archived T24 training loop."""
import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from . import train


SPLITS = ('train', 'val', 'test')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(row):
    n, correct = row['n'], row['correct']
    check(n > 0 and 0 <= correct <= n and row['accuracy'] == correct / n, 'Invalid split accuracy')
    return {key: row[key] for key in ('n', 'correct', 'accuracy', 'cross_entropy')}


def tensor_differences(a, b):
    check(a.keys() == b.keys(), 'Parameter keys differ')
    maxima = {}
    for key, left in a.items():
        right = b[key]
        check(left.shape == right.shape and left.dtype == right.dtype, f'{key} tensor shape/dtype differs')
        maxima[str(key)] = (left - right).abs().max().item()
        check(np.isfinite(maxima[str(key)]), f'{key} tensor is non-finite')
    return {'all_equal': all(torch.equal(a[k], b[k]) for k in a), 'max_abs_by_key': maxima}


def prediction_comparison(original, replay, original_correct, replay_correct):
    with np.load(original, allow_pickle=False) as a, np.load(replay, allow_pickle=False) as b:
        for key in ('ids', 'labels'):
            check(np.array_equal(a[key], b[key]), f'{key} order differs in {original.name}')
        labels = a['labels']
        x, y = a['scores'], b['scores']
        check(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid prediction scores')
        left, right = x.argmax(1), y.argmax(1)
        check(np.array_equal(left, a['predictions']) and np.array_equal(right, b['predictions']), 'argmax mismatch')
        check((left == labels).sum() == original_correct and (right == labels).sum() == replay_correct,
              'Prediction counts differ from evaluated correct counts')
        return {'n': len(left), 'argmax_disagreements': int((left != right).sum()),
                'max_abs_probability_delta': float(np.max(np.abs(x - y))),
                'probabilities_equal': np.array_equal(x, y)}


def original_files(run):
    manifest = read(run / 'manifest.json')
    required = {'best_checkpoint.pt', 'last_checkpoint.pt', 'history.json', 'result.json',
                'selection_lock.json', 'metadata.json', *(f'predictions_{s}.npz' for s in SPLITS)}
    check(required.issubset(manifest), 'Original artifacts missing from manifest')
    check(read(run / 'status.json')['state'] == 'complete', 'Original training incomplete')
    for name in sorted(required):
        check(train.sha256(run / name) == manifest[name], f'Original artifact SHA mismatch: {name}')
    meta = read(run / 'metadata.json')
    for rel, expected in meta['source_sha256'].items():
        source = (train.REPO / rel).resolve()
        check(source.is_relative_to(train.REPO / 'LightGenV2') and train.sha256(source) == expected,
              f'Archived training source SHA mismatch: {rel}')
    config_path = (train.REPO / meta['command'][meta['command'].index('--config') + 1]).resolve()
    check(config_path.is_relative_to(train.TASK / 'configs') and train.sha256(config_path) == meta['config_file_sha256'],
          'Archived config file SHA mismatch')
    expected_config = read(config_path)
    expected_config['seed'] = meta['config']['seed']
    check(expected_config == meta['config'], 'Effective config differs beyond seed override')
    lock = read(run / 'selection_lock.json')
    result = read(run / 'result.json')
    check(all(result[k] == v for k, v in lock.items()) and lock['TEST_selection_calls'] == 0
          and lock['best_checkpoint_sha256'] == manifest['best_checkpoint.pt'], 'Original checkpoint lock mismatch')
    return manifest, meta, config_path, result


def run_replay(original, out, device, scratch):
    manifest, meta, config_path, result = original_files(original)
    config = meta['config']
    check(device.type == meta['device'] and str(torch.__version__) == meta['torch']
          and torch.version.cuda == meta['cuda'] and platform.python_version() == meta['python'],
          'Original device type or Python/PyTorch/CUDA runtime differs')
    if device.type == 'cuda':
        check(train.require_authorized_gpu(config['authorized_gpu_uuids']) == meta['gpu_uuid']
              and torch.cuda.get_device_name(device) == meta['gpu_model'], 'Replay requires original physical GPU')
    original_hist = read(original / 'history.json')
    n_epochs, best_epoch = config['epochs'], result['selected_epoch']
    check([r['epoch'] for r in original_hist] == list(range(1, n_epochs + 1)), 'Original history incomplete')
    check(best_epoch == max(original_hist, key=lambda r: (r['validation']['correct'],
          -r['validation']['cross_entropy']))['epoch'], 'Original VAL selection mismatch')
    inputs = {}
    observations = []
    states = {}
    original_loader, original_evaluate = train.load_inputs, train.evaluate

    def load_and_check(*args):
        splits, identity = original_loader(*args)
        check(identity == meta['data'], 'Original input manifest or raw pixel metadata differs')
        for split in SPLITS:
            _, labels, ids = splits[split]
            with np.load(original / f'predictions_{split}.npz', allow_pickle=False) as saved:
                check(np.array_equal(ids, saved['ids']) and np.array_equal(labels.numpy(), saved['labels']),
                      f'Original {split} IDs/labels differ from loader')
        inputs.update(splits)
        return splits, identity

    def at_epoch(model, split_data, batch, active_device, prediction_path=None):
        if prediction_path is not None:
            return original_evaluate(model, split_data, batch, active_device, prediction_path)
        check(split_data is inputs['val'], 'Unexpected training-loop selection/evaluation call')
        epoch = len(observations) + 1
        save = epoch in (best_epoch, n_epochs)

        def pred_path(split):
            return scratch / f'e{epoch}_{split}.npz' if save else None

        val = original_evaluate(model, split_data, batch, active_device, pred_path('val'))
        row = {'val': metrics(val)}
        # The original loop receives its own unmodified VAL object. Extra passes are eval-only.
        gpus = [torch.cuda.current_device()] if active_device.type == 'cuda' else []
        with torch.random.fork_rng(devices=gpus):
            for split in ('train', 'test'):
                row[split] = metrics(original_evaluate(model, inputs[split], batch, active_device, pred_path(split)))
            if save:
                states[epoch] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        observations.append(row)
        train.write(out / 'status.json', {'state': 'replaying', 'completed_epochs': epoch, 'pid': os.getpid()})
        return val

    temp = scratch / 'training'
    temp.mkdir()
    args = argparse.Namespace(config=config_path, out=temp, mode=meta['mode'], device=str(device))
    with patch.dict(os.environ, {'T24_SOURCE_COMMIT': meta['source_commit']}), \
            patch.object(train, 'load_inputs', load_and_check), patch.object(train, 'evaluate', at_epoch):
        train.run(args, config)

    replay_meta = read(temp / 'metadata.json')
    for field in ('source_commit', 'config_file_sha256', 'config', 'mode', 'initial_state_sha256', 'optimizer', 'data',
                  'selection', 'torch', 'cuda', 'python', 'device', 'gpu_model', 'gpu_uuid', 'parameter_count'):
        check(replay_meta[field] == meta[field], f'Replay metadata mismatch: {field}')
    check(all(replay_meta['source_sha256'].get(k) == v for k, v in meta['source_sha256'].items()),
          'Actual replay imports do not match archived runtime sources')
    replay_hist, replay_lock = read(temp / 'history.json'), read(temp / 'selection_lock.json')
    check(len(replay_hist) == len(observations) == n_epochs and replay_lock['TEST_selection_calls'] == 0,
          'Replay budget or checkpoint-selection rule changed')
    rows = []
    for a, b, obs in zip(original_hist, replay_hist, observations):
        for field in ('epoch', 'train_steps', 'minibatch_order_sha256'):
            check(a[field] == b[field], f'Epoch {a["epoch"]} replay {field} differs')
        av, bv = metrics(a['validation']), obs['val']
        check(av['n'] == bv['n'] and bv == metrics(b['validation']), 'Replay validation count mismatch')
        rows.append({'epoch': a['epoch'], 'train': obs['train'], 'val': av, 'test': obs['test'], 'replay': obs,
                     'source': {'train': 'seeded_replay_diagnostic', 'val': 'original_logged_validation',
                                'test': 'seeded_replay_diagnostic'}, 'original_val': av,
                     'original_online_train_loss': a['train_loss'], 'replay_online_train_loss': b['train_loss'],
                     'val_correct_delta': bv['correct'] - av['correct'],
                     'val_ce_abs_delta': abs(bv['cross_entropy'] - av['cross_entropy']),
                     'online_train_loss_abs_delta': abs(b['train_loss'] - a['train_loss']),
                     'train_steps': a['train_steps'], 'minibatch_order_sha256': a['minibatch_order_sha256']})

    def checkpoint(path):
        return torch.load(path, map_location='cpu', weights_only=False)

    endpoint_checks = {}
    for name, epoch in (('best', best_epoch), ('last', n_epochs)):
        payload = checkpoint(original / f'{name}_checkpoint.pt')
        check(payload['epoch'] == epoch and payload['source_commit'] == meta['source_commit']
              and payload['mode'] == meta['mode'] and payload['config'] == config,
              f'Original {name} payload identity mismatch')
        state_diff = tensor_differences(payload['model'], states[epoch])
        check(len(payload['model']) == 4, 'Unexpected optical model state')
        observed = rows[epoch-1]
        # Original best predictions were archived for all 3 splits. Evaluate the actual
        # original last PT directly as well when that epoch has no archived predictions.
        if name == 'last' and epoch != best_epoch:
            model = train.build_model(meta['mode'], config['num_classes']).to(device)
            model.load_state_dict(payload['model'], strict=True)
            saved_metrics, saved_paths = {}, {}
            for split in SPLITS:
                saved_paths[split] = scratch / f'original_last_{split}.npz'
                saved_metrics[split] = metrics(original_evaluate(model, inputs[split], config['batch'], device, saved_paths[split]))
            del model
        else:
            if name == 'last':
                check(tensor_differences(payload['model'], checkpoint(original / 'best_checkpoint.pt')['model'])['all_equal'],
                      'Identical best and last epoch have different original tensors')
            saved_paths = {s: original / f'predictions_{s}.npz' for s in SPLITS}
            saved_metrics = {s: metrics(result[s]) for s in SPLITS}
        predictions = {}
        for split in SPLITS:
            predictions[split] = prediction_comparison(saved_paths[split], scratch / f'e{epoch}_{split}.npz',
                                                       saved_metrics[split]['correct'], observed['replay'][split]['correct'])
            if split != 'val':
                observed[split] = saved_metrics[split]
                observed['source'][split] = ('original_best_archived_predictions' if epoch == best_epoch
                                              else 'original_last_direct_evaluation')
        endpoint_checks[name] = {'epoch': epoch, 'model_tensors': state_diff, 'predictions': predictions,
                                 'original_metrics': saved_metrics}
    replay_last = checkpoint(temp / 'last_checkpoint.pt')
    total_steps = sum(r['train_steps'] for r in rows)
    opt_a, opt_b = payload['optimizer'], replay_last['optimizer']
    check(opt_a['param_groups'] == opt_b['param_groups'], 'Adam parameter groups differ')
    check(opt_a['state'].keys() == opt_b['state'].keys() and len(opt_a['state']) == 4, 'Adam parameter states differ')
    optimizer_comparison = {}
    for key, value in opt_a['state'].items():
        check(value['step'].item() == opt_b['state'][key]['step'].item() == total_steps, 'Adam update count differs')
        optimizer_comparison[str(key)] = tensor_differences(value, opt_b['state'][key])
    check(torch.equal(payload['torch_rng_state'], replay_last['torch_rng_state'])
          and payload['shuffle_rng_state'] == replay_last['shuffle_rng_state'], 'Final RNG state differs')
    tolerances = {'val_count_abs': 0, 'val_ce_abs': 1e-5, 'online_train_loss_abs': 1e-6}
    within_tol = all(r['val_correct_delta'] == 0 and r['val_ce_abs_delta'] <= tolerances['val_ce_abs']
                     and r['online_train_loss_abs_delta'] <= tolerances['online_train_loss_abs'] for r in rows)
    exact_history = all(r['val_correct_delta'] == r['val_ce_abs_delta'] == r['online_train_loss_abs_delta'] == 0 for r in rows)
    matched = (exact_history and replay_lock['selected_epoch'] == best_epoch
               and all(e['model_tensors']['all_equal'] and all(p['probabilities_equal'] for p in e['predictions'].values())
                       for e in endpoint_checks.values()) and all(d['all_equal'] for d in optimizer_comparison.values()))
    return {'dataset': config['dataset'], 'mode': meta['mode'], 'seed': config['seed'], 'epochs': rows,
            'original_run': original.relative_to(train.TASK).as_posix(), 'original_selected_epoch': best_epoch,
            'original_source_commit': meta['source_commit'], 'original_history_sha256': manifest['history.json'],
            'original_best_checkpoint_sha256': manifest['best_checkpoint.pt'],
            'original_last_checkpoint_sha256': manifest['last_checkpoint.pt'],
            'original_manifest_sha256': train.sha256(original / 'manifest.json'), 'original_artifact_sha256': manifest,
            'replay_script_sha256': train.sha256(Path(__file__)), 'original_source_sha256': meta['source_sha256'],
            'original_config_file_sha256': meta['config_file_sha256'], 'initial_state_sha256': meta['initial_state_sha256'],
            'device': str(device), 'gpu_uuid': meta['gpu_uuid'], 'gpu_model': meta['gpu_model'],
            'selection': meta['selection'], 'replay_TEST_selection_calls': replay_lock['TEST_selection_calls'],
            'verification': {'trajectory_status': 'matched_all_observable_checks' if matched else 'diverged_seeded_replay',
                             'per_epoch_history_numerically_exact': exact_history, 'per_epoch_within_tolerances': within_tol,
                             'history_tolerances': tolerances, 'replay_selected_epoch': replay_lock['selected_epoch'],
                             'all_split_ids_labels_and_order_verified': True, 'initial_state_verified': True,
                             'ordered_minibatch_hashes_verified': True, 'optimizer_steps_per_parameter': total_steps,
                             'final_torch_and_shuffle_rng_match': True, 'final_adam_state_comparison': optimizer_comparison,
                             'archived_checkpoint_comparisons': endpoint_checks},
            'interpretation': 'Original VAL is logged each epoch. Original TRAIN and TEST are available for best/last. '
                              'Other end-of-epoch TRAIN and TEST points are instrumented seeded replay measurements. '
                              'Original unsaved intermediate model tensors cannot be directly compared.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    original, out = args.run.resolve(), args.out.resolve()
    if not original.is_relative_to(train.TASK / 'runs' / 'simulation') or not out.is_relative_to(train.TASK / 'runs' / 'smoke'):
        parser.error('Original run must be under T24/runs/simulation and new output under T24/runs/smoke')
    out.mkdir(parents=True, exist_ok=False)
    train.write(out / 'status.json', {'state': 'verifying_original', 'pid': os.getpid()})
    try:
        with tempfile.TemporaryDirectory(prefix='isolated_replay_', dir=out) as temp:
            output = run_replay(original, out, torch.device(args.device), Path(temp))
        train.write(out / 'result.json', output)
        train.write(out / 'manifest.json', {'result.json': train.sha256(out / 'result.json')})
        train.write(out / 'status.json', {'state': 'complete', 'pid': os.getpid(),
                                         'trajectory_status': output['verification']['trajectory_status']})
    except BaseException as exc:
        train.write(out / 'status.json', {'state': 'failed', 'pid': os.getpid(), 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
