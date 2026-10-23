"""Fixed 0/1/5-epoch ABO diagnostics, on CPU, without TEST budget selection.

I2I has one zero-training condition: official frozen RN18 GAP512 + L2, with
no learned projection. I2T/T2I each have independent 0, 1 and 5 epoch runs of
the existing frozen-RN18 + random byte-GRU model. The zero-epoch cross-modal
condition is an unaligned random-head diagnostic, not a usable zero-shot model.
This entry point never initializes from an existing experiment checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .easy100_data import Easy100
from .frozen_resnet18 import (FrozenResNet18, OFFICIAL_CHECKPOINT_SHA256,
                             assert_encoder_unchanged, encoder_state_sha256,
                             file_sha256)
from .models import build_model, parameter_counts
from .other_data import abo_data
from .retrieval import cross_modal_loss, exact_sku_ranking
from .train_easy100 import git, require_published_source, save_checkpoint, write_json

TASK = Path(__file__).resolve().parent
TASKS = ('abo_image_to_image', 'abo_image_to_text', 'abo_text_to_image')
BUDGETS = (0, 1, 5)
SEED, BATCH, WORKERS = 42, 32, 2
FORMAT = 't19_official_rn18_gap512_cpu_budget_v1'
PROFILE_PATH = TASK / 'configs/retrieval_fixed_budget_cpu_20261024.json'
SOURCE_FILES = ('evaluate_retrieval_budget.py', 'easy100_data.py', 'other_data.py',
                'frozen_resnet18.py', 'frozen_visual_models.py',
                'frozen_multimodal_models.py', 'models.py', 'retrieval.py')


def structured_sha256(value):
    """Stable recursive identity for tensors plus primitive metadata."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            visit(['tensor', str(item.dtype), list(item.shape)])
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'{')
            for key in sorted(item):
                visit(key)
                visit(item[key])
            digest.update(b'}')
        elif isinstance(item, (list, tuple)):
            digest.update(b'[')
            for child in item:
                visit(child)
            digest.update(b']')
        else:
            digest.update(json.dumps(item, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8'))
        digest.update(b'\0')

    visit(value)
    return digest.hexdigest()


def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class BareImageRetrieval(nn.Module):
    """No fitted projection, classifier, text encoder or task head."""
    def __init__(self):
        super().__init__()
        self.encoder = FrozenResNet18()

    def forward(self, images):
        return F.normalize(self.encoder(images), dim=-1)


def build_budget_model(task):
    # Reconstruct the FULL model after resetting RNG, including torchvision's
    # original constructor consumption, just as the existing 30-epoch profile.
    reset_seed()
    if task == 'abo_image_to_image':
        return BareImageRetrieval()
    return build_model(task, backbone='resnet18_imagenet_frozen', text_encoder='byte_gru')


def source_identity():
    return {name: file_sha256(TASK / name) for name in SOURCE_FILES}


def profile_identity():
    profile = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
    return {'path': PROFILE_PATH.relative_to(TASK).as_posix(),
            'sha256': file_sha256(PROFILE_PATH), 'configuration': profile}


def validate_profile(tasks):
    identity = profile_identity()
    config = identity['configuration']
    fixed = {'device': 'cpu', 'seed': SEED, 'batch_size': BATCH,
        'sampler_workers': WORKERS, 'sampler_persistent_workers': True,
        'learning_rate': .001, 'optimizer': 'AdamW', 'weight_decay': .0001,
        'gradient_clip': 1., 'warmup_epochs': 0, 'schedule': 'constant',
        'precision': 'FP32', 'complete_model_parameter_limit_exclusive': 15_000_000,
        'visual_official_file_sha256': OFFICIAL_CHECKPOINT_SHA256}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError('Published diagnostic configuration differs from the fixed protocol')
    expected = [(task, epoch) for task in TASKS
                for epoch in ((0,) if task == TASKS[0] else BUDGETS)]
    if [(item['task'], item['epochs']) for item in config['jobs']] != expected:
        raise ValueError('Published diagnostic configuration must retain all seven fixed conditions')
    return [item for item in config['jobs'] if item['task'] in tasks]


def _check_cache(cache):
    if (set(cache) != {'features', 'sample_ids', 'skus'}
            or cache['features'].dtype != torch.float32
            or cache['features'].ndim != 2 or cache['features'].shape[1] != 512
            or not torch.isfinite(cache['features']).all()
            or len(cache['features']) != len(cache['sample_ids'])
            or len(cache['features']) != len(cache['skus'])
            or not cache['sample_ids']
            or len(set(cache['sample_ids'])) != len(cache['sample_ids'])
            or any(not isinstance(x, str) or not x for x in cache['sample_ids'] + cache['skus'])):
        raise RuntimeError('Frozen feature cache shape, dtype or sample identity differs')


@torch.no_grad()
def cache_features(encoder, dataset, expected_ids, *, split, data_identity, output):
    """Own deterministic CPU cache; no task-head or label metric evaluation.

    Every stored value is checked by exact disk round-trip and blob SHA. Only
    the first/last batches get a second encoder forward, with honest coverage
    recorded. No task head or TEST metric is involved in this cache check.
    """
    if output.exists():
        raise FileExistsError('Refusing to replace an existing diagnostic cache')
    encoder_hash = encoder_state_sha256(encoder)
    assert_encoder_unchanged(encoder, encoder_hash)
    if encoder.pretrained_provenance['checkpoint_sha256'] != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError('Cache requires the pinned official pretrained visual encoder')
    began = time.monotonic()
    features, ids, skus, input_bindings, verified_ids = [], [], [], [], []
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED))
    for batch_number, batch in enumerate(loader):
        pixels = batch['image']
        if pixels.device.type != 'cpu' or pixels.dtype != torch.float32:
            raise RuntimeError('This diagnostic accepts CPU FP32 inputs only')
        z = encoder(pixels)
        if batch_number in (0, len(loader) - 1):
            direct = encoder(pixels)
            if not torch.equal(z, direct):
                raise RuntimeError('Deterministic same-batch frozen feature cache failed exact verification')
            verified_ids.extend(batch['sample_id'])
        features.append(z.detach().cpu())
        ids.extend(batch['sample_id'])
        skus.extend(batch['sku'])
        input_bindings.append(structured_sha256({'sample_ids': list(batch['sample_id']), 'pixels': pixels}))
    cache = {'features': torch.cat(features), 'sample_ids': ids, 'skus': skus}
    _check_cache(cache)
    if ids != list(expected_ids) or len(ids) != len(dataset):
        raise RuntimeError('Cache does not contain every original sample in order')
    audit = assert_encoder_unchanged(encoder, encoder_hash)
    metadata = {'format': FORMAT, 'split': split, 'samples': len(ids),
        'data_identity': data_identity, 'source_sha256': source_identity(),
        'encoder_state_sha256': encoder_hash,
        'pretrained_checkpoint_sha256': OFFICIAL_CHECKPOINT_SHA256,
        'cache_binding_sha256': structured_sha256(cache),
        'input_binding_sha256': structured_sha256(input_bindings),
        'all_stored_tensor_values_roundtrip_exactly_verified': True,
        'encoder_recomputed_samples': len(verified_ids), 'encoder_recomputed_sample_ids': verified_ids,
        'encoder_recomputation_coverage': 'first_and_last_batch_only',
        'frozen_encoder_audit': audit,
        'batch_size': BATCH, 'dtype': 'torch.float32', 'device': 'cpu',
        'numerical_scope': 'exact repeated CPU same-batch encoder forwards; '
            'no claim of bitwise identity to historical GPU or different batch layouts',
        'task_head_forward_count': 0, 'test_metrics_observed': False}
    output.mkdir(parents=True, exist_ok=False)
    save_checkpoint(output / 'features.pt', cache)
    metadata['cache_file_sha256'] = file_sha256(output / 'features.pt')
    restored = torch.load(output / 'features.pt', map_location='cpu', weights_only=True)
    if (not torch.equal(restored['features'], cache['features'])
            or restored['sample_ids'] != ids or restored['skus'] != skus
            or structured_sha256(restored) != metadata['cache_binding_sha256']):
        raise RuntimeError('Full feature cache disk round-trip is not exact')
    metadata['elapsed_seconds'] = time.monotonic() - began
    write_json(output / 'manifest.json', metadata)
    return cache, metadata


def load_feature_cache(output, *, data_identity, encoder_hash):
    """Fail closed on stale sources, metadata, IDs, tensors or disk bytes."""
    metadata = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if (metadata['format'] != FORMAT or metadata['data_identity'] != data_identity
            or metadata['encoder_state_sha256'] != encoder_hash
            or metadata['pretrained_checkpoint_sha256'] != OFFICIAL_CHECKPOINT_SHA256
            or metadata['source_sha256'] != source_identity()
            or not metadata['all_stored_tensor_values_roundtrip_exactly_verified']
            or file_sha256(output / 'features.pt') != metadata['cache_file_sha256']):
        raise RuntimeError('Diagnostic cache provenance or file SHA differs')
    cache = torch.load(output / 'features.pt', map_location='cpu', weights_only=True)
    _check_cache(cache)
    if (structured_sha256(cache) != metadata['cache_binding_sha256']
            or len(cache['sample_ids']) != metadata['samples']):
        raise RuntimeError('Diagnostic cache contents differ')
    return cache, metadata


def training_index_loader(length):
    """Same real RandomSampler/base-seed draws as the existing training loader."""
    return DataLoader(range(length), batch_size=BATCH, shuffle=True, num_workers=WORKERS,
        multiprocessing_context='spawn', persistent_workers=True,
        generator=torch.Generator().manual_seed(SEED), pin_memory=False)


def shutdown_loader(loader):
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()


def cached_images(model, features, *, bare=False):
    return F.normalize(features if bare else model.projection(features), dim=-1)


@torch.inference_mode()
def evaluate_cached(model, query_cache, candidate_cache=None, *, tokens=None,
                    title_skus=None, direction=None):
    model.eval()
    images = cached_images(model, query_cache['features'], bare=candidate_cache is not None)
    if candidate_cache is not None:
        candidates = cached_images(model, candidate_cache['features'], bare=True)
        qids, cids = query_cache['sample_ids'], candidate_cache['sample_ids']
        qskus, cskus = query_cache['skus'], candidate_cache['skus']
        scores = images @ candidates.T
    else:
        titles = model.encode_text(tokens)
        scores = images @ titles.T
        if direction == 'image_to_text':
            qids, cids = query_cache['sample_ids'], list(title_skus)
            qskus, cskus = query_cache['skus'], list(title_skus)
        elif direction == 'text_to_image':
            scores = scores.T
            qids, cids = list(title_skus), query_cache['sample_ids']
            qskus, cskus = list(title_skus), query_cache['skus']
        else:
            raise ValueError('Cross-modal evaluation needs a fixed direction')
    return {**exact_sku_ranking(scores, qskus, cskus), 'query_ids': qids,
            'candidate_ids': cids, 'query_skus': qskus, 'candidate_skus': cskus}


def run_budget(task, epochs, run_dir, train_cache, test_cache, cache_receipts,
               data_identity, *, tokens=None, title_skus=None):
    """Run one predeclared condition; never inspect TEST before its final state."""
    if task not in TASKS or epochs not in ((0,) if task == TASKS[0] else BUDGETS):
        raise ValueError('Only predeclared I2I0 and I2T/T2I0/1/5 conditions are allowed')
    run_dir.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    model = build_budget_model(task)
    counts = parameter_counts(model)
    if counts['total'] >= 15_000_000:
        raise RuntimeError('Complete diagnostic model must remain below 15M parameters')
    encoder_hash = encoder_state_sha256(model.encoder)
    if any(item['encoder_state_sha256'] != encoder_hash for item in cache_receipts.values()):
        raise RuntimeError('Fresh official encoder differs from diagnostic cache encoder')
    bare = task == TASKS[0]
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    allowed = ('projection.', 'text_projection.', 'text_encoder.')
    if any(not name.startswith(allowed) for name, _ in trainable):
        raise RuntimeError('Optimizer scope includes something outside GRU and task projections')
    if (bare and trainable) or (not bare and (counts['text_encoder'] != 27_104 or counts['trainable'] != 64_096)):
        raise RuntimeError('Unexpected task-head/GRU optimizer scope')
    optimizer = None if bare else torch.optim.AdamW([value for _, value in trainable], lr=.001, weight_decay=.0001)
    manifest = {'status': 'running', 'profile': FORMAT, 'task': task, 'epochs': epochs,
        'seed': SEED, 'device': 'cpu', 'precision': 'FP32_TF32_disabled', 'pid': os.getpid(),
        'git_commit': git('rev-parse', 'HEAD'), 'source_sha256': source_identity(),
        'execution_profile': profile_identity(),
        'command': [sys.executable, '-m', __spec__.name, *sys.argv[1:]],
        'torch': torch.__version__, 'python': platform.python_version(),
        'parameters': counts, 'pretrained_provenance': model.encoder.pretrained_provenance,
        'initialization': 'official_visual_only_no_trained_experiment_initialization',
        'architecture': 'official_RN18_GAP512_L2_no_projection' if bare else
            'unchanged_frozen_RN18_byte_GRU_with_64dim_projections',
        'zero_epoch_interpretation': 'direct_pretrained_visual_descriptor' if bare else
            'random_unaligned_crossmodal_diagnostic_not_usable_zero_shot_baseline',
        'data_identity': data_identity, 'cache_receipts': cache_receipts,
        'encoder_state_sha256_before': encoder_hash,
        'model_state_sha256_before': encoder_state_sha256(model),
        'text_state_sha256_before': encoder_state_sha256(model.text_encoder) if not bare else None,
        'optimizer_parameter_names': [name for name, _ in trainable],
        'optimizer_parameters': sum(value.numel() for _, value in trainable),
        'optimizer': None if bare else 'AdamW', 'learning_rate': .001,
        'weight_decay': .0001, 'gradient_clip': 1., 'schedule': 'constant', 'warmup_epochs': 0,
        'batch_size': BATCH, 'sampler_workers': WORKERS, 'sampler_persistent_workers': True,
        'sampler': 'actual_DataLoader_range_shuffle_generator42_spawn2_persistent',
        'selection': 'predeclared_fixed_budget_last_state', 'selection_uses_test': False,
        'previous_test_observations': 'historical tests observed; no claim of new independent blind test',
        'frozen_encoder_audits': [], 'task_test_evaluations': 0, 'started_unix': time.time()}
    write_json(run_dir / 'run_manifest.json', manifest)
    history, orders = [], []
    loader = training_index_loader(len(train_cache['features'])) if epochs else None
    direction = task.removeprefix('abo_')
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total, seen, order = 0., 0, []
            for indices in loader:
                indexes = indices.tolist()
                order.extend(indexes)
                optimizer.zero_grad(set_to_none=True)
                loss = cross_modal_loss(cached_images(model, train_cache['features'][indices]),
                    model.encode_text(tokens), [train_cache['skus'][i] for i in indexes],
                    title_skus, direction=direction)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite TRAIN loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_([value for _, value in trainable], 1.)
                optimizer.step()
                total += float(loss.detach()) * len(indexes)
                seen += len(indexes)
            if sorted(order) != list(range(len(train_cache['features']))):
                raise RuntimeError('TRAIN sampler did not visit every original sample once')
            entry = {'epoch': epoch, 'train_loss': total / seen, 'learning_rate': .001,
                     'sampler_order_sha256': structured_sha256(order)}
            history.append(entry)
            orders.append({'epoch': epoch, 'indices': order, 'sha256': entry['sampler_order_sha256']})
            manifest['frozen_encoder_audits'].append({'epoch': epoch,
                **assert_encoder_unchanged(model.encoder, encoder_hash)})
            write_json(run_dir / 'history.json', history)
            write_json(run_dir / 'sampler_orders.json', orders)
            write_json(run_dir / 'run_manifest.json', manifest)
            save_checkpoint(run_dir / 'last_checkpoint.pt', {'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'manifest': manifest, 'metrics': entry,
                'sampler_generator_state': loader.generator.get_state()})
        if not epochs:
            write_json(run_dir / 'history.json', [])
            write_json(run_dir / 'sampler_orders.json', [])
            save_checkpoint(run_dir / 'last_checkpoint.pt', {'epoch': 0, 'model': model.state_dict(),
                'optimizer': None if optimizer is None else optimizer.state_dict(), 'manifest': manifest})
        # Bind the one final evaluation to the actual newly saved disk last,
        # including epoch and AdamW state, before any task TEST observation.
        last_path = run_dir / 'last_checkpoint.pt'
        last_sha = file_sha256(last_path)
        selected = torch.load(last_path, map_location='cpu', weights_only=False)
        memory_hash = encoder_state_sha256(model)
        optimizer_hash = structured_sha256(None if optimizer is None else optimizer.state_dict())
        if (selected['epoch'] != epochs
                or selected['manifest']['model_state_sha256_before'] != manifest['model_state_sha256_before']
                or structured_sha256(selected['optimizer']) != optimizer_hash):
            raise RuntimeError('Saved last checkpoint epoch, initialization or optimizer differs')
        model.load_state_dict(selected['model'], strict=True)
        if encoder_state_sha256(model) != memory_hash:
            raise RuntimeError('Saved last checkpoint differs from the fixed terminal model')
        manifest['fixed_last_reload'] = {'epoch': epochs, 'checkpoint_sha256': last_sha,
            'terminal_model_state_sha256': memory_hash, 'optimizer_state_sha256': optimizer_hash,
            'strict_model_reload_verified': True}
        # Exactly one full task TEST observation after the fixed budget. No
        # periodic TEST calls, early stopping or desired-score budget selection.
        records = evaluate_cached(model, test_cache, train_cache if bare else None,
                                  tokens=tokens, title_skus=title_skus, direction=direction)
        manifest['task_test_evaluations'] = 1
        manifest['frozen_encoder_final_audit'] = assert_encoder_unchanged(model.encoder, encoder_hash)
        manifest['model_state_sha256_after'] = encoder_state_sha256(model)
        manifest['text_state_sha256_after'] = encoder_state_sha256(model.text_encoder) if not bare else None
        manifest['optimizer_steps'] = sorted({int(state['step']) for state in optimizer.state.values()}) if optimizer else []
        expected_steps = epochs * ((len(train_cache['features']) + BATCH - 1) // BATCH)
        if manifest['optimizer_steps'] != ([expected_steps] if expected_steps else []):
            raise RuntimeError('Optimizer steps differ from the fixed TRAIN budget')
        if not epochs and manifest['model_state_sha256_before'] != manifest['model_state_sha256_after']:
            raise RuntimeError('Zero-epoch diagnostic changed model state')
        write_json(run_dir / 'fixed_weight_predictions.json', records)
        result = {'status': 'complete', 'selected_epoch': epochs, 'selection': manifest['selection'],
            'selection_uses_test': False, 'checkpoint': 'last_checkpoint.pt',
            'checkpoint_sha256': last_sha,
            'metrics': {k: v for k, v in records.items() if isinstance(v, (int, float))},
            'task_test_evaluations': 1, 'diagnostic_only': True,
            'zero_epoch_interpretation': manifest['zero_epoch_interpretation'],
            'elapsed_seconds': time.monotonic() - began}
        if file_sha256(last_path) != last_sha:
            raise RuntimeError('Fixed last checkpoint changed during its final evaluation')
        if loader is not None:
            shutdown_loader(loader)
            loader = None
        write_json(run_dir / 'result.json', result)
        manifest['status'] = 'complete'
        return result
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        if loader is not None:
            try:
                shutdown_loader(loader)
            except BaseException as cleanup:
                manifest['loader_cleanup_error'] = repr(cleanup)
                # Preserve the original failure; a healthy run cleans before
                # marking complete, so normal cleanup failure also fails run.
        manifest['finished_unix'] = time.time()
        write_json(run_dir / 'run_manifest.json', manifest)


def run(args):
    require_published_source()
    if (args.device != 'cpu' or not 1 <= args.threads <= 8 or not args.tasks
            or len(set(args.tasks)) != len(args.tasks) or any(task not in TASKS for task in args.tasks)):
        raise ValueError('CPU-only fixed diagnostics require distinct supported tasks and 1..8 CPU threads')
    suite = args.suite_dir.resolve()
    if not any(suite.is_relative_to((TASK / 'runs' / category).resolve())
               for category in ('smoke', 'simulation')):
        raise ValueError('Diagnostic suite must be inside this task runs/smoke or runs/simulation')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    conditions = [{'task': item['task'], 'epochs': item['epochs']}
                  for item in validate_profile(args.tasks)]
    suite.mkdir(parents=True, exist_ok=False)
    write_json(suite / 'protocol.json', {'profile': FORMAT, 'conditions': conditions,
        'device': 'cpu', 'seed': SEED, 'sampling_workers': WORKERS,
        'all_conditions_retained': True, 'metric_thresholds_or_budget_selection': None,
        'test_policy': 'one complete task evaluation per predeclared condition, after its fixed budget',
        'source_sha256': source_identity(), 'git_commit': git('rev-parse', 'HEAD'),
        'execution_profile': profile_identity()})
    # These adapters verify the entire original image/CSV or ABO200 protocol
    # before a frozen feature cache is generated. No trained features are read.
    easy = Easy100(args.data_root) if any(task != TASKS[0] for task in args.tasks) else None
    bundles = {}
    if TASKS[0] in args.tasks:
        train, test, identity = abo_data()
        bundles['abo200'] = (train, test, identity)
    if easy is not None:
        bundles['easy100'] = (easy.images('train'), easy.images('test'), easy.identity)
    encoder = build_budget_model(TASKS[0]).encoder
    cached = {}
    for name, (train, test, identity) in bundles.items():
        cached[name] = {}
        for split, dataset in (('train', train), ('test', test)):
            ids = [row['sample_id'] for row in dataset.rows]
            _, built = cache_features(encoder, dataset, ids, split=split,
                data_identity=identity, output=suite / 'cache' / name / split)
            cache, receipt = load_feature_cache(suite / 'cache' / name / split,
                data_identity=identity, encoder_hash=built['encoder_state_sha256'])
            cached[name][split] = (cache, receipt)
    del encoder
    results = []
    for item in conditions:
        task, epochs = item['task'], item['epochs']
        name = 'abo200' if task == TASKS[0] else 'easy100'
        caches = cached[name]
        result = run_budget(task, epochs, suite / f'{task}_seed42_epoch{epochs}',
            caches['train'][0], caches['test'][0],
            {split: caches[split][1] for split in ('train', 'test')}, bundles[name][2],
            tokens=None if task == TASKS[0] else easy.title_tokens,
            title_skus=None if task == TASKS[0] else easy.title_skus)
        results.append({**item, 'run_id': f'{task}_seed42_epoch{epochs}', **result})
        write_json(suite / 'results.json', {'profile': FORMAT, 'results': results,
            'all_conditions_complete': len(results) == len(conditions),
            'selection_performed': False})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tasks', choices=TASKS, nargs='+', default=list(TASKS))
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
