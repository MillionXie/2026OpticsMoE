"""Export fixed-weight predictions for every layered TRAIN/TEST example.

Run as a file with --runtime-root pointing at the original audited source checkout.
No training, model selection, dataset mutation, or target-assisted repair occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


IDENTITIES = {
    'ours': ('layered_scene_exp05_dc30_ccdsmall',
             'layered_dc30_ccdsmall_selected_e45_s73_20260925',
             'selected_checkpoint.pt', 'normal_predictions.jsonl',
             '03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21', .8765),
    'qwen': ('layered_scene_qwen_shared',
             'layered_scene_qwen_shared_s73_e100_20260920',
             'best_checkpoint.pt', 'test_predictions.jsonl',
             '0be775796d5f64df3fab6359eb354ec6666dfc29329484a620b619a6f7ec7cf8', .8120),
}
MANIFESTS = {
    'train': '452c0d4cede80ee8f838a853566b2bc52204d1c3c588fd2c6b4f7c99fac31fff',
    'test': '93aa528c61fb7da116dc0bb7e749f7f3e525d1140050982e33604e3e72d22fca',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    runtime = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime))
    import torch
    from torch.utils.data import DataLoader
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    from LightGenV2.tasks.t04_semantic_interaction.training import _set_phase_dropout
    from LightGenV2.tasks.t04_semantic_interaction.qwen_shared import CachedDataset, cache_path
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
        OpenMojiEditingDataset, collate_samples, load_prompt_cache,
    )
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator

    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; never overwrite evidence')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(73)
    device = torch.device(args.device)
    task = runtime / 'LightGenV2/tasks/t04_semantic_interaction'
    provenance = {'runtime_root': str(runtime), 'runtime_commit': subprocess.check_output(
        ['git', '-C', str(runtime), 'rev-parse', 'HEAD'], text=True).strip(),
        'exporter_sha256': digest(__file__), 'device': str(device), 'batch_size': args.batch_size,
        'evaluation': 'fixed checkpoint, perturbations disabled; TEST-selected development metrics',
        'models': {}}
    image_entries = []
    original_root = None
    for name, (profile, runname, filename, metricsname, expected_sha, expected_cca) in IDENTITIES.items():
        run_dir = args.runs_root / runname
        config = json.loads((run_dir / 'resolved_config.json').read_text())
        settings = load_settings(task / 'configs' / (profile + '.yaml'))
        for key in ('data_dir', 'svg_asset_dir', 'asset_dir', 'qwen_checkpoint',
                    'prompt_cache_path', 'optical_base_config'):
            if config.get(key):
                setattr(settings, key, Path(config[key]))
        settings.num_workers = 0
        if original_root is None:
            original_root = settings.data_dir
        assert settings.data_dir == original_root
        checkpoint = run_dir / filename
        assert digest(checkpoint) == expected_sha, f'Wrong {name} checkpoint'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = build_model(settings, device)
        assert payload['architecture'] == model.checkpoint_architecture
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        _set_phase_dropout(model, False)
        info = {'checkpoint_sha256': expected_sha, 'checkpoint': str(checkpoint),
                'epoch': payload['epoch'], 'architecture': payload['architecture'], 'splits': {}}
        archived = {r['sample_id']: r for r in map(json.loads,
            (run_dir / metricsname).read_text().splitlines())}
        prompt_cache = None if name == 'qwen' else load_prompt_cache(settings.prompt_cache_path)
        for split in ('test', 'train'):
            manifest = settings.data_dir / (split + '.jsonl')
            assert digest(manifest) == MANIFESTS[split], f'Wrong {split} dataset'
            dataset = (CachedDataset(settings, split) if name == 'qwen' else
                       OpenMojiEditingDataset(manifest, settings, prompt_cache))
            assert len(dataset) == (1000 if split == 'test' else 5000)
            if name == 'ours' and args.reuse_ours_from:
                previous = args.reuse_ours_from / f'ours_{split}.jsonl'
                reused = [json.loads(line) for line in previous.read_text().splitlines()]
                assert [r['sample_id'] for r in reused] == [r['sample_id'] for r in dataset.records]
                summary_acc = MetricAccumulator()
                for record, original in zip(reused, dataset.records):
                    assert all(record[k] == original[k] for k in ('instruction', 'source_grid', 'target_grid'))
                    if split == 'test':
                        assert record['archived_metric_max_error'] < 1e-6
                        assert all(abs(v-archived[record['sample_id']][k]) < 1e-6
                                   for k, v in record['metrics'].items() if isinstance(v, (float, int)))
                    for group in ('overall', record['task']):
                        summary_acc.groups[group]['count'] += 1
                        for key, value in record['metrics'].items():
                            if isinstance(value, (float, int)):
                                summary_acc.groups[group][key] += value
                    image_entries.extend(str(Path(record['relative_dir']) / file)
                                         for file in ('source.png', 'target.png'))
                destination = args.output / f'ours_{split}.jsonl'
                destination.write_bytes(previous.read_bytes())
                metrics = summary_acc.compute()
                if split == 'test':
                    assert abs(metrics['overall']['changed_cell_accuracy'] - expected_cca) < 1e-8
                info['splits'][split] = {'manifest_sha256': MANIFESTS[split],
                    'records_sha256': digest(destination), 'metrics': metrics,
                    'archived_metric_max_error': 0., 'bf16_autocast': True,
                    'reused_verified_export': str(previous)}
                print(f'Reused all verified ours {split}: {len(reused)}', flush=True)
                del dataset
                continue
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate_samples)
            acc = MetricAccumulator()
            rows = {r['sample_id']: r for r in dataset.records}
            seen = []
            max_error = 0.
            archived_differences = []
            destination = args.output / f'{name}_{split}.jsonl'
            with destination.open('w', encoding='utf-8') as stream, torch.inference_mode():
                for batch in loader:
                    # Match the original evaluation's BF16 autocast and GPU
                    # sigmoid thresholding, rather than silently changing precision.
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=settings.amp_enabled and device.type == 'cuda'):
                        outputs = model(batch['source_image'].to(device),
                                        [p.to(device) for p in batch['prompt_hidden']])
                    metric_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in batch.items()}
                    values, predictions, edits = acc.update(outputs, metric_batch)
                    predictions, edits = predictions.cpu(), edits.cpu()
                    for i, values_row in enumerate(values):
                        sample_id = values_row['sample_id']
                        row = rows[sample_id]
                        error = None
                        if split == 'test':
                            differences = {k: {'archived': archived[sample_id][k], 'current': v}
                                           for k, v in values_row.items() if isinstance(v, (float, int))
                                           and abs(v-archived[sample_id][k]) >= 1e-6}
                            error = max(abs(v - archived[sample_id][k]) for k, v in values_row.items()
                                        if isinstance(v, (float, int)))
                            # A100 BF16 matches every CCA/IoU and all but one
                            # secondary F1 record. Keep the actual fresh grid/F1;
                            # never paste the archived F1 onto a different grid.
                            if differences:
                                assert (name, sample_id, set(differences)) == (
                                    'qwen', 'test_000486', {'object_f1'}), (name, sample_id, differences)
                                archived_differences.append({'sample_id': sample_id, 'differences': differences})
                            max_error = max(max_error, error)
                        image_hashes = {}
                        for file in ('source.png', 'target.png'):
                            rel = Path(row['relative_dir']) / file
                            image_hashes[file] = digest(settings.data_dir / rel)
                            if name == 'ours':
                                image_entries.append(str(rel))
                        result = {'sample_id': sample_id, 'split': split, 'task': row['task'],
                                  'instruction': row['instruction'], 'relative_dir': row['relative_dir'],
                                  'source_grid': batch['source_grid'][i].tolist(),
                                  'target_grid': batch['target_grid'][i].tolist(),
                                  'prediction_grid': predictions[i].tolist(),
                                  'predicted_edit_grid': edits[i].int().tolist(),
                                  'metrics': values_row, 'original_png_sha256': image_hashes,
                                  'archived_metric_max_error': error,
                                  'archived_secondary_differences': differences if split == 'test' else {}}
                        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                        seen.append(sample_id)
                    print(f'{name} {split}: {len(seen)}/{len(dataset)}', flush=True)
            assert seen == [r['sample_id'] for r in dataset.records]
            summary = {'manifest_sha256': MANIFESTS[split], 'records_sha256': digest(destination),
                       'metrics': acc.compute(), 'archived_metric_max_error': max_error,
                       'archived_secondary_differences': archived_differences,
                       'bf16_autocast': settings.amp_enabled and device.type == 'cuda'}
            if split == 'test':
                assert abs(summary['metrics']['overall']['changed_cell_accuracy'] - expected_cca) < 1e-8
            if name == 'qwen':
                summary['frozen_native_cache'] = json.loads(cache_path(settings, split).with_suffix('.json').read_text())
            info['splits'][split] = summary
            del dataset, loader
        provenance['models'][name] = info
        del model, payload, prompt_cache
        torch.cuda.empty_cache()
    assert len(set(image_entries)) == 12000
    archive = args.output / 'original_pngs.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for relative in image_entries:
            tar.add(original_root / relative, arcname=relative, recursive=False)
    provenance['original_images'] = {'count': 12000, 'archive_sha256': digest(archive)}
    provenance['source_file_sha256'] = {str(p.relative_to(runtime)): digest(p) for p in (
        task / 'modeling.py', task / 'embedding_model.py', task / 'shared_readout.py', task / 'qwen_shared.py')}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Completed: 6000 examples x 2 fixed models; archived TEST metrics reproduced.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:6')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--reuse-ours-from', type=Path,
                        help='Reuse complete optical split exports after strict manifest/metric checks')
    run(parser.parse_args())
