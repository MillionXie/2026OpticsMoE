"""Copied to reproduce.py by build_lab_package; paths are package-relative."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        p = ROOT / row['path']
        if not p.is_file() or digest(p) != row['sha256']:
            raise RuntimeError(f"Missing/modified package file: {row['path']}")
    print(f"Verified {len(manifest['files'])} files; source {manifest['package_git_commit']}", flush=True)
    return manifest


def settings(output):
    from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((ROOT / 'settings.json').read_text(encoding='utf-8')))
    cfg.config_path = ROOT / 'settings.json'
    cfg.data_dir = ROOT / 'data'
    cfg.asset_dir = ROOT / 'assets'
    cfg.output_dir = output
    cfg.qwen_checkpoint = ROOT / 'frontend'
    cfg.prompt_cache_path = ROOT / 'data/token_embeddings_v1.pt'
    cfg.optical_base_config = ROOT / 'experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint.yaml'
    cfg.legacy_warmstart_checkpoint = ROOT / 'weights/best_checkpoint.pt'
    cfg.shared_readout_variant = 'standard'
    cfg.num_workers = 0  # Portable Windows spawn / Linux file-handle behavior.
    return cfg


def output_directory(value):
    from datetime import datetime, timezone
    p = Path(value).resolve() if value else ROOT / 'outputs' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {p}')
    p.mkdir(parents=True, exist_ok=True)
    return p


def assert_local_sources():
    for name, mod in list(sys.modules.items()):
        if name.startswith(('experiments.', 'LightGenV2.')) and getattr(mod, '__file__', None):
            if not Path(mod.__file__).resolve().is_relative_to(ROOT):
                raise RuntimeError(f'External repository dependency: {name} -> {mod.__file__}')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def demo(cfg, device, sample_id):
    import torch
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import OpenMojiEditingDataset, collate_samples, load_prompt_cache
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.assets import load_icons, render_grid
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import compose_prediction
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    dataset = OpenMojiEditingDataset(cfg.test_manifest, cfg, load_prompt_cache(cfg.prompt_cache_path))
    indices = [i for i, r in enumerate(dataset.records) if r['sample_id'] == sample_id]
    if len(indices) != 1:
        raise ValueError(f'Unknown test sample: {sample_id}')
    sample = dataset[indices[0]]
    batch = collate_samples([sample])
    model = build_model(cfg, device)
    ckpt = torch.load(ROOT / 'weights/best_checkpoint.pt', map_location='cpu', weights_only=False)
    if ckpt['architecture'] != model.checkpoint_architecture:
        raise RuntimeError('Wrong architecture/checkpoint')
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    with torch.inference_mode():
        result = model(batch['source_image'].to(device), [g.to(device) for g in batch['prompt_hidden']])
        prediction, categories, edits = compose_prediction(result['category_logits'], result['edit_logits'], batch['source_grid'].to(device))
    report = {'sample_id': sample_id, 'instruction': sample['instruction'],
              'source_grid': sample['source_grid'].tolist(), 'target_grid': sample['target_grid'].tolist(),
              'predicted_grid': prediction[0].cpu().tolist(), 'edit_probability': result['edit_logits'][0].sigmoid().cpu().tolist(),
              'scene_exact': bool(torch.equal(prediction[0].cpu(), sample['target_grid'])),
              'source_grid_used_for_preservation': True,
              'routing': {label: {'selected_indices': core.optical_branch.core.last_routing['selected_indices'].tolist(),
                                  'energy_fraction': core.optical_branch.core.last_routing['detector_energy_fraction'].tolist()}
                          for label, core in [('language', model.language_core), ('vision', model.vision_core)]}}
    icons = load_icons(cfg)
    for name, grid in [('source', sample['source_grid']), ('target', sample['target_grid']), ('prediction', prediction[0].cpu())]:
        render_grid(grid.numpy(), cfg, icons).save(cfg.output_dir / (name + '.png'))
    write_json(cfg.output_dir / 'demo.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description='Standalone OURS OpenMoji standard-head reproduction')
    parser.add_argument('command', choices=['verify', 'demo', 'evaluate', 'train'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output')
    parser.add_argument('--sample-id', default='test_000008')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    manifest = verify()
    if args.command == 'verify':
        return 0
    import torch
    torch.set_num_threads(4)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    output = output_directory(args.output)
    cfg = settings(output)
    cfg.batch_size = args.batch_size
    if cfg.batch_size < 1 or args.epochs < 1:
        raise ValueError('Positive batch size and epochs required')
    from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.training import seed_everything
    seed_everything(cfg.seed)
    write_json(output / 'execution.json', {'command': sys.argv, 'torch': torch.__version__, 'device': str(device),
                                         'package_git_commit': manifest['package_git_commit'], 'settings': cfg.to_dict()})
    if args.command == 'demo':
        report = demo(cfg, device, args.sample_id)
    else:
        from LightGenV2.tasks.t04_semantic_interaction.training import train, evaluate_selected
        checkpoint = ROOT / 'weights/best_checkpoint.pt'
        if args.command == 'train':
            cfg.epochs = args.epochs
            write_json(output / 'resolved_config.json', cfg.to_dict())
            train(cfg, device)
            checkpoint = output / 'best_checkpoint.pt'
        report = evaluate_selected(cfg, device, checkpoint)
        if args.command == 'evaluate':
            expected = json.loads((ROOT / 'reference/selected_checkpoint_test_evaluation.json').read_text())['metrics']['overall']
            actual = report['metrics']['overall']
            names = ['changed_cell_accuracy', 'edit_grid_iou', 'object_f1', 'scene_exact_match']
            errors = {name: abs(actual[name] - expected[name]) for name in names}
            check = {'absolute_errors': errors, 'tolerance': .005, 'passed': max(errors.values()) <= .005,
                     'note': 'Cross-device precision tolerance; source metrics were selected by periodic test.'}
            write_json(output / 'reference_comparison.json', check)
            if not check['passed']:
                raise RuntimeError(f'Reproduction differs from reference: {check}; inspect output, do not overwrite reference')
    assert_local_sources()
    print(json.dumps({'output_dir': str(output), 'result': report}, ensure_ascii=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
