"""Compare the full-head ablation weights on the same TRAIN holdout stresses.

The optical perturbations are normally training-only.  For this diagnostic we
activate them explicitly while keeping all learned child modules in eval mode.
No TEST samples or checkpoint selection are involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
from .profiles import PROFILES, assert_contract, install
from .train import BASE, OUTPUTS, SOURCE, SOURCE_SHA, metric, save_json, sha


GROUPS = ('r0_base', 'r1_ccd', 'r2_ccd_dc30', 'r3_ccd_dc30_grid')


def _settings(output_dir: Path) -> Settings:
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((BASE / 'resolved_config.json').read_text()))
    for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir',
                'qwen_checkpoint', 'prompt_cache_path', 'optical_base_config',
                'legacy_warmstart_checkpoint'):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.output_dir = output_dir
    cfg.num_workers = 0
    cfg.shared_readout_variant = 'standard'
    return cfg


def _activate_stress_only(model, profile: str) -> None:
    if profile == 'r0_base':
        return
    # profiles.install checks the model flag for router CCD noise, while the
    # optical path checks its own flag for DC/CCD perturbations. Setting these
    # two flags directly does not switch dropout/readout child modules to train.
    model.training = True
    for path in model._optical_paths():
        path.training = True
        path.core.router.training = True


def run(output: Path, seeds: tuple[int, ...], noise_scale: float,
        pixel_shift: int, weight_set: str = 'legacy') -> None:
    assert_contract()
    output.mkdir(parents=True, exist_ok=False)
    cfg = _settings(output)
    device = torch.device('cuda')
    train, _ = t.build_loaders(cfg)
    if weight_set == 'user889':
        split = json.loads((OUTPUTS / 'fullhead_probe1_r1_ccd_20260929/split.json').read_text())
        checkpoint_paths = {
            'r0_base': SOURCE,
            **{group: OUTPUTS / f'fullhead889_strong_{group}_20260929/best.pt'
               for group in GROUPS[1:]}}
        weight_groups = GROUPS
    elif weight_set == 'user889_equalstep_control':
        split = json.loads((OUTPUTS / 'fullhead_probe1_r1_ccd_20260929/split.json').read_text())
        checkpoint_paths = {
            'r0_equalstep': OUTPUTS / 'fullhead889_clean_equalstep_control_20260929/best.pt'}
        weight_groups = ('r0_equalstep',)
    elif weight_set == 'legacy':
        split = json.loads((OUTPUTS / 'r0_base/split.json').read_text())
        checkpoint_paths = {group: OUTPUTS / group / 'best.pt' for group in GROUPS}
        weight_groups = GROUPS
    else:
        raise ValueError(weight_set)
    ids = split['validation']
    assert len(ids) == 1000 and not set(ids).intersection(split['fit'])
    loader = DataLoader(Subset(train.dataset, ids), batch_size=32,
                        collate_fn=train.collate_fn, num_workers=0)
    results = []
    for group in weight_groups:
        checkpoint = checkpoint_paths[group]
        if weight_set == 'user889' and group == 'r0_base':
            assert sha(checkpoint) == SOURCE_SHA
        else:
            directory = checkpoint.parent
            report = json.loads((directory / 'report.json').read_text())
            protocol = json.loads((directory / 'protocol.json').read_text())
            assert sha(checkpoint) == report['best_sha256']
            assert json.loads((directory / 'split.json').read_text()) == split
            if weight_set in ('user889', 'user889_equalstep_control'):
                assert report['source_sha256'] == SOURCE_SHA
                assert protocol['source_sha256'] == SOURCE_SHA
            else:
                assert report['source_sha256'] == protocol['initial_sha256']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'group' in payload:
            assert payload['group'] == ('r0_base' if group == 'r0_equalstep' else group)
        for profile in GROUPS:
            model = t.build_model(cfg, device)
            model.load_state_dict(payload['model'], strict=True)
            assert sum(p.numel() for p in model.shared_readout.parameters()) == 381976
            install(model, profile, noise_scale=noise_scale,
                    pixel_shift=pixel_shift)
            model.eval()
            t._set_phase_dropout(model, False)
            _activate_stress_only(model, profile)
            for seed in ((seeds[0],) if profile == 'r0_base' else seeds):
                torch.manual_seed(seed)
                with torch.inference_mode():
                    metrics, _, _ = t.evaluate_with_routes(model, loader, cfg, device)
                row = {'weight_group': group, 'stress_profile': profile,
                       'seed': seed, 'changed_cell_accuracy': metric(metrics),
                       'router_audit': metrics.get('router_audit')}
                results.append(row)
                save_json(output / 'report.json', {
                    'status': 'running', 'split': 'TRAIN holdout validation',
                    'weight_set': weight_set,
                    'samples': len(ids), 'seeds': seeds,
                    'noise_scale': noise_scale, 'pixel_shift': pixel_shift,
                    'profile_contract': PROFILES,
                    'stress_forced_on_during_eval': True,
                    'checkpoint_selection': 'Existing TRAIN-holdout-selected best, no TEST selection',
                    'rows': results})
                print(json.dumps(row), flush=True)
            del model
            torch.cuda.empty_cache()
    save_json(output / 'report.json', {
        'status': 'complete', 'split': 'TRAIN holdout validation',
        'weight_set': weight_set,
        'samples': len(ids), 'seeds': seeds,
        'noise_scale': noise_scale, 'pixel_shift': pixel_shift,
        'profile_contract': PROFILES,
        'stress_forced_on_during_eval': True,
        'checkpoint_selection': 'Existing TRAIN-holdout-selected best, no TEST selection',
        'rows': results})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1042, 1043])
    parser.add_argument('--noise-scale', type=float, default=1.0)
    parser.add_argument('--pixel-shift', type=int, default=0)
    parser.add_argument('--weight-set',
                        choices=('legacy', 'user889', 'user889_equalstep_control'),
                        default='legacy')
    args = parser.parse_args()
    run(args.output, tuple(args.seeds), args.noise_scale, args.pixel_shift,
        args.weight_set)


if __name__ == '__main__':
    main()
