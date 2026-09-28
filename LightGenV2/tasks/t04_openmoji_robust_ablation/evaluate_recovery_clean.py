"""Evaluate VAL-selected OpenMoji recovery checkpoints once on normal TEST."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
from .profiles import install
from .train import BASE, OUTPUTS, save_json, sha

GROUPS = ('r0_base', 'r1_ccd', 'r2_ccd_dc30', 'r3_ccd_dc30_grid')


def main() -> None:
    output = OUTPUTS / 'five_conditions_clean_recovered_lowrank16.json'
    if output.exists():
        raise FileExistsError(f'Preserve existing evaluation: {output}')
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((BASE / 'resolved_config.json').read_text()))
    for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir', 'qwen_checkpoint',
                'prompt_cache_path', 'optical_base_config', 'legacy_warmstart_checkpoint'):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.num_workers = 0
    cfg.shared_readout_variant = 'lowrank16'
    _, test = t.build_loaders(cfg)
    assert len(test.dataset) == 1000
    device = torch.device('cuda')

    selected = {}
    for group in GROUPS:
        original = OUTPUTS / f'compact_lowrank16_{group}'
        if group == 'r0_base':
            selected[group] = {'run': original.name, 'criterion': 'base VAL-selected checkpoint'}
            continue
        recovery = OUTPUTS / f'compact_lowrank16_cleanrecover8_{group}'
        report = json.loads((recovery / 'report.json').read_text())
        assert report['status'] == 'complete' and report['test_evaluated'] is False
        better = report['best_clean_validation'] > report['initial_clean_validation']
        chosen = recovery if better else original
        selected[group] = {'run': chosen.name, 'criterion': 'clean TRAIN holdout',
                           'initial_clean_validation': report['initial_clean_validation'],
                           'best_clean_validation': report['best_clean_validation'],
                           'recovery_accepted': better}

    results = []
    for label, group in (('G1_ideal_17um', GROUPS[0]),
                         ('G2_direct_deployment_simulation', GROUPS[0]),
                         ('G3_detector_noise_training', GROUPS[1]),
                         ('G4_coherent_dc30_training', GROUPS[2]),
                         ('G5_train_grid_training', GROUPS[3])):
        run = OUTPUTS / selected[group]['run']
        checkpoint = run / 'best.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['group'] == group
        model = t.build_model(cfg, device).eval()
        model.load_state_dict(payload['model'], strict=True)
        install(model, 'r0_base')
        t._set_phase_dropout(model, False)
        metrics = t.evaluate_with_routes(model, test, cfg, device)[0]
        results.append({'condition': label, 'trained_group': group,
                        'selected_run': run.name, 'checkpoint_sha256': sha(checkpoint),
                        'inference_profile': 'r0_base', 'metrics': metrics})
        print(json.dumps({'condition': label,
                          'changed_cell_accuracy': metrics['overall']['changed_cell_accuracy']}), flush=True)
        del model
        torch.cuda.empty_cache()
    assert results[0]['checkpoint_sha256'] == results[1]['checkpoint_sha256']
    assert results[0]['metrics'] == results[1]['metrics']
    save_json(output, {'status': 'complete', 'selection': selected,
                      'inference_contract': 'all groups r0_base; no extra CCD/DC/raster stress',
                      'test_samples': 1000, 'test_used_for_epoch_selection': False,
                      'g2_experiment_status': 'not measured', 'conditions': results})


if __name__ == '__main__':
    main()
