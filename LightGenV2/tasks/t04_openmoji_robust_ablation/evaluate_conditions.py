"""Evaluate fixed VAL-selected checkpoints under five predeclared display conditions."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
from .profiles import install
from .train import BASE, OUTPUTS, save_json

CONDITIONS = (
    ('G1_ideal_17um', 'r0_base', 'r0_base'),
    ('G2_basic_device_proxy', 'r0_base', 'r3_ccd_dc30_grid'),
    ('G3_detector_noise', 'r1_ccd', 'r3_ccd_dc30_grid'),
    ('G4_coherent_dc30', 'r2_ccd_dc30', 'r3_ccd_dc30_grid'),
    ('G5_train_grid_proxy', 'r3_ccd_dc30_grid', 'r3_ccd_dc30_grid'),
)


def main():
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((BASE / 'resolved_config.json').read_text()))
    for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir', 'qwen_checkpoint',
                'prompt_cache_path', 'optical_base_config', 'legacy_warmstart_checkpoint'):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.num_workers = 0
    cfg.shared_readout_variant = 'standard'
    _, test = t.build_loaders(cfg)
    assert len(test.dataset) == 1000
    device = torch.device('cuda')
    results = []
    for label, trained, inference_profile in CONDITIONS:
        ckpt = OUTPUTS / trained / 'best.pt'
        assert (OUTPUTS / trained / 'report.json').exists(), trained
        payload = torch.load(ckpt, map_location='cpu', weights_only=False)
        model = t.build_model(cfg, device).eval()
        model.load_state_dict(payload['model'], strict=True)
        install(model, inference_profile)
        t._set_phase_dropout(model, False)
        metrics = t.evaluate_with_routes(model, test, cfg, device)[0]
        row = {'condition': label, 'trained_group': trained,
               'inference_profile': inference_profile,
               'checkpoint_epoch': payload['epoch'], 'metrics': metrics,
               'grid_proxy_not_exact_8um_propagation': inference_profile != 'r0_base'}
        results.append(row)
        print(json.dumps({'condition': label,
                          'changed_cell_accuracy': metrics['overall']['changed_cell_accuracy']}), flush=True)
        del model
        torch.cuda.empty_cache()
    save_json(OUTPUTS / 'five_conditions.json', {'status': 'complete',
              'source_reference_simulation_changed_cell_accuracy': 0.889,
              'conditions': results,
              'selection': 'Each checkpoint selected only on original TRAIN holdout; conditions predefined before TEST.'})


if __name__ == '__main__':
    main()
