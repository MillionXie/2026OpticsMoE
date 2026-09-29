"""One common clean TEST pass for the four preselected full-head checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from LightGenV2.tasks.t04_semantic_interaction import training as t
from .evaluate_matched_stress import GROUPS, _settings
from .profiles import install
from .train import OUTPUTS, SOURCE, SOURCE_SHA, save_json, sha


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    cfg = _settings(output)
    cfg.shared_readout_variant = 'standard'
    _, test = t.build_loaders(cfg)
    assert len(test.dataset) == 1000
    paths = {'r0_base': SOURCE}
    paths.update({g: OUTPUTS / f'fullhead889_strong_{g}_20260929/best.pt'
                  for g in GROUPS[1:]})
    results = []
    for group in GROUPS:
        checkpoint = paths[group]
        if group == 'r0_base':
            assert sha(checkpoint) == SOURCE_SHA
        else:
            report = json.loads((checkpoint.parent / 'report.json').read_text())
            assert report['status'] == 'complete'
            assert report['test_evaluated'] is False
            assert report['source_sha256'] == SOURCE_SHA
            assert report['best_sha256'] == sha(checkpoint)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = t.build_model(cfg, torch.device('cuda')).eval()
        model.load_state_dict(payload['model'], strict=True)
        assert sum(p.numel() for p in model.shared_readout.parameters()) == 381976
        install(model, 'r0_base')
        t._set_phase_dropout(model, False)
        with torch.inference_mode():
            metrics = t.evaluate_with_routes(model, test, cfg, torch.device('cuda'))[0]
        results.append({'group': group, 'checkpoint': str(checkpoint),
                        'checkpoint_sha256': sha(checkpoint), 'metrics': metrics})
        print(json.dumps({'group': group, 'clean_test_accuracy':
                          metrics['overall']['changed_cell_accuracy']}), flush=True)
        del model
        torch.cuda.empty_cache()
    save_json(output, {'status': 'complete', 'inference_profile': 'r0_base',
              'all_weights_preselected_by_train_holdout': True,
              'test_used_for_epoch_selection': False, 'test_samples': 1000,
              'g1_g2_same_weight': True, 'g2_hardware_measured': False,
              'results': results})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
