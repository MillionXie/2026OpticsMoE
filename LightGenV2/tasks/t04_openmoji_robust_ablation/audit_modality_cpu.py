"""Fresh default CPU replay of the entire original TEST; no optical SDK."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import train as base, split_rank_head
from .profiles import assert_contract, install
from .train_modality_fusion import settings
from LightGenV2.tasks.t04_semantic_interaction import training as t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def write(name, data):
        path = args.output / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(path)

    write('progress.json', {'status': 'loading'})
    try:
        torch.set_num_threads(4)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        assert payload['group'] == 'r0_base'
        assert payload['settings']['editor_rank'] == 48
        cfg = settings(payload['settings'])
        model = split_rank_head.build_model(cfg, torch.device('cpu'))
        model.load_state_dict(payload['model'], strict=True)
        actual = {f'{modality}_core.block{block}': float(getattr(
            getattr(model, modality + '_core'), f'block{block}_optical_fusion').detach())
            for modality in ('language', 'vision') for block in (1, 2)}
        assert set(actual) == set(payload['fixed_fusion_alphas'])
        assert all(abs(actual[key] - payload['fixed_fusion_alphas'][key]) < 1e-6
                   for key in actual)
        assert sum(p.numel() for p in model.shared_readout.parameters()) == 240664
        assert sum(p.numel() for p in model.shared_readout.decoder.parameters()) == 30162
        assert_contract()
        install(model, 'r0_base')
        model.eval()
        t._set_phase_dropout(model, False)
        _, original = t.build_loaders(cfg)
        assert len(original.dataset) == 1000
        loader = DataLoader(original.dataset, batch_size=1, shuffle=False,
                            collate_fn=original.collate_fn, num_workers=0)

        class Tracked:
            dataset = loader.dataset

            def __len__(self):
                return len(loader)

            def __iter__(self):
                for count, batch in enumerate(loader, 1):
                    yield batch
                    if count % 50 == 0:
                        write('progress.json', dict(status='evaluating', samples=count,
                              elapsed_seconds=time.time() - started))

        with torch.no_grad():
            metrics, predictions, _ = t.evaluate_with_routes(
                model, Tracked(), cfg, torch.device('cpu'))
        torch.save(predictions, args.output / 'predictions.pt')
        write('report.json', dict(status='complete', checkpoint_sha256=base.sha(args.checkpoint),
              checkpoint_epoch=payload['epoch'], metrics=metrics, samples=1000,
              batch_size=1, device='cpu', fixed_fusion_alphas=actual,
              original_saved_gpu_score=payload['score'], physical_accuracy=None,
              development_only=True, elapsed_seconds=time.time() - started))
        write('progress.json', dict(status='complete', samples=1000, score=base.metric(metrics)))
    except Exception as exc:
        write('progress.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
