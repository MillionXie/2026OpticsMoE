"""Full original TEST CPU batch1 audit, never choose a low-scoring subset."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
from . import train as base, split_rank_head
from .profiles import install, assert_contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, data):
        path = args.output / name
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2) + '\n')
        temp.replace(path)
    started = time.time()
    write('progress.json', {'status': 'loading'})
    try:
        torch.set_num_threads(4)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        cfg = Settings.__new__(Settings)
        cfg.__dict__.update(payload['settings'])
        for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir', 'qwen_checkpoint',
                    'prompt_cache_path', 'optical_base_config', 'legacy_warmstart_checkpoint'):
            setattr(cfg, key, Path(getattr(cfg, key)))
        cfg.num_workers = 0
        model = split_rank_head.build_model(cfg, torch.device('cpu'))
        model.load_state_dict(payload['model'], strict=True)
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
                        write('progress.json', {'status': 'evaluating', 'samples': count,
                                               'elapsed_seconds': time.time()-started})
        metrics, predictions, _ = t.evaluate_with_routes(model, Tracked(), cfg, torch.device('cpu'))
        gates = {f'{name}.block{block}': float(getattr(getattr(model, name), f'block{block}_optical_fusion'))
                 for name in ('language_core', 'vision_core') for block in (1, 2)}
        assert all(abs(v-payload['fixed_fusion_alpha']) < 1e-6 for v in gates.values())
        write('report.json', {'status': 'complete', 'checkpoint_sha256': base.sha(args.checkpoint),
                              'metrics': metrics, 'samples': 1000, 'batch_size': 1, 'device': 'cpu',
                              'fixed_alpha': gates, 'original_saved_gpu_score': payload.get('score'),
                              'physical_accuracy': None, 'development_only': True,
                              'elapsed_seconds': time.time()-started})
        torch.save(predictions, args.output/'predictions.pt')
        write('progress.json', {'status': 'complete', 'samples': 1000, 'score': base.metric(metrics)})
    except Exception as exc:
        write('progress.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
