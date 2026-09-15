"""Synthetic-input execution check; never reports dataset accuracy."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    source = Path(__file__).resolve().parents[1] / 'EuroSAT_MoE_D2NN/code'
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from PIL import Image
    from experiments.vision_transfer.settings import load_settings
    from experiments.vision_transfer.backbone import load_backbone
    from experiments.vision_transfer import model as m
    from experiments.expert_merge import core as c
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import seed_everything
    a.out.mkdir(parents=True, exist_ok=False)
    settings = load_settings(source / 'experiments/vision_transfer/config.yaml')
    settings.cache_dir = Path(a.cache)
    settings.output_dir = a.out
    settings.classification_head_seed = 1042
    settings.router_optimization_seed = settings.split_seed = 42
    settings.num_workers = 0
    torch.set_num_threads(4)
    seed_everything(42)
    loaded = load_backbone(settings, torch.device('cuda'))
    expected_stem = '3f494085f1c65fcb5c691d8fc5c048e4cb5d925cb220a1d54f2a04565fe3f477'
    assert loaded.source_metadata['frozen_stem_sha256'] == expected_stem
    rng = np.random.default_rng(42)
    images = [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    inputs = loaded.processor(images=images, return_tensors='pt').to('cuda')
    assert set(inputs) == {'pixel_values', 'image_grid_thw'}
    rows = []
    for architecture in ['moe', 'd2nn']:
        seed_everything(42)
        replacement, head = c.build(loaded, settings) if architecture == 'moe' else m.build(loaded, settings, architecture)
        resources = m.resource_report(replacement, head)
        assert sum(resources['parameter_counts'].values()) == {'moe': 1186804, 'd2nn': 1164408}[architecture]
        cases = ['automatic', 'uniform', 'isolated'] if architecture == 'moe' else ['automatic']
        for mode in cases:
            for param in m.named(replacement, head).values():
                param.grad = None
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=settings.amp_enabled):
                logits = c.forward(loaded, replacement, head, inputs, mode,
                                   torch.tensor([0, 1], device='cuda') if mode == 'isolated' else None) if architecture == 'moe' else m.predict(loaded, replacement, head, inputs)
                loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0, 1], device='cuda'))
            assert logits.shape == (2, 10) and bool(torch.isfinite(loss))
            loss.backward()
            grads = {}
            for name, param in m.named(replacement, head).items():
                if param.grad is not None:
                    assert bool(torch.isfinite(param.grad).all()), name
                    group = m.group_of(name)
                    grads[group] = grads.get(group, 0.) + float(param.grad.float().square().sum())
            route = m.routes(replacement).get('vision')
            power_error = None
            if route is not None:
                power_error = float((route['weights'].square().sum(1)-1).abs().max())
                assert power_error < 1e-5
            rows.append(dict(architecture=architecture, mode=mode, logits=logits.detach().float().cpu().tolist(),
                             loss=float(loss), gradient_squared_norm_by_group=grads, power_error=power_error,
                             resources=resources))
        replacement.close()
        del replacement, head, logits, loss
        torch.cuda.empty_cache()
    report = dict(scope='synthetic-input forward/backward; no dataset or trained weights', results=rows,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
                  command=sys.argv, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  settings=vars(settings),
                  source_hashes={str(x.relative_to(source)):hashlib.sha256(x.read_bytes()).hexdigest()
                                 for x in source.rglob('*') if x.is_file() and x.suffix in {'.py', '.yaml', '.json'}})
    (a.out / 'result.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(dict(state='complete', cases=len(rows), frozen_stem_match=True)), flush=True)


if __name__ == '__main__':
    main()
