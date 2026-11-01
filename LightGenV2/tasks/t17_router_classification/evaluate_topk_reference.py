"""Read-only, SHA-locked complete-split evaluation of existing actual-k PTs."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from .train_figure_revision import build_model, evaluate
from .readout_data import load_raw
from .model_four import encode_rgb
from .gpu_policy import require_authorized_gpu
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.profile.read_text())
    gpu = require_authorized_gpu(cfg['authorized_gpu_uuids'])
    torch.set_num_threads(4)
    task = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=False)
    for name, job in cfg['jobs'].items():
        out = args.out / name
        out.mkdir()
        path = task / job['checkpoint']
        if sha256(path) != job['sha256']:
            raise ValueError('Reference SHA mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        profile = payload['metadata']['profile']
        if profile['dataset'] != 'kather2016' or profile.get('top_k', 2) != job['top_k']:
            raise ValueError('Actual checkpoint k mismatch')
        model = build_model(profile).cuda().eval().requires_grad_(False)
        model.load_state_dict(payload['model'], strict=True)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        raw, identity = load_raw(Path(profile['data_root']), profile)
        result = dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT', checkpoint_reference=job['checkpoint'],
                      checkpoint_sha256=job['sha256'], actual_top_k=job['top_k'], additional_training_steps=0)
        for split, key in [('train', 'train'), ('val', 'val'), ('test', 'original_test_dev')]:
            fields = torch.cat([encode_rgb(torch.from_numpy(raw[split]['images'][i:i+64]).cuda())
                                for i in range(0, len(raw[split]['labels']), 64)])
            labels = torch.tensor(raw[split]['labels'], dtype=torch.long, device='cuda')
            result[key], arrays = evaluate(model, fields, labels, 64)
            if not np.all((arrays['route_power'] > 0).sum(1) == job['top_k']):
                raise RuntimeError('Actual hard top-k differs')
            np.savez_compressed(out / f'{split}_predictions.npz', ids=raw[split]['ids'], **arrays)
        assert all(torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items())
        write(out / 'metadata.json', dict(profile=profile, data=identity, pid=os.getpid(), gpu_uuid=gpu,
              source_commit=os.environ.get('T17_SOURCE_COMMIT'), source_sha256=sha256(Path(__file__))))
        write(out / 'model_audit.json', dict(strict=True, actual_top_k=job['top_k'], all_parameters_bitwise_unchanged=True,
              parameters=sum(p.numel() for p in model.parameters()), linear_count=sum(isinstance(m, torch.nn.Linear) for m in model.modules()),
              Conv_count=sum(isinstance(m, torch.nn.modules.conv._ConvNd) for m in model.modules())))
        write(out / 'result.json', result)
        print(json.dumps(dict(job=name, correct=result['original_test_dev']['correct'])), flush=True)
        del fields, labels, model, raw
        torch.cuda.empty_cache()
    write(args.out / 'status.json', dict(state='complete', pid=os.getpid()))


if __name__ == '__main__':
    main()
