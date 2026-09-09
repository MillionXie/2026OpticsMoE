"""Fixed-weight, full-test recheck of aligned Qwen or optical SALICON (no training)."""
import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.modeling import preprocess_vision
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import SaliencyAccumulator, density_from_logits
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import FrozenQwenVisionTeacher
from .aligned_baseline import AlignedReadout
from .modeling import build_student, load_vision_backbone, sha256_file, architecture_label
from .settings import load_settings, save_resolved_config
from .reproduce_baseline import independent_cc
from .run import _seed
from .training import _write_json


def load_hashed_checkpoint(path):
    """Bind provenance to exactly the bytes loaded, even if best is updated later."""
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    payload = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    return payload, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=['qwen', 'optical'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    s = load_settings(args.config)
    s.output_dir = args.run_dir.resolve()
    if s.output_dir.exists():
        raise FileExistsError('Use a new recheck directory; preserve previous evidence')
    if args.batch_size < 1:
        raise ValueError('batch size must be positive')
    _seed(42)
    s.random_seed, s.inference_batch_size, s.num_workers = 42, args.batch_size, 2
    s.local_files_only, s.download = True, False
    s.output_dir.mkdir(parents=True)
    save_resolved_config(s)
    bundle = prepare_salicon(s, persist=True)
    if len(bundle.validation_records) != 5000:
        raise RuntimeError('Require all 5000 public-test records')
    _, loader = legacy.build_loaders(bundle, s, training=False)
    loaded = load_vision_backbone(s, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    s.resolve_architecture(loaded.model)
    loaded.model.requires_grad_(False).eval()
    payload, checkpoint_digest = load_hashed_checkpoint(args.checkpoint)
    aligned = AlignedReadout(s.vision_hidden_size)
    if args.system == 'qwen':
        if payload['architecture'] != 'frozen_qwen24_adapter192_identical_progressive_decoder_v1':
            raise ValueError('Expected aligned-head Qwen checkpoint, not legacy head')
        aligned.load_state_dict(payload['head'], strict=True)
        model = FrozenQwenVisionTeacher(loaded, aligned.to(loaded.device))
    else:
        if payload['architecture'] != architecture_label(s):
            raise ValueError('Optical checkpoint/config architecture mismatch')
        model = build_student(loaded, s)
        model.core.load_state_dict(payload['core'], strict=True)
        model.head.load_state_dict(payload['saliency_head'], strict=True)
        model.core.set_phase_dropout_active(False)
    model.eval()
    accumulator, rows = SaliencyAccumulator(), []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = preprocess_vision(loaded.processor, batch['images'], loaded.device)
                logits = model(inputs['pixel_values'], inputs['image_grid_thw'])[0]
                target = batch['density'].to(loaded.device)
                accumulator.update(logits, target, batch['fixation'].to(loaded.device))
                cc = independent_cc(density_from_logits(logits).cpu().numpy(), target.cpu().numpy())
                rows.extend({'sample_id': sid, 'cc_float64': float(v)} for sid, v in zip(batch['sample_ids'], cc))
                if len(rows) % 512 < args.batch_size:
                    print(f"[{args.system} recheck] {len(rows)}/5000 CC64={np.mean([r['cc_float64'] for r in rows]):.7f}", flush=True)
    finally:
        if args.system == 'qwen':
            model.close()
        else:
            model.restore_native()
    if len(rows) != 5000 or len({r['sample_id'] for r in rows}) != 5000:
        raise RuntimeError('Invalid test identities/count')
    with (s.output_dir/'per_image_cc.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id','cc_float64'])
        writer.writeheader(); writer.writerows(rows)
    metrics = accumulator.compute()
    cc64 = float(np.mean([r['cc_float64'] for r in rows]))
    report = {'mode': 'fixed_weight_reevaluation_no_training', 'system': args.system,
              'metrics': metrics, 'independent_float64_cc': cc64,
              'cc_implementation_difference': abs(cc64-metrics['cc']),
              'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_digest,
              'checkpoint_hash_contract': 'SHA256 of the exact in-memory bytes deserialized before evaluation; source path may subsequently change',
              'selected_epoch': payload['epoch'], 'architecture': payload['architecture'],
              'aligned_readout_parameter_audit': aligned.parameter_audit(),
              'config_sha256': sha256_file(args.config),
              'test_ids_sha256': hashlib.sha256('\n'.join(r['sample_id'] for r in rows).encode()).hexdigest(),
              'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'command': [sys.executable,'-m',__spec__.name,*sys.argv[1:]],
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
              'selection_biased': True, 'split': 'official val2014 as public test; no independent validation',
              'speed_and_power': 'not measured'}
    _write_json(s.output_dir/'reproduction.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
