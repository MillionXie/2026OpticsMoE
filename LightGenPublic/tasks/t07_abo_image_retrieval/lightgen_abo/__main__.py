"""Portable public-review entry point. Only local assets; no full Qwen model."""
import argparse
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
from .data import _load_contract
from .io import verify_assets, write_json, sha256, source_commit
from .model import OpticalRetrieval
from .runtime import evaluate, preview, inspect_ccd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'evaluate', 'train'])
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/train.json'))
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--reference', type=Path, help='Original retrieval_features.pt for fixed-checkpoint numerical equivalence')
    parser.add_argument('--inspect-ccd', action='store_true', help='Read-only first-view-per-product decoder audit')
    args = parser.parse_args()
    verify_assets(args.assets)
    if args.command == 'verify':
        print('Local asset hashes verified. Full Qwen weights are not needed.')
        return
    if args.data is None or args.output is None:
        parser.error('--data and --output are required')
    if any(v is not None and v < 1 for v in (args.epochs, args.steps, args.batch_size)):
        parser.error('Counts must be positive')
    if args.output.exists():
        parser.error('Output exists; choose a new directory (old results are never overwritten)')
    args.output.mkdir(parents=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    checkpoint = args.checkpoint or args.assets / 'best.pt'
    model = None
    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = OpticalRetrieval(payload['metadata'])
        model.load_state_dict(payload['state_dict'], strict=True)
        del payload
        model.to(device)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(args.assets / 'processor'), local_files_only=True)
        samples, _ = _load_contract(args.data)
        train_samples = [s for s in samples if s.split == 'train']
        test_samples = [s for s in samples if s.split == 'test']
        execution = dict(source_commit=source_commit(), command=sys.argv, pid=os.getpid(),
                         torch=torch.__version__, python=sys.version, device=str(device),
                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         checkpoint_sha256=sha256(checkpoint),
                         dataset_manifest_sha256=sha256(args.data / 'data/abo_similarity10_manifest.csv'),
                         model_audit=model.audit())
        if args.command == 'train':
            config = json.loads(args.config.read_text(encoding='utf-8'))
            for key in ('epochs', 'steps'):
                if getattr(args, key) is not None:
                    config['adapt'][key] = getattr(args, key)
            execution['config'] = config
        write_json(args.output / 'execution.json', execution)
        if args.command == 'train':
            from .training import train
            train(model, processor, train_samples, test_samples, device, config, args.output, args.batch_size)
        normal = evaluate(model, processor, train_samples, test_samples, device, args.batch_size, args.output)
        equivalence = None
        if args.reference:
            original = torch.load(args.reference, map_location='cpu', weights_only=True)
            current = torch.load(args.output / 'retrieval_features.pt', map_location='cpu', weights_only=True)
            equivalence = {}
            for split in ('train', 'test'):
                if original[split + '_ids'] != current[split + '_ids']:
                    raise ValueError('Reference feature identity/order differs')
                a, b = original[split].float(), current[split].float()
                if a.shape != b.shape:
                    raise ValueError('Reference feature dimensions differ')
                equivalence[split] = dict(bitwise_equal=torch.equal(a, b), max_absolute_error=float((a-b).abs().max()))
                if not torch.allclose(a, b, rtol=0, atol=1e-6):
                    raise RuntimeError(f'Numerical equivalence failed: {equivalence}')
            write_json(args.output / 'equivalence.json', equivalence)
        model.set_remove_optical(True)
        try:
            removed = evaluate(model, processor, train_samples, test_samples, device, args.batch_size)
        finally:
            model.set_remove_optical(False)
        preview(model, args.output)
        if args.inspect_ccd:
            inspect_ccd(model, processor, test_samples, device, args.output)
        report = dict(status='complete', metrics=normal, remove_optical_same_weights=removed,
                      optical_removal_hit1_drop_percentage_points=100 * (normal['hit_at_1'] - removed['hit_at_1']),
                      audit=model.audit(), test_selected=True, numerical_equivalence=equivalence,
                      checkpoint_origin_note='Fixed imported test-selected weight or best continuation; not independent confirmation')
        write_json(args.output / 'final_report.json', report)
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        write_json(args.output / 'failure.json', dict(error=str(exc), type=type(exc).__name__))
        raise
    finally:
        if model is not None:
            del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # Process exit, including Ctrl+C, destroys this process's CUDA context.


if __name__ == '__main__':
    main()
