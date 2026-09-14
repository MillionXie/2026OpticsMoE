"""Fixed-weight diagnostics of the EXISTING Vision outer skip, never training.

Every intervention rebuilds BOTH gallery and query descriptors. No checkpoint is
modified/saved; the default model/optics source and deployment graph are unchanged.
Discarding a computed result here is not an accelerated implementation benchmark.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch
from torch.nn import functional as F

from .cli import autocast
from .io import inputs, picture, verify_assets, evaluation_checkpoint, sha256, source_commit, write_json, write_csv
from .model import OpticalRetrieval
from .retrieval_screen import load_screen, rank_instances, OPTICS_SHA256

MODES = ('normal', 'no_outer_skip', 'skip_only', 'remove_vision_optics',
         'remove_language_optics', 'remove_all_optics')


def skip_statistics(original, projected, gate):
    """Per-image feature RMS, not optical power or attribution percentage."""
    a, b = original.float().flatten(1), (gate * projected).float().flatten(1)
    ar, br = a.square().mean(1).sqrt(), b.square().mean(1).sqrt()
    return dict(input_rms=ar, gated_update_rms=br,
        update_to_input_rms=br/ar.clamp_min(1e-8),
        cosine=F.cosine_similarity(a, b, dim=1))


@contextmanager
def intervention(model, mode, observations):
    if mode not in MODES:
        raise ValueError('Unknown outer-skip diagnostic')
    previous = (model.vision.remove_optical, model.language.remove_optical)
    captured = {}
    handles = []
    try:
        model.vision.remove_optical = mode in ('remove_vision_optics', 'remove_all_optics')
        model.language.remove_optical = mode in ('remove_language_optics', 'remove_all_optics')
        def projection_hook(module, args, output):
            captured['projection'] = output
        def vision_hook(module, args, output):
            original, projected = args[0], captured.pop('projection')
            gate = module.residual_logit.sigmoid()
            values = skip_statistics(original, projected, gate)
            observations.extend({k: float(v[i]) for k,v in values.items()} for i in range(len(original)))
            if mode == 'no_outer_skip':
                return (gate * projected).to(original.dtype)
            if mode == 'skip_only':
                return original
            return output  # Preserve exact original normal forward, including BF16 rounding.
        handles.append(model.vision.output_adapter.register_forward_hook(projection_hook))
        handles.append(model.vision.register_forward_hook(vision_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
        model.vision.remove_optical, model.language.remove_optical = previous


@torch.inference_mode()
def run(args):
    from transformers import AutoProcessor
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_sha = sha256(args.manifest)
    protocol, groups = load_screen(args.manifest, args.data)
    verify_assets(args.assets)
    path, digest = evaluation_checkpoint(args.assets, args.checkpoint, args.expected_checkpoint_sha256)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if sha256(path) != digest or sha256(Path(__file__).with_name('optics.py')) != OPTICS_SHA256:
        raise ValueError('Pinned checkpoint/physical source changed')
    model = OpticalRetrieval(payload['metadata'])
    model.load_state_dict(payload['state_dict'], strict=True)
    audit = model.audit()
    if audit['frontend_trainable_parameters'] or audit['descriptor_dimension'] != 64:
        raise ValueError('Audit requires original frozen frontend/64D model')
    processor = AutoProcessor.from_pretrained(str(args.assets/'processor'), local_files_only=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    rows = groups['gallery'] + groups['query']
    identity = dict(source_commit=source_commit(), checkpoint_sha256=digest,
        manifest_sha256=manifest_sha, protocol=protocol['protocol'], command=sys.argv,
        model_audit=audit, pid=os.getpid(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        torch=torch.__version__, python=sys.version,
        outer_gate=float(model.vision.residual_logit.sigmoid()),
        note='Same fixed checkpoint, no optimization. Both galleries and queries rebuilt per intervention. No checkpoint saved. Not a speed benchmark.')
    args.output.mkdir(parents=True)
    status = dict(status='running', **identity)
    write_json(args.output/'status.json', status)
    write_json(args.output/'execution.json', identity)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    started = time.time()
    results = {}
    try:
        model.to(device).eval().requires_grad_(False)
        for mode in MODES:
            vectors, observations = [], []
            with intervention(model, mode, observations):
                for start in range(0, len(rows), args.batch_size):
                    images = [picture(args.data/r['image_path'], model.metadata['input_preprocessing'])
                              for r in rows[start:start+args.batch_size]]
                    with autocast(device):
                        vectors.append(model(inputs(processor, images, device)).float().cpu())
            z = torch.cat(vectors)
            if len(observations) != len(rows) or not torch.isfinite(z).all():
                raise RuntimeError('Incomplete/nonfinite diagnostic')
            metrics, predictions = rank_instances(z, rows)
            if mode == 'normal' and abs(metrics['hit_at_1']-args.expected_hit_at_1) > 1e-8:
                raise RuntimeError('Normal reference mismatch; do not interpret interventions')
            write_csv(args.output/f'{mode}_predictions.csv', predictions)
            records = [dict(sample_id=r['sample_id'], split=r['split'], **s) for r,s in zip(rows, observations)]
            write_csv(args.output/f'{mode}_scales.csv', records)
            scales = {split: {k: sum(s[k] for r,s in zip(rows,observations) if r['split']==split)
                / sum(r['split']==split for r in rows) for k in observations[0]}
                for split in ('gallery','query')}
            results[mode] = dict(metrics=metrics, mean_per_sample_feature_statistics=scales,
                semantic=('Discard whole Vision processed update, retaining original patch input; Language optics still active'
                          if mode=='skip_only' else mode))
            write_json(args.output/'partial_report.json', dict(identity, results=results))
            print(json.dumps(dict(mode=mode, **results[mode])), flush=True)
            status['last_completed_mode'] = mode
            write_json(args.output/'status.json', status)
        if sha256(path) != digest or sha256(args.manifest) != manifest_sha:
            raise RuntimeError('Input identity changed during audit')
        report = dict(identity, status='complete', results=results, elapsed_seconds=time.time()-started)
        write_json(args.output/'final_report.json', report)
        status['status'] = 'complete'
    except BaseException as exc:
        status.update(status='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        write_json(args.output/'status.json', status)
        del model
        if device.type=='cuda':
            torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','manifest','assets','checkpoint','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-checkpoint-sha256', required=True)
    p.add_argument('--expected-hit-at-1', type=float, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    args=p.parse_args()
    if args.batch_size < 1 or not 0 <= args.expected_hit_at_1 <= 1:
        p.error('Positive batch and expected metric in [0,1] required')
    run(args)


if __name__=='__main__':
    main()
