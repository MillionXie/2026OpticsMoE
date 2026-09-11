"""Offline averaging of compatible students; emits ONE normal optical model.

Not prediction ensembling. Optical raw parameters are averaged just like the
electronic parameters; geometry, propagation, frozen frontend and metadata stay
identical. Only use nearby checkpoints sharing a training initialization.
"""
import argparse
import copy
from pathlib import Path
import torch
from .io import evaluation_checkpoint, sha256, source_commit, write_json


def average_payloads(left, right, right_weight=.5):
    if not 0 < right_weight < 1:
        raise ValueError('Mixing weight must be strictly between zero and one')
    if left['metadata'] != right['metadata']:
        raise ValueError('Cannot average different model/optical/input contracts')
    if left['metadata'].get('fusion_alpha_min', 0) <= .4:
        raise ValueError('This experiment requires alpha strictly above .4')
    a, b = left['state_dict'], right['state_dict']
    if a.keys() != b.keys():
        raise ValueError('State keys differ')
    result = {}
    for name, value in a.items():
        other = b[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f'State shape/dtype differs: {name}')
        if not torch.isfinite(value).all() or not torch.isfinite(other).all():
            raise ValueError(f'Nonfinite state: {name}')
        if name.startswith('frontend.') or not value.is_floating_point():
            if not torch.equal(value, other):
                raise ValueError(f'Frozen frontend/buffer differs: {name}')
            result[name] = value.clone()
        else:
            result[name] = torch.lerp(value.double(), other.double(), right_weight).to(value.dtype)
    # Never inherit either parent's test score, optimizer, or auxiliary head.
    return dict(metadata=copy.deepcopy(left['metadata']), state_dict=result,
                epoch=-1, stage='weight_average', selection_variant='unevaluated',
                weight_average=dict(right_weight=right_weight, prediction_ensemble=False,
                                    phase_space='raw; physical phase remains 2*pi*sigmoid(raw)',
                                    teacher_at_inference=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--left-sha256', required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--right-sha256', required=True)
    p.add_argument('--right-weight', type=float, default=.5)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    paths = [evaluation_checkpoint(None, args.left, args.left_sha256),
             evaluation_checkpoint(None, args.right, args.right_sha256)]
    payloads = [torch.load(path, map_location='cpu', weights_only=True) for path, _ in paths]
    if any(sha256(path) != digest for path, digest in paths):
        raise RuntimeError('Source checkpoint changed during loading')
    result = average_payloads(*payloads, args.right_weight)
    from .model import OpticalRetrieval
    model = OpticalRetrieval(result['metadata'])
    model.load_state_dict(result['state_dict'], strict=True)
    audit = model.audit()
    if audit['attention_modules'] or audit['native_transformer_modules'] or audit['capture_count'] != 6 or audit['top_k'] != 2:
        raise RuntimeError('Inference architecture contract violated')
    result['weight_average']['sources'] = [dict(path=str(path), sha256=digest) for path, digest in paths]
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output/'best.pt'
    torch.save(result, target)
    write_json(args.output/'average_report.json', dict(status='built_not_evaluated',
               source_commit=source_commit(), checkpoint_sha256=sha256(target),
               audit=audit, **result['weight_average']))
    print('checkpoint_sha256='+sha256(target), flush=True)


if __name__ == '__main__':
    main()
