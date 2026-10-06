"""Read-only CPU load of the original 17M physical checkpoint; no datasets or SDKs."""
import argparse
import hashlib
import json
from pathlib import Path

PIN = '5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac'
PARAMETERS = 17026642


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_sealed_cpu(checkpoint):
    import torch
    from .sealed_editor import build_sealed
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    with torch.device('cpu'):
        return build_sealed(saved).eval().requires_grad_(False)


def inspect_checkpoint(checkpoint, loader=None):
    checkpoint = Path(checkpoint)
    if digest(checkpoint) != PIN:
        raise ValueError('Not the fixed original 17M physical checkpoint; do not substitute adapted or historical weights')
    model = (loader or load_sealed_cpu)(checkpoint)
    count = sum(p.numel() for p in model.parameters())
    if model.kind != 'small' or count != PARAMETERS:
        raise ValueError('Original 17M architecture contract mismatch')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Verification requires CPU-only model loading')
    if digest(checkpoint) != PIN:
        raise RuntimeError('Checkpoint changed during verification')
    return dict(checkpoint_sha256=PIN, parameters=count, kind=model.kind, device='cpu',
                strict_load=True, read_only=True, datasets_read=False, hardware_used=False,
                scope='Original 5b4f 17M identity and strict load only; not adapted weights or performance reproduction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2))


if __name__ == '__main__':
    main()
