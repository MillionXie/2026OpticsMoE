"""Read-only CPU load of explicitly selected 17M checkpoints; no datasets or SDKs."""
import argparse
import hashlib
import json
from pathlib import Path

PIN = '5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac'
PARAMETERS = 17026642
ADAPTED_PIN = 'eeeca764b11a23613459fdae0a190fdba362a6e3bdeed9180279d9031c297a14'


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


def inspect_checkpoint(checkpoint, loader=None, variant='original'):
    checkpoint = Path(checkpoint)
    if variant not in ('original', 'decoder-adapted'):
        raise ValueError('Unknown formal checkpoint variant')
    expected = PIN if variant == 'original' else ADAPTED_PIN
    if digest(checkpoint) != expected:
        raise ValueError('Not the explicitly selected 17M checkpoint; do not substitute other weights')
    model = (loader or load_sealed_cpu)(checkpoint)
    count = sum(p.numel() for p in model.parameters())
    if model.kind != 'small' or count != PARAMETERS:
        raise ValueError('Original 17M architecture contract mismatch')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Verification requires CPU-only model loading')
    if digest(checkpoint) != expected:
        raise RuntimeError('Checkpoint changed during verification')
    return dict(checkpoint_sha256=expected, variant=variant, parameters=count, kind=model.kind, device='cpu',
                strict_load=True, read_only=True, datasets_read=False, hardware_used=False,
                scope='Explicitly selected original 5b4f or adapted eeec 17M identity and strict load only; not performance reproduction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--variant', choices=('original', 'decoder-adapted'), default='original')
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint, variant=args.variant), indent=2))


if __name__ == '__main__':
    main()
