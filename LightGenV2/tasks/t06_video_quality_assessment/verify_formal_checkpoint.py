"""Verify the fixed Spatial/Temporal physical checkpoint on CPU, without data or SDKs."""
import argparse
import json
from pathlib import Path

TARGETS = ('spatial', 'temporal')


def inspect_checkpoint(target, checkpoint, loader=None):
    if target not in TARGETS:
        raise ValueError('Choose the fixed spatial or temporal physical model')
    from .lab_runtime import PINS, load_model
    # Existing loader checks exact checkpoint SHA, architecture and strict state
    # keys. Do not substitute another profile or update the pin on mismatch.
    model, settings = (loader or load_model)(target, Path(checkpoint), 'cpu')
    return {'target': target, 'checkpoint_sha256': PINS[target]['sha256'],
            'architecture': settings.architecture_label,
            'parameters': sum(p.numel() for p in model.parameters()),
            'device': 'cpu', 'strict_load': True, 'read_only': True,
            'scope': 'Fixed checkpoint identity and model load only; not data, performance, SDK or optical validation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=TARGETS, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.target, args.checkpoint), indent=2))


if __name__ == '__main__':
    main()
