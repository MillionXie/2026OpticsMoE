"""Prepare an unmeasured initialization by removing the projection-output LN."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess

if __package__:
    from .data import sha256
    from .model import LightGenTemporal, load_lightgen
    from .run import ROOT, save_checkpoint
else:
    from data import sha256
    from model import LightGenTemporal, load_lightgen
    from run import ROOT, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite initialization: ' + str(args.output))
    parent, parent_metadata = load_lightgen(args.source)
    if parent.cfg.readout_mode != 'five_levels' or not parent.cfg.visual_output_norm:
        raise ValueError('A five-level checkpoint with projection-output LN is required')
    model = LightGenTemporal(replace(parent.cfg, visual_output_norm=False))
    source = parent.state_dict()
    removed = sorted(set(source) - set(model.state_dict()))
    if removed != ['visual_input_norm.bias', 'visual_input_norm.weight']:
        raise ValueError('Unexpected state difference during LN removal')
    model.load_state_dict({name: source[name] for name in model.state_dict()}, strict=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    metadata = {
        'role': 'UNMEASURED_LN_REMOVAL_INITIALIZATION',
        'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT', commit),
        'source_SHA256': {Path(__file__).name: sha256(__file__)},
        'dataset': parent_metadata['dataset'],
        'model_config': asdict(model.cfg),
        'parent_checkpoint': {'path': str(args.source), 'sha256': sha256(args.source)},
        'removed_tensors': removed,
        'measured_metrics_available': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, model.checkpoint(**metadata))
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
