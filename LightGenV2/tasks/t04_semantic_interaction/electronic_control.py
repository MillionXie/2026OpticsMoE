"""Historical matched electronic control; explicitly TEST-selected development run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.settings import load_settings
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.training import train, test


def run(args: argparse.Namespace):
    settings = load_settings(args.config)
    if settings.optical_enabled:
        raise ValueError('Matched electronic control requires optical_enabled=false')
    if args.run_dir is not None:
        settings.output_dir = args.run_dir.expanduser().resolve()
    if args.phase == 'evaluate' and any((settings.output_dir / name).exists() for name in ('test_metrics.json', 'test_predictions.jsonl')):
        raise FileExistsError('Electronic control evaluation already exists; preserve the run')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    function = train if args.phase == 'train' else test
    return function(settings, device, checkpoint_selection='test_development')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent/'configs/qwen_matched_electronic_control.yaml')
    parser.add_argument('--phase', choices=('train', 'evaluate'), required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--device')
    print(json.dumps(run(parser.parse_args()), indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
