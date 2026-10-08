"""Read-only-weight LSP replay and amplitude-range audit before optical capture.

Uses the existing evaluator and checkpoint architecture. No device SDK imports,
no weight changes, no rescaling or clipping of the simulated optical field.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

EXPECTED_SHA = '495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-mode', choices=('original', 'continuous', 'uint8'), default='original')
    parser.add_argument('--amplitude-scale', type=float, default=16.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing preflight output')
    from . import run, training, modeling
    from .build_lab_package import sha
    import torch
    if sha(args.checkpoint) != EXPECTED_SHA:
        raise ValueError('Wrong historical LSP checkpoint')
    stages = {s: {'minimum': None, 'maximum': 0., 'above_one': 0,
                  'pixels': 0, 'forward_samples': 0, 'shapes': []}
              for s in ('router', 'expert', 'global')}
    build = modeling.build_student

    def observe(model, _inputs, _output):
        path = model.core.optical_branch
        fields = (path.core.router.last_input_amplitude,
                  path.last_expert_input_amplitude, path.last_global_input_amplitude)
        for name, field in zip(stages, fields):
            if field is None or not torch.isfinite(field).all() or field.min() < 0:
                raise ValueError('Missing/nonfinite/negative optical amplitude')
            stat = stages[name]
            value = field.detach().float()
            minimum, maximum = float(value.min()), float(value.max())
            stat['minimum'] = minimum if stat['minimum'] is None else min(stat['minimum'], minimum)
            stat['maximum'] = max(stat['maximum'], maximum)
            stat['above_one'] += int((value > 1).sum())
            stat['pixels'] += value.numel()
            stat['forward_samples'] += len(value)
            shape = list(value.shape[-2:])
            if shape not in stat['shapes']:
                stat['shapes'].append(shape)
            if args.physical_mode != 'original':
                physical = model._lab_unit_amplitudes[name]
                stat['physical_maximum'] = max(stat.get('physical_maximum', 0.), float(physical.max()))

    def factory(loaded, settings):
        model = build(loaded, settings)
        model.register_forward_hook(observe)
        if args.physical_mode != 'original':
            from .lab_field_units import FieldUnits
            original_forward = model.forward
            def physical_forward(*values, **keywords):
                with FieldUnits(model, scale=args.amplitude_scale,
                                quantize=args.physical_mode == 'uint8') as tap:
                    result = original_forward(*values, **keywords)
                    if tap.index != 3:
                        raise RuntimeError('Require exactly three optical propagations')
                    model._lab_unit_amplitudes = tap.amplitudes
                    return result
            model.forward = physical_forward
        return model

    # Parameter-free instrumentation only; the original evaluator builds and
    # strict-loads the original core/head, uses original crop/TTA/PCK definitions.
    training.build_student = factory
    run.build_student = factory
    result = run.run(SimpleNamespace(profile='main_dc20_no_shift_warmstart',
                        run_dir=str(args.output), seed=42, phase='evaluate',
                        checkpoint=str(args.checkpoint)))
    if result['test_samples'] != 1000 or result['checkpoint_sha256'] != EXPECTED_SHA:
        raise ValueError('Wrong dataset or checkpoint in replay')
    report = {'status': 'complete', 'checkpoint_sha256': EXPECTED_SHA,
              'test_samples': 1000, 'metrics': result['metrics'], 'stages': stages,
              'hardware_started': False, 'weights_unchanged': sha(args.checkpoint) == EXPECTED_SHA,
              'physical_mode': args.physical_mode, 'amplitude_scale': args.amplitude_scale,
              'physical_intensity_to_model_units': args.amplitude_scale ** 2,
              'unit_interval_export_safe': all(s.get('physical_maximum', s['maximum']) <= 1 for s in stages.values()),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    (args.output / 'amplitude_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
