"""Sequential, fail-closed deployment of the frozen language55/vision80 PT.

Only task-local process references are substituted; previous capture/tune code,
optical contract and other checkpoints are untouched. No new adaptation layers.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from . import split_rank_head
from .train import sha
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import STAGES

WEIGHT = 'modality_language55_vision80_e40.pt'
WEIGHT_SHA = 'd94dd0450edd415eee613ee28e424242239bf3dd1dd559cae57fe27d634b8c86'
SIMULATION = .8935
PREFIX = 'modality_l55_v80'
GATES = {'language_core.block1': .55, 'language_core.block2': .55,
         'vision_core.block1': .8, 'vision_core.block2': .8}
BACKEND_SHA = {
    'lab_shs_capture': '605d17590c5947fedec030fd913c19a4ee31be08b1ae832c99377ebb015fe5c6',
    'lab_shs_capture_g2_saturation': '8d08a13b0cfca0bec28b5a531535b45420caf257aea91f46d49ef162ea92b89b',
    'lab_tune_g2_test': 'd4c0c4fd59096bd627cc621ce7019a0f989841885b2c198659efd622812c9a0f',
}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def backend(name):
    assert sha(Path(__file__).with_name(name + '.py')) == BACKEND_SHA[name], name
    return importlib.import_module('.' + name, __package__)


def factory(project):
    path = project / 'weights' / WEIGHT
    assert sha(path) == WEIGHT_SHA
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['group'] == 'r0_base'
    assert payload['settings']['editor_rank'] == 48
    assert payload['settings']['shared_readout_variant'] == 'lowrank64'
    assert set(payload['fixed_fusion_alphas']) == set(GATES)
    assert all(abs(payload['fixed_fusion_alphas'][k] - GATES[k]) < 1e-6 for k in GATES)

    def build(cfg, device):
        assert device.type == 'cpu', 'All hardware and feature-cache models stay CPU'
        cfg.shared_readout_variant = 'lowrank64'
        cfg.editor_rank = 48
        cfg.optical_fusion_initial = .8
        model = split_rank_head.build_model(cfg, device)
        model.load_state_dict(payload['model'], strict=True)
        assert sum(p.numel() for p in model.shared_readout.parameters()) == 240664
        assert sum(p.numel() for p in model.shared_readout.decoder.parameters()) == 30162
        actual = {f'{name}.block{block}': float(getattr(getattr(model, name),
                  f'block{block}_optical_fusion').detach())
                  for name in ('language_core', 'vision_core') for block in (1, 2)}
        assert all(abs(actual[k] - GATES[k]) < 1e-6 for k in GATES)
        return model
    return build


def capture(project, mode, resume):
    scope = 'train' if mode == 'train' else 'test'
    name = 'lab_shs_capture_g2_saturation' if scope == 'train' else 'lab_shs_capture'
    module = backend(name)
    module.GROUPS = {'g2': (WEIGHT, WEIGHT_SHA)}
    module.t = SimpleNamespace(build_model=factory(project))
    suffix = {'selftest': 'selftest', 'pilot': 'pilot4', 'test': 'test1000', 'train': 'train1000'}[mode]
    output = project / 'runs' / (PREFIX + '_' + suffix)
    identity = dict(weight_sha256=WEIGHT_SHA, gates=GATES, editor_rank=48,
                    decoder_parameters=30162, scope=scope, mode=mode,
                    backend_sha256=BACKEND_SHA[name], wrapper_sha256=sha(Path(__file__)),
                    factory_sha256=sha(Path(split_rank_head.__file__)))
    identity_path = output / 'deployment_identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    else:
        assert not output.exists(), 'Existing unidentified output must be preserved'
        write(identity_path, identity)
    done = output / ('selftest.json' if mode == 'selftest' else 'report.json')
    if done.exists():
        if mode == 'selftest':
            result = json.loads(done.read_text())
            assert result['status'] == 'pass' and result['error'] < 1e-4 and result['replay_error'] < 1e-4
        else:
            audit_capture(output, 4 if mode == 'pilot' else 1000, scope)
        return output
    # The ordinary backend contract remains unchanged, enabling safe same-identity resume.
    if (output / 'contract.json').exists() and not resume:
        raise ValueError('Incomplete output exists: inspect failure before explicit --resume')
    sys.argv = [str(Path(module.__file__)), '--project', str(project), '--group', 'g2',
                '--scope', scope, '--output', str(output), '--limit',
                '4' if mode in ('selftest', 'pilot') else '1000',
                '--exposure-us', '2000', '--device', 'cpu']
    if mode == 'selftest':
        sys.argv += ['--selftest']
    if resume:
        sys.argv += ['--resume']
    module.main()
    if mode != 'selftest':
        audit_capture(output, 4 if mode == 'pilot' else 1000, scope)
    else:
        result = json.loads(done.read_text())
        assert result['status'] == 'pass' and result['error'] < 1e-4 and result['replay_error'] < 1e-4
    return output


def audit_capture(output, count, scope):
    report = json.loads((output / 'report.json').read_text())
    contract = report['contract']
    assert report['status'] == 'complete' and report['ccd_count'] == 6 * count
    assert contract['checkpoint_sha256'] == WEIGHT_SHA
    assert contract['scope'] == scope and contract['count'] == count
    assert contract['model_device'] == 'cpu'
    assert contract['exposure_us'] == 2000 and contract['gain'] == 'Gain_X4'
    assert contract['wait_ms'] == 240 and contract['camera_orientation'] == 'flip_v'
    signal = {}
    for stage in STAGES:
        folder = output / 'ccd' / stage
        assert len(list(folder.glob('*.png'))) == count
        assert len(list(folder.glob('*.json'))) == count
        phase_sha = sha(output / 'phase' / (stage + '.bmp'))
        lows, saturations = [], []
        for receipt in folder.glob('*.json'):
            row = json.loads(receipt.read_text())
            assert row['stage'] == stage and row['sample_id'] == receipt.stem
            assert row['phase_sha256'] == phase_sha
            assert row['exposure']['exposure_us'] == 2000 and row['exposure']['gain'] == 'Gain_X4'
            assert row['wait_ms'] == 240 and row['canonical_orientation'] == 'flip_v'
            assert row['p99'] >= 15
            image = np.asarray(Image.open(receipt.with_suffix('.png')))
            assert image.dtype == np.uint8 and image.ndim == 2
            assert np.percentile(image, 99) >= 15
            lows.append(row['p99'])
            saturations.append(row['saturation_fraction'])
        signal[stage] = dict(count=count, min_p99=min(lows), max_saturation=max(saturations), phase_sha256=phase_sha)
    write(output / 'capture_audit.json', dict(status='pass', weight_sha256=WEIGHT_SHA,
                                            ccd_count=6 * count, layers=signal))


def strict_tune(tune, project, output):
    execution = json.loads((output / 'execution.json').read_text())
    report = json.loads((output / 'report.json').read_text())
    assert execution['checkpoint_sha256'] == WEIGHT_SHA
    assert execution['trainable_prefix'] == 'shared_readout.decoder'
    assert execution['trainable_parameters'] == 30162
    assert execution['fit_count'] == 1000 and execution['validation_count'] == 0
    assert report['protected_unchanged'] and not report['test_gradient']
    _, model = tune.config(project, torch.device('cpu'))
    before = tune.protected_sha(model)
    assert before == execution['protected_before']
    for name in ('best.pt', 'last.pt'):
        saved = torch.load(output / name, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        assert tune.protected_sha(model) == before
    saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    cached = torch.load(output / 'test_features.pt', map_location='cpu', weights_only=False)
    assert len(cached['ids']) == 1000
    meter = MetricAccumulator()
    with torch.no_grad():
        for start in range(0, 1000, 32):
            rows = cached['rows'][start:start + 32]
            batch = {k: torch.cat([r[k] for r in rows]) if torch.is_tensor(rows[0][k])
                     else sum([r[k] for r in rows], []) for k in rows[0]}
            category, edit = model.shared_readout.decoder(cached['features'][start:start + 32])
            meter.update(dict(category_logits=category, edit_logits=edit,
                              task_logits=batch['task_logits']), batch)
    actual = meter.compute()
    assert abs(actual['overall']['changed_cell_accuracy'] - report['physical_test']['overall']['changed_cell_accuracy']) < 1e-8
    write(output / 'strict_reload.json', dict(status='pass', device='cpu', metrics=actual,
          best_sha256=sha(output / 'best.pt'), last_sha256=sha(output / 'last.pt'),
          protected_before=before, protected_after=tune.protected_sha(model),
          original_simulation=SIMULATION, target_relative_97percent=SIMULATION * .97,
          test_selection_development=True, test_gradient=False, architecture_unchanged=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--phase', choices=('selftest', 'pilot', 'all'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epochs', type=int, default=160)
    args = parser.parse_args()
    project = args.project.resolve()
    status_path = project / 'runs' / (PREFIX + '_pipeline_status.json')
    started = time.time()
    torch.set_num_threads(4)

    def status(phase, state='running', **extra):
        write(status_path, dict(status=state, phase=phase, checkpoint_sha256=WEIGHT_SHA,
                                elapsed_seconds=time.time() - started, **extra))
        print(json.dumps(dict(phase=phase, status=state, **extra)), flush=True)

    try:
        status('selftest')
        capture(project, 'selftest', args.resume)
        if args.phase == 'selftest':
            status('selftest', 'complete')
            return
        status('pilot')
        capture(project, 'pilot', args.resume)
        if args.phase == 'pilot':
            status('pilot', 'complete')
            return
        status('test')
        test_run = capture(project, 'test', args.resume)
        direct = json.loads((test_run / 'report.json').read_text())['physical_metrics']['overall']['changed_cell_accuracy']
        if direct >= SIMULATION * .97:
            status('complete', 'complete', direct=direct, adaptation='not required for recovery',
                   direct_gap_pp=100 * (SIMULATION - direct))
            return
        status('train', direct=direct)
        train_run = capture(project, 'train', args.resume)
        tune = backend('lab_tune_g2_test')
        tune.GROUPS = {'g2': (WEIGHT, WEIGHT_SHA)}
        tune.t = SimpleNamespace(build_model=factory(project))
        output = project / 'runs' / (PREFIX + '_decoder_train1000_testselected')
        assert not output.exists(), 'Existing adapter output must be preserved; inspect before recovery'
        assert torch.cuda.is_available(), 'Bench decoder GPU unavailable; preserve complete CCD'
        free, total = torch.cuda.mem_get_info()
        assert free > 2 * 1024**3, 'GPU busy; preserve complete CCD and wait safely'
        status('tune', direct=direct, trainable_parameters=30162)
        sys.argv = [str(Path(tune.__file__)), '--project', str(project),
                    '--train-run', str(train_run), '--test-run', str(test_run),
                    '--output', str(output), '--epochs', str(args.epochs), '--device', 'cuda']
        # Backend checks source-disjoint TRAIN, all six phases/receipts and the
        # exact full TEST CPU baseline before any decoder gradient.
        tune.main()
        strict_tune(tune, project, output)
        result = json.loads((output / 'strict_reload.json').read_text())
        adapted = result['metrics']['overall']['changed_cell_accuracy']
        gap_pp = 100 * (SIMULATION - direct)
        recovery_met = adapted >= SIMULATION * .97
        status('complete', 'complete', simulation=SIMULATION, direct=direct, adapted=adapted,
               recovery_target_met=recovery_met,
               all_three_targets_met=.88 <= SIMULATION < .9 and 30 <= gap_pp <= 40 and recovery_met,
               direct_gap_pp=100 * (SIMULATION - direct), development_only=True)
    except Exception as exc:
        status('failed', 'failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
