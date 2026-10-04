"""Manifest-pinned serial G3/G4/G5 captures, then original G5 decoder tune.

Each changed optical PT gets new CCD. Normal inference uses r0, including G5.
No shared backend writes, no new layers, TEST selection is development only.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from . import train_preserved_upstream as architecture
from . import lab_modality_pipeline as pipeline
from .train import sha

GROUPS = ('r1_ccd', 'r2_ccd_dc30', 'r3_ccd_dc30_grid')
PREFIX = 'editor16_robust_chain_20261003'
LEGACY_PREFIX = PREFIX


def selected_rows(manifest, groups, prefix):
    """Require an explicit fresh namespace for any partial-group rerun."""
    assert prefix.isascii() and prefix.replace('_', '').isalnum(), 'Unsafe run prefix'
    assert tuple(groups) == tuple(g for g in GROUPS if g in groups), 'Duplicate or unordered groups'
    assert groups, 'No groups selected'
    if tuple(groups) != GROUPS:
        assert prefix != LEGACY_PREFIX, 'Partial reruns must not use sealed run prefix'
    rows = manifest['groups']
    assert [r['group'] for r in rows] == list(groups), 'Manifest must contain exactly selected groups'
    assert len({r['sha256'] for r in rows}) == len(rows), 'Duplicate weights'
    return rows


def factory(project, row):
    path = project / 'weights' / row['weight']
    assert sha(path) == row['sha256']
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['group'] == row['group']
    assert payload['settings']['editor_rank'] == 16
    assert payload['settings']['shared_readout_variant'] == 'lowrank64'
    assert payload['source_sha256'] == '01f7fc4a8de4f20901fae09e8be55c67fce2177db0a7ed3975a7fc90853eb05a'
    assert row['simulation'] == row['cpu_report']['overall']['changed_cell_accuracy']
    gates = payload['fixed_alpha_logits']
    assert len(gates) == 4

    def build(cfg, device):
        assert device.type == 'cpu', 'Hardware and cache use CPU before SDK'
        cfg.editor_rank = 16
        cfg.shared_readout_variant = 'lowrank64'
        model = architecture.build_model(cfg, device)
        model.load_state_dict(payload['model'], strict=True)
        assert sum(p.numel() for p in model.shared_readout.parameters()) == 179224
        assert sum(p.numel() for p in model.shared_readout.decoder.parameters()) == 30162
        named = dict(model.named_parameters())
        assert all(abs(float(named[n].detach()) - v) < 1e-7 for n, v in gates.items())
        # Capture/tune backends install clean r0 exactly once after this factory.
        return model
    return build


def configure(project, row):
    pipeline.WEIGHT = row['weight']
    pipeline.WEIGHT_SHA = row['sha256']
    pipeline.SIMULATION = row['simulation']
    pipeline.PREFIX = PREFIX + '_' + row['group']
    pipeline.GATES = row['fixed_alpha_logits']
    build = factory(project, row)
    pipeline.factory = lambda project: build
    return build


def capture(project, row, mode, resume):
    # Use its hash-pinned capture/audit and ordinary, unchanged resume contract.
    # Identity metadata is supplied here with the actual editor16 rank.
    configure(project, row)
    scope = 'train' if mode == 'train' else 'test'
    name = 'lab_shs_capture_g2_saturation' if scope == 'train' else 'lab_shs_capture'
    if scope == 'test': pipeline.accept_moderate_saturation(project)
    module = pipeline.backend(name)
    module.GROUPS = {'g2': (row['weight'], row['sha256'])}
    module.t = SimpleNamespace(build_model=factory(project, row))
    suffix = {'selftest': 'selftest', 'pilot': 'pilot4', 'test': 'test1000', 'train': 'train1000'}[mode]
    out = project / 'runs' / (pipeline.PREFIX + '_' + suffix)
    identity = dict(group=row['group'], checkpoint_sha256=row['sha256'],
                    editor_rank=16, decoder_rank=64, head_parameters=179224,
                    decoder_parameters=30162, scope=scope, mode=mode,
                    inference_profile='r0_clean_no_training_noise_DC_grid',
                    fixed_alpha_logits=row['fixed_alpha_logits'],
                    wrapper_sha256=sha(Path(__file__)),
                    backend_sha256=pipeline.BACKEND_SHA[name])
    identity_path = out / 'deployment_identity.json'
    if identity_path.exists(): assert json.loads(identity_path.read_text()) == identity
    else:
        assert not out.exists(), 'Preserve existing unidentified output'
        pipeline.write(identity_path, identity)
    done = out / ('selftest.json' if mode == 'selftest' else 'report.json')
    if not done.exists():
        assert resume or not (out / 'contract.json').exists(), 'Inspect incomplete capture before resume'
        sys.argv = [str(Path(module.__file__)), '--project', str(project), '--group', 'g2',
                    '--scope', scope, '--output', str(out), '--limit',
                    '4' if mode in ('selftest', 'pilot') else '1000',
                    '--exposure-us', '2000', '--device', 'cpu']
        if mode == 'selftest': sys.argv += ['--selftest']
        if resume: sys.argv += ['--resume']
        module.main()
    if mode == 'selftest':
        result = json.loads(done.read_text())
        assert result['status'] == 'pass' and result['error'] < 1e-4 and result['replay_error'] < 1e-4
    else: pipeline.audit_capture(out, 4 if mode == 'pilot' else 1000, scope)
    return out


def main():
    global PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--phase', choices=('selftest', 'pilot', 'all'), default='all')
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--groups', choices=GROUPS, nargs='+', default=list(GROUPS))
    parser.add_argument('--prefix', default=LEGACY_PREFIX)
    parser.add_argument('--capture-only', action='store_true',
                        help='Finish after direct TEST captures; never collect TRAIN or tune decoder')
    args = parser.parse_args()
    project = args.project.resolve()
    manifest = json.loads(args.manifest.read_text())
    rows = selected_rows(manifest, args.groups, args.prefix)
    PREFIX = args.prefix
    torch.set_num_threads(4)
    started = time.time()
    status_path = project / 'runs' / (PREFIX + '_pipeline_status.json')
    results = []
    def status(phase, state='running', **extra):
        pipeline.write(status_path, dict(status=state, phase=phase, results=results,
            manifest_sha256=sha(args.manifest), elapsed_seconds=time.time()-started, **extra))
        print(json.dumps(dict(phase=phase, status=state, **extra)), flush=True)
    try:
        # Validate all PT identities before touching the SDK.
        for row in rows: factory(project, row)
        for row in rows:
            status(row['group'] + '/selftest')
            capture(project, row, 'selftest', args.resume)
            if args.phase == 'selftest': continue
            status(row['group'] + '/pilot')
            capture(project, row, 'pilot', args.resume)
            if args.phase == 'pilot': continue
            status(row['group'] + '/test')
            out = capture(project, row, 'test', args.resume)
            report = json.loads((out / 'report.json').read_text())
            results.append(dict(group=row['group'], simulation=row['simulation'],
                direct=report['physical_metrics']['overall']['changed_cell_accuracy'],
                weight_sha256=row['sha256'], report_path=str(out/'report.json')))
            pipeline.write(project/'runs'/(PREFIX+'_comparison.json'), results)
        if args.phase != 'all':
            status(args.phase, 'complete'); return
        if args.capture_only:
            status('complete', 'complete', adaptation='explicitly disabled', development_only=True)
            return
        row = rows[-1]; configure(project, row)
        target = row['simulation'] * .975
        if results[-1]['direct'] >= target:
            status('complete', 'complete', target=target, adaptation='not required', development_only=True)
            return
        status('G5/train')
        train_run = capture(project, row, 'train', args.resume)
        test_run = project/'runs'/(pipeline.PREFIX+'_test1000')
        tune = pipeline.backend('lab_tune_g2_test')
        tune.GROUPS = {'g2': (row['weight'], row['sha256'])}
        tune.t = SimpleNamespace(build_model=factory(project, row))
        output = project/'runs'/(pipeline.PREFIX+'_decoder_train1000_testselected')
        assert not output.exists(), 'Preserve existing tune output; inspect failures before recovery'
        assert torch.cuda.is_available()
        free, _ = torch.cuda.mem_get_info()
        assert free > 2*1024**3, 'Wait for decoder GPU; do not affect others'
        status('G5/tune', target=target, decoder_parameters=30162)
        sys.argv = [str(Path(tune.__file__)), '--project', str(project), '--train-run', str(train_run),
                    '--test-run', str(test_run), '--output', str(output), '--epochs', str(args.epochs), '--device', 'cuda']
        tune.main()
        pipeline.strict_tune(tune, project, output)
        strict_path = output/'strict_reload.json'
        strict = json.loads(strict_path.read_text())
        strict.pop('target_relative_97percent', None)
        strict['target_relative_97_5percent'] = target
        pipeline.write(strict_path, strict)
        adapted = strict['metrics']['overall']['changed_cell_accuracy']
        results[-1].update(adapted=adapted, target=target, target_met=adapted >= target,
                           adapted_sha256=strict['best_sha256'])
        pipeline.write(project/'runs'/(PREFIX+'_comparison.json'), results)
        status('complete', 'complete', development_only=True)
    except Exception as exc:
        status('failed', 'failed', error=repr(exc)); raise


if __name__ == '__main__': main()
