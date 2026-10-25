"""T20 four-expert capture after explicit completed-T21 handoff.

Uses immutable native camera/display utilities, never T21's model loader. All
records of a layer precede the next layer; only the original terminal Linear
may be fitted after full acquisition. No retry, exposure scan or double frame.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch

from .build_lab_package import CONTRACT, PINS, load_locked, put, sha
from LightGenV2.tasks.t21_continual_learning_hardware.hardware_bridge import (
    hardware, hardware_identity, native_image, crop_native, signal, guard,
    stage_signal_admission)
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import tensor_sha


@contextmanager
def sdk_owner(lock, out):
    """Atomic exclusive claim; never remove a foreign owner's lock."""
    with Path(lock).open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), task='T20', out=str(out)), f)
    try:
        yield
    finally:
        if json.loads(Path(lock).read_text())['pid'] != os.getpid():
            raise RuntimeError('SDK lock ownership changed; foreign lock preserved')
        Path(lock).unlink()


def validate_handoff(handoff):
    """An empty lock or stopped process alone is not completed acquisition."""
    if (handoff.get('T21_capture_complete') is not True or handoff.get('SDK_released') is not True or
        handoff.get('next_task') != 'T20' or not handoff.get('report_sha256')):
        raise ValueError('Explicit completed T21 capture handoff required; not an idle failure boundary')
    p = Path(handoff['report'])
    if sha(p) != handoff['report_sha256']:
        raise ValueError('T21 handoff report changed')
    r = json.loads(p.read_text())
    if (r.get('passed') is not True or r.get('pilot') is not False or r.get('SDK_released') is not True or
        r.get('full_physical_prefix') is not True or r.get('checkpoint_sha256') != handoff.get('checkpoint_sha256') or
        r.get('stages') != ['router', 'first', 'final'] or r.get('records') != 5900 or r.get('frames') != 17700):
        raise ValueError('T21 LG D has not supplied the complete expected physical capture')


def read_entries(folder, manifest, case, pilot):
    rows = []
    spec = manifest['cases'][case]
    for split in ('train', 'test'):
        if pilot and split != 'train':
            continue
        info = manifest['datasets'][split]
        path = folder/info['file']
        expected = next(r['sha256'] for r in manifest['files'] if r['name'] == info['file'])
        if sha(path) != expected:
            raise ValueError('Encoded source data SHA mismatch')
        indices = spec['TRAIN_indices' if split == 'train' else 'TEST_indices']
        if pilot:
            indices = np.asarray(indices)[np.linspace(0, len(indices)-1, 24, dtype=int)].tolist()
        with np.load(path, allow_pickle=False) as d:
            for i in indices:
                rows.append(dict(split=split, local_index=int(i), source_index=int(d['source_indices'][i]),
                    source_id=str(d['ids'][i]), domain=int(d['domains'][i]), label=int(d['labels'][i]),
                    amplitude=d['amplitudes'][i].copy(), source_identity_sha256=info['identity_sha256']))
    if len(rows) != (24 if pilot else spec['TRAIN_records']+2000):
        raise ValueError('Incomplete fixed acquisition identities')
    return rows


def router_mask(model):
    mask = np.zeros((model.active, model.active), bool)
    for index in model.active_indices[:int(model.active_count)].tolist():
        y, x = model.router_centers[index]
        y -= model.border; x -= model.border; h = model.router_side//2
        mask[y-h:y+h, x-h:x+h] = True
    return mask


def verify_admission(path, cfg):
    r = json.loads(Path(path).read_text())
    if r.get('passed') is not True or r.get('hardware_payload_sha256') != hardware_identity(cfg):
        raise ValueError('Spatial admission belongs to another physical hardware session')
    return sha(path)


@torch.no_grad()
def capture(folder, case, cfg, out, handoff_path, spatial_path, source_commit, pilot=False, pilot_report=None):
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['contract'] != CONTRACT or not manifest['passed'] or manifest.get('hardware_admitted'):
        raise ValueError('Expected locked preparation package, not an altered physical result')
    validate_handoff(json.loads(handoff_path.read_text()))
    if abs(cfg['exposure_us']-2000) > .5 or cfg['wait_ms'] != 240 or cfg['sensor_full_scale'] != 65535:
        raise ValueError('Fixed single-frame 2ms / wait240 / uint16 contract required')
    spatial_sha = verify_admission(spatial_path, cfg)
    spec = manifest['cases'][case]
    model, _ = load_locked(case, folder/spec['checkpoint'])
    expected = PINS[case]['sha256']
    if spec['checkpoint_sha256'] != expected:
        raise ValueError('Package checkpoint differs from fixed case')
    if not pilot:
        if pilot_report is None:
            raise ValueError('Same-model TRAIN24 pilot must precede full capture')
        pr = json.loads(pilot_report.read_text())
        if not (pr['passed'] and pr['pilot'] and pr['checkpoint_sha256'] == expected and
                pr['hardware_payload_sha256'] == hardware_identity(cfg) and pr['records'] == 24):
            raise ValueError('Pilot identity/pass mismatch')
    entries = read_entries(folder, manifest, case, pilot)
    order = tuple(spec['stages'])
    if order != (model.STAGES if case == 'LG' else model.STAGES[1:]):
        raise ValueError('Changed physical stage order')
    # Native arrays alone cost ~12.5MB/network-frame, plus BMP and diagnostics.
    need = len(entries)*len(order)*15_000_000+2_000_000_000
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < need:
        raise RuntimeError('Insufficient storage; no deletion or SDK acquisition performed')
    Bench, lock, _, drive, warp = hardware(cfg)
    out.mkdir(exist_ok=False)
    frozen = {k: tensor_sha(v) for k, v in model.state_dict().items()}
    sources = [Path(__file__), Path(__file__).with_name('build_lab_package.py')]
    import LightGenV2.tasks.t21_continual_learning_hardware.hardware_bridge as backend
    sources.append(Path(backend.__file__))
    source_sha = {str(p): sha(p) for p in sources}
    root = Path(__file__).resolve().parents[3]
    relative = Path(__file__).resolve().relative_to(root).as_posix()
    published = subprocess.check_output(['git','show',source_commit+':'+relative], cwd=root)
    if hashlib.sha256(published).hexdigest() != sha(__file__):
        raise ValueError('Capture entry is not the specified Git-published source')
    put(out/'protocol.json', dict(case=case, pilot=pilot, package_sha256=sha(manifest_path),
        checkpoint_sha256=expected, hardware=cfg, hardware_payload_sha256=hardware_identity(cfg),
        spatial_admission_sha256=spatial_sha, handoff_sha256=sha(handoff_path),
        source_sha256=source_sha, source_publication_commit=source_commit,
        runtime_repository_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        TRAIN_records=sum(e['split']=='train' for e in entries),
        TEST_records=sum(e['split']=='test' for e in entries), frozen_state_sha256=frozen,
        interlayer_tuning=False, network_background_subtraction=False, single_frame=True,
        SNR_minimum=10, repeat_correlation_minimum=.95, repeat_consumer_drift_maximum=.05,
        signal_guard='pointwise fixed TRAIN/control dark; SNR10, minimum64 counts and dynamic-range guard',
        saturation_maximum=.15, TEST_gradients=False, automatic_retry=False))
    attempted = valid = 0
    began = time.monotonic()
    try:
        with sdk_owner(lock, out):
            with Bench(cfg, out) as bench:
                for stage in order:
                    layer = out/stage; layer.mkdir()
                    phase_file = layer/'phase.bmp'
                    phase_digest = None
                    mask = router_mask(model) if stage == 'router' else None
                    def payload(i):
                        prefix = {s: torch.from_numpy(np.load(out/s/f'{i:05d}'/'ccd.npy'))[None]
                            for s in order[:order.index(stage)]}
                        amp, phase, _ = model.payload(torch.from_numpy(entries[i]['amplitude'])[None], stage, prefix)
                        gray = np.rint(amp[0].numpy()*255).clip(0,255).astype(np.uint8)
                        phase_gray = np.rint(phase.numpy()*256).clip(0,255).astype(np.uint8)
                        return gray, native_image(phase_gray, True), prefix
                    train = [i for i, e in enumerate(entries) if e['split'] == 'train']
                    controls = []
                    for i in (train[0], train[-1]):
                        gray, phase, _ = payload(i)
                        digest = hashlib.sha256(phase.tobytes()).hexdigest()
                        if phase_digest is None:
                            Image.fromarray(phase).save(phase_file)
                            bench.stage(phase_file); phase_digest = digest
                        elif digest != phase_digest:
                            raise ValueError('Input-dependent phase violates frozen-mask capture')
                        controls.append(gray)
                    noise = stage_signal_admission(bench, layer/'TRAIN_signal_admission', controls,
                        drive, warp, mask, stage, hardware_identity(cfg), model)
                    dark = np.load(layer/'TRAIN_signal_admission/dark_ccd.npy')
                    if sha(layer/'TRAIN_signal_admission/dark_ccd.npy') != noise['dark_CCD_sha256']:
                        raise ValueError('Same-stage control dark-frame identity changed')
                    for i, entry in enumerate(entries):
                        tick = time.monotonic()
                        target = layer/f'{i:05d}'; target.mkdir()
                        gray, phase, prefix = payload(i)
                        if hashlib.sha256(phase.tobytes()).hexdigest() != phase_digest:
                            raise ValueError('Frozen phase changed between records')
                        Image.fromarray(native_image(drive[gray])).save(target/'input.bmp')
                        raw, metadata = bench.capture(target/'input.bmp')
                        np.save(target/'raw.npy', raw)
                        value = crop_native(raw, warp); np.save(target/'ccd.npy', value)
                        # Diagnostic delta is pointwise, not a difference of
                        # spatial quantiles. Network value is never subtracted.
                        quality = signal(value, mask, dark)
                        attempted += 1
                        row = {k: v for k, v in entry.items() if k != 'amplitude'}
                        row.update(stage=stage, index=i, checkpoint_sha256=expected,
                            hardware_payload_sha256=hardware_identity(cfg), input_canonical_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
                            input_BMP_sha256=sha(target/'input.bmp'), phase_BMP_sha256=sha(phase_file),
                            raw_sha256=sha(target/'raw.npy'), ccd_sha256=sha(target/'ccd.npy'),
                            prefix_CCD_sha256={s: sha(out/s/f'{i:05d}'/'ccd.npy') for s in prefix},
                            signal=quality, metadata=metadata, passed=False,
                            workflow_ms=(time.monotonic()-tick)*1000)
                        put(target/'receipt.json', row)
                        guard(quality, noise, router=stage=='router', policy='pointwise_dark_p99')
                        valid += 1; row['passed'] = True; put(target/'receipt.json', row)
                        put(out/'progress.json', dict(stage=stage, stage_frames=i+1, stage_total=len(entries),
                            valid_network_frames=valid, attempted_network_frames=attempted,
                            expected_network_frames=len(entries)*len(order), latest=row, updated=time.time()))
                features = [model.features_from_ccd(torch.from_numpy(np.load(out/'final'/f'{i:05d}'/'ccd.npy'))[None])[0].numpy()
                    for i in range(len(entries))]
                np.savez_compressed(out/'features.npz', features=np.stack(features),
                    labels=[e['label'] for e in entries], domains=[e['domain'] for e in entries],
                    splits=[e['split'] for e in entries], ids=[e['source_id'] for e in entries],
                    source_indices=[e['source_index'] for e in entries])
        if frozen != {k: tensor_sha(v) for k, v in model.state_dict().items()} or any(sha(Path(p)) != v for p, v in source_sha.items()):
            raise RuntimeError('Captured model/source identity changed')
        put(out/'report.json', dict(passed=True, pilot=pilot, case=case, records=len(entries),
            frames=valid, attempted_frames=attempted, stages=list(order),
            TRAIN_records=sum(e['split']=='train' for e in entries), TEST_records=sum(e['split']=='test' for e in entries),
            checkpoint_sha256=expected, features_sha256=sha(out/'features.npz'),
            hardware_payload_sha256=hardware_identity(cfg), SDK_released=True,
            full_physical_prefix=True, interlayer_tuning=False, seconds=time.monotonic()-began,
            extra_TRAIN_control_frames=7*len(order), mean_seconds_per_network_frame=(time.monotonic()-began)/valid))
    except BaseException:
        put(out/'failure.json', dict(traceback=traceback.format_exc(), valid_network_frames=valid,
            attempted_network_frames=attempted, failed_evidence_frames=attempted-valid,
            all_evidence_preserved=True, automatic_retry=False))
        raise


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('package', 'hardware', 'handoff', 'spatial-admission', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--case', choices=tuple(PINS), required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--pilot-report', type=Path)
    a = p.parse_args(); torch.set_num_threads(2)
    capture(a.package, a.case, json.loads(a.hardware.read_text()), a.out, a.handoff,
        a.spatial_admission, a.source_commit, pilot=a.pilot, pilot_report=a.pilot_report)


if __name__ == '__main__':
    main()
