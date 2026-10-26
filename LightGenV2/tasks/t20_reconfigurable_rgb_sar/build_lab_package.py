"""Export the locked static-four experiment; never opens a camera or selects PTs.

Only weights, encoded data and provenance are packed. Python sources travel by
Git, not in this archive. TEST labels are retained for subsequent explicit
development evaluation, never used to select inputs or to compute encodings.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
CONTRACT = 'T20_static4_fullaperture_native8_v5'
FRONTEND_SHA = 'b3a5c165cceeb4fb3329d951ddecbe7a03d4804c1dcecb2f081081ff461cf083'
PINS = {
    'LG': dict(architecture='moe',
        checkpoint='static4_routing_diversity_20261026/equivalent_phase_export_20261026/model.pt',
        sha256='0e0587e11bad939bedc5171cee02164c768862d4a39d5e51c20b7fee616a2592',
        CPU_report='static4_routing_diversity_20261026/equivalent_phase_export_20261026/CPU_report.json',
        correct={'RGB': 839, 'SAR': 700}, max_drop_correct=20, training_domains=[0, 1]),
    'D2NN': dict(architecture='d2nn',
        checkpoint='static4_aligned_native8_20261025/D2NN_RGB/best_checkpoint.pt',
        sha256='14b36e6f8a98189db7fe02e51ed7086452adf2ad4f9b214b21b174c3795edfa2',
        CPU_report='static4_aligned_native8_20261025/D2NN_RGB_CPU/best.json',
        correct={'RGB': 828, 'SAR': 621}, max_drop_correct=80, training_domains=[0]),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def put(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def selected_training(data, per_class=50):
    """Fixed per-domain/class TRAIN sample; independent of PT performance."""
    if per_class != 50:
        raise ValueError('Published acquisition budget is 50 TRAIN per domain/class')
    y, domains, ids = (np.asarray(data[k]) for k in ('labels', 'domains', 'ids'))
    if len(y) != len(domains) or len(y) != len(ids) or len(set(map(str, ids))) != len(y):
        raise ValueError('Ambiguous TRAIN identities')
    rows = []
    for domain in (0, 1):
        for label in range(10):
            pool = np.flatnonzero((domains == domain) & (y == label))
            if len(pool) < per_class:
                raise ValueError('Insufficient original TRAIN; no TEST/VAL replacement')
            ordered = sorted(pool, key=lambda i: hashlib.sha256(str(ids[i]).encode()).digest())
            rows.extend(ordered[:per_class])
    return np.asarray(rows, dtype=np.int64)


def datasets_for(case, train_domains):
    if case not in PINS:
        raise ValueError('Only the two locked experimental cases are permitted')
    domains = np.asarray(train_domains)
    if len(domains) != 1000 or any(int((domains == d).sum()) != 500 for d in (0, 1)):
        raise ValueError('Changed shared TRAIN budget')
    return np.flatnonzero(np.isin(domains, PINS[case]['training_domains'])).tolist()


def export_pilot_subsets(folder):
    """Lossless TRAIN24 transport while the complete input package is in flight.

    This is exactly the existing pilot selection, not a new subset or encoding.
    The small derivative is bound to the original full-file/package hashes.
    """
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    expected=next(r['sha256'] for r in manifest['files'] if r['name']=='train_inputs.npz')
    path=folder/'train_inputs.npz'
    if sha(path)!=expected:raise ValueError('Original full TRAIN encoding SHA mismatch')
    proof=dict(parent_manifest_sha256=sha(folder/'manifest.json'),full_TRAIN_file_sha256=expected,
        rule='Exact original np.linspace TRAIN24 pilot selection; no changed encoding',cases={})
    with np.load(path,allow_pickle=False) as source:
        for case,spec in manifest['cases'].items():
            pool=np.asarray(spec['TRAIN_indices'],dtype=np.int64)
            ix=pool[np.linspace(0,len(pool)-1,24,dtype=int)]
            target=folder/(case+'_pilot_inputs.npz')
            if target.exists():raise FileExistsError(target)
            np.savez_compressed(target,**{k:source[k][ix] for k in ('amplitudes','labels','domains','ids','source_indices')},local_indices=ix)
            proof['cases'][case]=dict(file=target.name,sha256=sha(target),records=24,
                checkpoint_sha256=spec['checkpoint_sha256'],local_indices=ix.tolist(),
                ids=source['ids'][ix].tolist(),source_indices=source['source_indices'][ix].tolist(),
                amplitude_array_sha256=hashlib.sha256(source['amplitudes'][ix].tobytes()).hexdigest())
    put(folder/'pilot_inputs.json',proof)
    return proof


def load_locked(case, path):
    """T20 factory, never the incompatible T21 sixteen-slot checkpoint loader."""
    import torch
    from .static_four import build_static_four, geometry
    pin = PINS[case]
    if sha(path) != pin['sha256']:
        raise ValueError('Locked checkpoint SHA mismatch')
    cp = torch.load(path, map_location='cpu', weights_only=False)
    if cp['contract'] != CONTRACT or cp['architecture'] != pin['architecture'] or cp['frontend_sha256'] != FRONTEND_SHA:
        raise ValueError('Changed graph or frozen frontend')
    model = build_static_four(cp['architecture'])
    model.load_state_dict(cp['model'], strict=True)
    model.eval().requires_grad_(False)
    g = geometry(model)
    if g['active_pixels'] != 1016 or g['canvas_pixels'] != 1102 or g['expert_grid'] != 2 or g['readout_parameters'] != 7840:
        raise ValueError('Incompatible physical layout or readout')
    return model, cp


def reference(case, simulation):
    pin = PINS[case]
    raw = json.loads((simulation / pin['CPU_report']).read_text())
    result = raw['full_result'] if 'full_result' in raw else raw
    if 'checkpoint_SHA256' in raw and raw['checkpoint_SHA256'] != pin['sha256']:
        raise ValueError('CPU report belongs to another checkpoint')
    for domain, correct in pin['correct'].items():
        if result['scores'][domain]['n'] != 1000 or result['scores'][domain]['correct'] != correct:
            raise ValueError('Full CPU reference differs from the locked simulation')
    return result


def bridge_audit(model, inputs):
    """TRAIN-only serial boundary replay including the physical 8bit codes."""
    import torch
    rows = []
    order = model.STAGES if model.architecture == 'moe' else model.STAGES[1:]
    with torch.no_grad():
        for i in np.linspace(0, len(inputs)-1, 24, dtype=int):
            x = torch.from_numpy(inputs[i:i+1])
            direct = model(x, return_debug=True)
            prefix, codes = {}, {}
            for stage in order:
                amplitude, phase, _ = model.payload(x, stage, prefix)
                gray = torch.round(amplitude * 255).clamp(0, 255).to(torch.uint8)
                phase_gray = torch.round(phase * 256).clamp(0, 255).to(torch.uint8)
                prefix[stage] = model.propagate_payload(gray.float()/255, phase_gray.float()/256)
                codes[stage] = dict(input_sha256=hashlib.sha256(gray.numpy().tobytes()).hexdigest(),
                    phase_sha256=hashlib.sha256(phase_gray.numpy().tobytes()).hexdigest(),
                    code_max=int(gray.max()), zero_code_preserved=True)
            features = model.features_from_ccd(prefix['final'])
            logits = model.shared_head(features)
            replay = model(x, measured=prefix)
            torch.testing.assert_close(logits, replay['logits'], atol=0, rtol=0)
            error = float((logits.softmax(1)-direct['probabilities']).abs().max())
            same = bool(torch.equal(logits.argmax(1), direct['logits'].argmax(1)))
            rows.append(dict(index=int(i), same_argmax=same, maximum_probability_error=error, stages=codes))
    passed = all(r['same_argmax'] and r['maximum_probability_error'] <= .002 for r in rows)
    return dict(passed=passed, TRAIN_only=True, records=24, rows=rows,
        scope='Software serial quantized payload replay, not hardware admission', SDK=False)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--frontend', type=Path, required=True)
    p.add_argument('--simulation', type=Path, default=TASK/'runs/simulation')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import torch
    from .train_unified_static import load_arrays
    from .diagnose_frontend_transfer import checked_features
    from .sensor_neutral import encode
    from .static_four import geometry
    from LightGenV2.tasks.t21_continual_learning_hardware.train_native import tensor_sha
    torch.set_num_threads(2)
    if sha(a.frontend) != FRONTEND_SHA:
        raise ValueError('Frozen frontend SHA mismatch')
    train, test = load_arrays(a.data, 'train'), load_arrays(a.data, 'test')
    if set(map(str, train['ids'])) & set(map(str, test['ids'])):
        raise ValueError('TRAIN/TEST source identities overlap')
    indices = selected_training(train)
    train = {k: v[indices] for k, v in train.items()}
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(contract=CONTRACT, created_unix=time.time(), source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), SDK=False, hardware_admitted=False,
        source_delivery='Git only; archive contains no Python source', frontend_SHA256=FRONTEND_SHA,
        frontend_frozen=True, TEST_selected_PT=False, no_independent_new_blind_TEST_claim=True,
        training_subset_rule='50/class/domain from original TRAIN; SHA256(ID) sort; no correctness filtering',
        data_manifest_sha256=sha(a.data/'manifest.json'), datasets={}, cases={}, files=[])
    shutil.copy2(a.frontend, a.out/'frontend.pt')
    encoded = {}
    for split, data, source_ix in (('train', train, indices), ('test', test, np.arange(2000))):
        front = checked_features(data, a.frontend, FRONTEND_SHA)
        pieces = []
        for start in range(0, len(data['labels']), 128):
            images = torch.from_numpy(data['images'][start:start+128])
            pieces.append(encode(images, front[start:start+128].float()).numpy())
        x = np.concatenate(pieces).astype(np.float32)
        if x.shape != (len(source_ix), 224, 224) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError('Invalid original single-sensor encoded inputs')
        np.testing.assert_allclose(np.square(x).sum((1, 2)), 1., atol=3e-6, rtol=3e-6)
        encoded[split] = x
        np.savez_compressed(a.out/(split+'_inputs.npz'), amplitudes=x, labels=data['labels'],
            domains=data['domains'], ids=data['ids'], source_indices=source_ix)
        manifest['datasets'][split] = dict(file=split+'_inputs.npz', records=len(x),
            ids=data['ids'].tolist(), source_indices=source_ix.tolist(),
            domain_class_counts={str(d): np.bincount(data['labels'][data['domains']==d], minlength=10).tolist() for d in (0, 1)},
            identity_sha256=hashlib.sha256(json.dumps(dict(split=split, ids=data['ids'].tolist(),
                domains=data['domains'].tolist(), source_indices=source_ix.tolist()), sort_keys=True).encode()).hexdigest())
        put(a.out/'progress.json', dict(state='encoding', split=split, records=len(x), SDK=False))
    for case, pin in PINS.items():
        path = a.simulation/pin['checkpoint']
        model, cp = load_locked(case, path)
        before = {k: tensor_sha(v) for k, v in model.state_dict().items()}
        refs = reference(case, a.simulation)
        for key in ('ids', 'labels', 'domains'):
            if refs[key] != test[key].tolist():
                raise ValueError('CPU reference/source order mismatch: '+key)
        ix = datasets_for(case, train['domains'])
        audit = bridge_audit(model, encoded['train'][ix])
        put(a.out/(case+'_bridge.json'), audit)
        if not audit['passed']:
            raise RuntimeError('Software bridge failed; do not deploy')
        if before != {k: tensor_sha(v) for k, v in model.state_dict().items()} or sha(path) != pin['sha256']:
            raise RuntimeError('Deployment export changed the model')
        shutil.copy2(path, a.out/(case+'.pt'))
        put(a.out/(case+'_CPU_reference.json'), refs)
        stages = ['router', 'first', 'final'] if case == 'LG' else ['first', 'final']
        manifest['cases'][case] = dict(checkpoint=case+'.pt', checkpoint_sha256=pin['sha256'],
            architecture=cp['architecture'], geometry=geometry(model), TRAIN_indices=ix,
            TEST_indices=list(range(2000)), stages=stages, TRAIN_records=len(ix), TEST_records=2000,
            network_frames=(len(ix)+2000)*len(stages), original_simulation_correct=pin['correct'],
            minimum_experimental_correct={d: n-pin['max_drop_correct'] for d, n in pin['correct'].items()},
            terminal_tuning_only=True, original_readout_parameters=7840, interlayer_tuning=False,
            D2NN_SAR_training=False if case == 'D2NN' else None)
    for path in sorted(a.out.iterdir()):
        if path.name not in ('manifest.json', 'progress.json'):
            manifest['files'].append(dict(name=path.name, bytes=path.stat().st_size, sha256=sha(path)))
    manifest.update(passed=True, seconds=time.monotonic()-started,
        total_network_frames=sum(c['network_frames'] for c in manifest['cases'].values()))
    put(a.out/'manifest.json', manifest)
    archive = a.out.with_suffix('.zip')
    with archive.open('xb') as target:
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as z:
            for path in sorted(a.out.iterdir()):
                if path.name != 'progress.json':
                    z.write(path, path.name)
    put(a.out.with_suffix('.archive.json'), dict(path=archive.name, bytes=archive.stat().st_size,
        sha256=sha(archive), manifest_sha256=sha(a.out/'manifest.json'), SDK=False,
        includes_source_code=False, hardware_admitted=False))
    put(a.out/'progress.json', dict(state='complete_not_hardware_admitted', SDK=False))
    print(json.dumps(dict(passed=True, total_network_frames=manifest['total_network_frames'], SDK=False)))


if __name__ == '__main__':
    main()
