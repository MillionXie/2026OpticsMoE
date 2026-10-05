"""CPU-only interface/behavior gate for the preserved EuroSAT shared frontend.

Checks SHA-pinned working fixtures, historical Git source and best PT without
datasets, training, metrics selection, GPU, hardware or source-file writes.
Strict loading and synthetic equality do not revalidate historical accuracy.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types


def check(root, manifest):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    task = root/'LightGenV2/demo_check'
    for row in manifest['source_files']:
        path = root/row['path']
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise RuntimeError('Unsafe fixture path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Candidate source differs: '+row['path'])
    import torch
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('audit_shared_frontend', task/'shared_frontend/model.py')
    current = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(current)
    def historical(commit, path, name):
        raw = subprocess.check_output(['git', '-C', str(root), 'show', commit+':'+path])
        module = types.ModuleType(name)
        module.__file__ = str(root/path)
        exec(compile(raw, module.__file__, 'exec'), module.__dict__)
        return module
    records = []
    for record in manifest['runs']:
        folder = task/'runs/simulation'/record['run']
        metadata = json.loads((folder/'metadata.json').read_text())
        commit = record['commit']
        if metadata['git_commit'] != commit:
            raise RuntimeError('Historical run identity changed')
        optics_path = 'LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py'
        raw_optics = subprocess.check_output(['git', '-C', str(root), 'show', commit+':'+optics_path])
        if raw_optics != (root/optics_path).read_bytes():
            raise RuntimeError('Historical propagator differs; separate backend required')
        old = historical(commit, 'LightGenV2/demo_check/pure_optical/models.py', 'audit_historical_optics')
        frontend_path = folder/'frontend/best_checkpoint.pt'
        if hashlib.sha256(frontend_path.read_bytes()).hexdigest() != record['frontend_sha256']:
            raise RuntimeError('Frontend PT changed')
        state = torch.load(frontend_path, map_location='cpu', weights_only=False)['model']
        frontend = current.SharedFrontend(state)
        assert all(not p.requires_grad for p in frontend.parameters())
        # Validate train() cannot unfreeze BatchNorm/dropout mode.
        frontend.train(True)
        assert all(not m.training for m in frontend.modules())
        generator = torch.Generator().manual_seed(20261005)
        images = torch.randint(0, 256, (2, 56, 56, 3), generator=generator, dtype=torch.uint8)
        features = frontend(images)
        amplitude = current.encode_features(features)
        assert amplitude.shape == (2, 224, 224)
        assert torch.allclose(amplitude.square().sum((-2, -1)), torch.ones(2), atol=1e-6)
        comparisons = []
        for architecture in ('dynamic_four', 'full_d2nn'):
            pt = folder/architecture/'best_checkpoint.pt'
            if hashlib.sha256(pt.read_bytes()).hexdigest() != record['optical_sha256'][architecture]:
                raise RuntimeError('Optical PT changed')
            weights = torch.load(pt, map_location='cpu', weights_only=False)['model']
            modern = current.PhaseOnly(architecture, metadata['optical_config'])
            original = old.PhaseOnly(architecture, metadata['optical_config'])
            modern.load_state_dict(weights, strict=True)
            original.load_state_dict(weights, strict=True)
            outputs = [m.forward_amplitude(amplitude) for m in (modern, original)]
            assert torch.equal(outputs[0]['probabilities'], outputs[1]['probabilities'])
            for output in outputs:
                (-output['probabilities'][:, 0].clamp_min(1e-12).log().mean()).backward()
            for (name, parameter), (old_name, old_parameter) in zip(modern.named_parameters(), original.named_parameters()):
                assert name == old_name
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert torch.equal(parameter.grad, old_parameter.grad)
            comparisons.append({'architecture': architecture, 'strict_load': True,
                                'synthetic_probabilities_exact': True, 'synthetic_phase_gradients_exact': True})
        records.append({'run': record['run'], 'frontend_frozen': True,
                        'unit_power': True, 'comparisons': comparisons})
    return {'cpu_only': True, 'dataset_read': False, 'historical_accuracy_revalidated': False,
            'fixture_sources': manifest['source_files'], 'runs': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.root, json.loads(args.manifest.read_text())), indent=2))
