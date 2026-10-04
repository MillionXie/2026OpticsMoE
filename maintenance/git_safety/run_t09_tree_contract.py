"""Strict T09/T16 CPU contracts against Git blobs, without a new checkout.

No datasets, hardware, metrics selection, GPU or source working-file mutation.
Existing config and test fixtures are allowed only after candidate SHA checks.
"""
import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

PREFIXES = ('LightGenV2/tasks/t09_multimodal_matching/',
            'LightGenV2/tasks/t16_zero_phase_ccd_lifelong/',
            'LightGenV2/tasks/t13_four_modal_lifelong/',
            'LightGenV2/tasks/t14_shared_readout_lifelong/',
            'LightGenV2/demo_check/')
OPTICS = 'LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/'


class Importer(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, blobs, fixture):
        self.blobs, self.fixture = blobs, fixture

    def location(self, name):
        path = (OPTICS+name.partition('.')[2].replace('.', '/')
                if name.startswith('optical_reference.') else
                OPTICS.rstrip('/') if name == 'optical_reference' else name.replace('.', '/'))
        for p, package in ((path+'.py', False), (path+'/__init__.py', True)):
            if p in self.blobs:
                return p, package
        if any(p.startswith(path+'/') for p in self.blobs):
            return path+'/__init__.py', True

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'LightGenV2' and not fullname.startswith(('LightGenV2.', 'optical_reference')):
            return None
        found = self.location(fullname)
        return importlib.util.spec_from_loader(fullname, self, is_package=found[1]) if found else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, package = self.location(module.__name__)
        content = self.blobs.get(path, b'')
        module.__file__ = str(self.fixture/path)
        module.__git_blob_sha256__ = hashlib.sha256(content).hexdigest()
        if package:
            module.__path__ = [str((self.fixture/path).parent)]
        exec(compile(content, module.__file__, 'exec'), module.__dict__)


def run(repository, commit, fixture, t16_fixture):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repository), *args])
    def source(path):
        return git('show', commit+':'+path)
    paths = git('ls-tree', '-r', '--name-only', commit, '--', *PREFIXES,
                'LightGenV2/__init__.py', 'LightGenV2/tasks/__init__.py').decode().splitlines()
    blobs = {p: source(p) for p in paths if p.endswith('.py')}
    for path, raw in blobs.items():
        compile(raw.decode(), path, 'exec')
    config = 'LightGenV2/demo_check/pure_optical/config.json'
    if (fixture/config).read_bytes() != source(config):
        raise RuntimeError('Optical config fixture not identical to candidate')
    tests = 'LightGenV2/tasks/t16_zero_phase_ccd_lifelong/tests/'
    for path, raw in blobs.items():
        if path.startswith(tests) and (t16_fixture/path).read_bytes() != raw:
            raise RuntimeError('T16 fixture changed: '+path)
    sys.meta_path.insert(0, Importer(blobs, fixture))
    import torch
    import pytest
    torch.set_num_threads(2)
    mod = importlib.import_module('LightGenV2.tasks.t09_multimodal_matching.model')
    front = importlib.import_module('LightGenV2.tasks.t09_multimodal_matching.vision')
    # All declared runnable core imports must resolve, without calling their CLIs.
    for name in ('run', 'audio_prepare', 'audio_frontend', 'test_selected', 'smoke_layout'):
        importlib.import_module('LightGenV2.tasks.t09_multimodal_matching.'+name)
    for name in ('fixed', 'fixed_dense', 'learned'):
        encoder = mod.TextEncoder(26, name)
        encoded = encoder(torch.tensor([[2, 3, 4]+[0]*29]))
        assert encoded.shape == (1, 32, 64) and not encoded[:, 3:].any()
    encoder = mod.TextEncoder(26, 'fixed')
    text = encoder(torch.tensor([[2, 3, 4]+[0]*29]))
    raw = torch.randint(1, 256, (1, 64, 101, 1), dtype=torch.uint8).expand(-1, -1, -1, 3)
    gradient_cases = []
    for layout in ('legacy', 'two_band', 'interleaved', 'left_right'):
        amplitude = mod.encode(raw, text, layout)
        assert amplitude.shape == (1, 224, 224)
        assert torch.allclose(amplitude.square().sum((-2, -1)), torch.ones(1), atol=1e-6)
        expanded = mod.enlarge_tiles(amplitude, 478, layout)
        assert torch.allclose(expanded.square().sum((-2, -1)), torch.ones(1), atol=1e-6)
        if layout != 'legacy':
            broken = raw.clone(); broken[..., 1] = 0
            try:
                mod.encode(broken, text, layout)
            except AssertionError:
                pass
            else:
                raise RuntimeError('Raw RGB silently discarded')
        for arch in ('moe', 'd2nn'):
            model = mod.OpticalOEO(arch, input_layout=layout, oeo_activation='intensity_softsign')
            output = model(amplitude)
            assert torch.isfinite(output['probabilities']).all()
            assert torch.allclose(output['probabilities'].sum(1), torch.ones(1), atol=1e-6)
            mod.loss(output, torch.tensor([1])).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
                       for p in model.parameters())
            gradient_cases.append(dict(layout=layout, architecture=arch))
            del model
    inventory = json.loads(source('maintenance/storage/T09_RUNTIME_IDENTITY_20261004.json'))
    records = {r['path']: r['sha256'] for r in inventory['protected_asset_records']}
    reloads = []
    for run in ('clevr_visual_softsign_pd005_s17_v1', 'audio_raw_leftright_positive_unbalanced_s17_v1'):
        prefix = 'LightGenV2/tasks/t09_multimodal_matching/runs/simulation/'+run+'/'
        meta_path = prefix+'metadata.json'
        raw_meta = (fixture/meta_path).read_bytes()
        if hashlib.sha256(raw_meta).hexdigest() != records[meta_path]:
            raise RuntimeError('Formal metadata changed')
        cfg = json.loads(raw_meta)['config']
        for arch in ('moe', 'd2nn'):
            rel = prefix+'fixed/'+arch+'/best_checkpoint.pt'
            pt = fixture/rel
            digest = hashlib.sha256(pt.read_bytes()).hexdigest()
            if digest != records[rel]:
                raise RuntimeError('Formal PT changed')
            state = torch.load(pt, map_location='cpu', weights_only=False)
            model = mod.OpticalOEO(arch, cfg['seed'], cfg['phase_dropout'],
                                   input_layout=cfg['input_layout'], oeo_activation=cfg['oeo_activation'])
            model.load_state_dict(state['model'], strict=True)
            reloads.append(dict(run=run, architecture=arch, sha256=digest, strict=True))
            del model
    for run in ('clevr_visual_aux_s17_v1', 'clevr_visual_lite_aux_s17_v1', 'audio_frontend_s17_v1'):
        rel = 'LightGenV2/tasks/t09_multimodal_matching/runs/simulation/'+run+'/best_checkpoint.pt'
        pt = fixture/rel
        digest = hashlib.sha256(pt.read_bytes()).hexdigest()
        if digest != records[rel]:
            raise RuntimeError('Frontend PT changed')
        state = torch.load(pt, map_location='cpu', weights_only=False)
        model = front.VisionEncoder(state['model']['head.weight'].shape[0],
                                    state.get('architecture', {}).get('widths', [16, 32, 64]))
        model.load_state_dict(state['model'], strict=True)
        reloads.append(dict(run=run, sha256=digest, strict=True))
    code = int(pytest.main([str(t16_fixture/tests), '-q', '-p', 'no:cacheprovider']))
    if code:
        raise RuntimeError('T16 contracts failed')
    expected = hashlib.sha256(blobs['LightGenV2/tasks/t09_multimodal_matching/model.py']).hexdigest()
    if mod.__git_blob_sha256__ != expected:
        raise RuntimeError('Not candidate model')
    return dict(commit=commit, python_blobs_compiled=len(blobs),
                optical_gradient_cases=gradient_cases, strict_reloads=reloads,
                t16_test_exit_code=code, candidate_model_sha256=expected,
                config_fixture_verified=True, dataset_evaluated=False,
                runtime_metrics_revalidated=False, new_checkout_created=False,
                cuda_visible_devices='')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--t16-fixture', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.repository, a.commit, a.fixture, a.t16_fixture), indent=2))
