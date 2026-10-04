"""Compare pinned main with original T08 runtime on CPU, not scientific data.

Uses exact Git blobs in memory and existing SHA-bound config/PT fixtures. Does
not checkout code, train, evaluate datasets or mutate original runtime assets.
"""
import argparse
import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = 'd0662a7d240340817948a3496c2cd43f4240e76d'
TASK = 'LightGenV2/tasks/t08_abo_image_text_retrieval/'
MODEL = 'LightGenV2/tasks/t01_object_retrieval/modeling.py'
ROBUST = 'experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/settings.py'


class GitImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, repository, commit, fixture):
        self.repository, self.commit, self.fixture = repository, commit, fixture
        self.tree = set(self.git('ls-tree', '-r', '--name-only', commit).decode().splitlines())
        self.loaded = {}

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repository), *args])

    def location(self, name):
        path = name.replace('.', '/')
        for found, package in ((path+'.py', False), (path+'/__init__.py', True)):
            if found in self.tree:
                return found, package

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in ('LightGenV2', 'experiments'):
            return None
        found = self.location(fullname)
        if found:
            return importlib.util.spec_from_loader(fullname, self, is_package=found[1])

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, package = self.location(module.__name__)
        raw = self.git('show', self.commit+':'+path)
        self.loaded[path] = hashlib.sha256(raw).hexdigest()
        module.__file__ = str(self.fixture/path)
        if package:
            module.__path__ = [str((self.fixture/path).parent)]
        exec(compile(raw.decode(), module.__file__, 'exec'), module.__dict__)


def run(repository, commit, fixture):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    importer = GitImporter(repository, commit, fixture)
    sys.meta_path.insert(0, importer)
    import torch
    torch.set_num_threads(2)
    settings_module = importlib.import_module('LightGenV2.tasks.t01_object_retrieval.settings')
    model = importlib.import_module('LightGenV2.tasks.t01_object_retrieval.modeling')
    # Exact original config fixtures, recursively inherited; no fallback to a
    # different profile when an absolute historical parent has disappeared.
    config = fixture/TASK/'configs/optical_text_to_image_64_10cm_compact_e0p5_no_alpha_curriculum.yaml'
    seen = set()
    import yaml
    current = config
    while current not in seen:
        seen.add(current)
        relative = current.relative_to(fixture).as_posix()
        if current.read_bytes() != importer.git('show', SOURCE+':'+relative):
            raise RuntimeError('Changed config fixture: '+relative)
        raw = yaml.safe_load(current.read_text())
        parent = raw.get('base_config')
        if not parent:
            break
        current = Path(parent)
        if not current.is_absolute():
            current = (config.parent/current).resolve()
        config = current
    else:
        raise RuntimeError('Cyclic config fixtures')
    config = fixture/TASK/'configs/optical_text_to_image_64_10cm_compact_e0p5_no_alpha_curriculum.yaml'
    settings = settings_module.load_settings(config)
    assert settings.language_optical_distance_m == 0.10
    assert settings.language_optical_pixel_pitch_um == 17.0
    # Compare the only changed shared modeling function for this exact profile.
    old_tree = ast.parse(importer.git('show', SOURCE+':'+MODEL))
    fn = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'checkpoint_architecture')
    namespace = {'Any': object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), MODEL, 'exec'), namespace)
    assert namespace['checkpoint_architecture'](settings) == model.checkpoint_architecture(settings)
    pt = Path('/DATA/DATA1/guest3/2026OpticsMoE')/TASK/'runs/simulation/optical_text_to_image_64_10cm_compact_e0p5_no_alpha_curriculum_seed42_20260925/best_checkpoint.pt'
    digest = hashlib.sha256(pt.read_bytes()).hexdigest()
    if digest != 'cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae':
        raise RuntimeError('Formal body PT changed')
    payload = torch.load(pt, map_location='cpu', weights_only=False)
    assert payload['metadata']['optical_architecture'] == model.checkpoint_architecture(settings)
    strict = []
    for direction, cls in (('vision', model.BalancedVisionReplacement), ('language', model.BalancedLanguageReplacement)):
        state = payload[direction+'_optical']
        hidden = int(state['core.input_adapter.weight'].shape[1])
        surrogate = cls(hidden, settings)
        model._install_optical_router(surrogate, settings)
        surrogate.load_state_dict(state, strict=True)
        strict.append(dict(modality=direction, hidden_size=hidden, strict=True))
    readout = model.ElectronicRetrievalReadout(settings.detector_output_size, settings.embedding_dim)
    readout.load_state_dict(payload['retrieval_readout'], strict=True)
    return dict(commit=commit, original_runtime=SOURCE, config_fixtures_verified=len(seen),
                geometry_m=0.10, body_sha256=digest, strict_reloads=strict+['readout'],
                architecture_label_identical_at_10cm=True, imports=importer.loaded,
                dataset_evaluated=False, scientific_metrics_revalidated=False,
                model_forward_equivalence_verified=False, windows_deployment_verified=False,
                runtime_assets_modified=False, cuda_visible_devices='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repository, args.commit, args.fixture), indent=2))
