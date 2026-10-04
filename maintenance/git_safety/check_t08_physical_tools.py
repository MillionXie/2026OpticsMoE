"""Audit migrated T08 replay tools from Git, without data/model/hardware runs."""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def run(repository, commit):
    def blob(path):
        return subprocess.check_output(['git', '-C', str(repository), 'show', commit+':'+path])
    helper = 'maintenance/git_safety/check_t08_reverse_backend.py'
    namespace = {'__name__': 't08_git_importer'}
    exec(compile(blob(helper), helper, 'exec'), namespace)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    importer = namespace['GitImporter'](repository, commit, Path('/__git_memory_only__'))
    sys.meta_path.insert(0, importer)
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    manifest = json.loads(blob('maintenance/storage/T08_PHYSICAL_TOOLS_IMPORT_20261004.json'))
    modules = {}
    for row in manifest['paths']:
        path = row['published_path']
        source = blob(path)
        assert hashlib.sha256(source).hexdigest() == row['published_sha256']
        old = repository/row['source_path']
        original = old.read_bytes()
        assert hashlib.sha256(original).hexdigest() == row['source_sha256']
        trees = [ast.parse(original), ast.parse(source)]
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    node.module, node.level = 'reviewed_import', 0
        assert ast.dump(trees[0]) == ast.dump(trees[1])
        modules[Path(path).stem] = importlib.import_module(path[:-3].replace('/', '.'))
    stage = modules['export_full_next_stage']
    contract = SimpleNamespace(train=[SimpleNamespace(sku_index=i//48) for i in range(4800)])
    keys = stage.selected_train_keys(contract, 8, 20260926)
    assert len(keys) == len(set(keys)) == 800
    assert all(key.startswith('train_') for key in keys)
    assert keys == stage.selected_train_keys(contract, 8, 20260926)
    assert all(sum(int(key[6:])//48 == i for key in keys) == 8 for i in range(100))
    settings = SimpleNamespace(optical_router_detector_intervals=((0, 2), (2, 4)),
                               optical_router_energy_eps=1e-8,
                               optical_router_score_normalization='standardized_region_energy',
                               router_temperature=1., top_k=2, router_weight_normalization='power_l2')
    image = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)
    route = stage.router_scores(image, settings)
    assert route['selected_mask'].sum().item() == 2
    assert set(route['selected_indices'][0].tolist()) == {2, 3}
    assert torch.allclose(route['weights'].square().sum(-1), torch.ones(1))
    with tempfile.TemporaryDirectory(prefix='t08_ccd_fixture_') as scratch:
        root = Path(scratch)
        captured = root/'01_vision_router'/'ccd_captured'
        captured.mkdir(parents=True)
        Image.fromarray(np.full((478, 478), 23, dtype=np.uint8)).save(captured/'fixture.png')
        assert stage.ccd(root, 'vision_router', 'fixture').shape == (478, 478)
        Image.fromarray(np.full((20, 20), 23, dtype=np.uint8)).save(captured/'bad.png')
        try:
            stage.ccd(root, 'vision_router', 'bad')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Malformed CCD accepted')
    # CPU synthetic readout metric check only; no training or scientific data.
    features = torch.eye(100).repeat_interleave(2, dim=0)
    metrics = modules['finetune_readout'].metrics(torch.nn.Identity(), features, torch.eye(100),
                                                 torch.arange(100).repeat_interleave(2))
    assert metrics['hit_at_1'] == 1. and metrics['mrr'] == 1.
    assert metrics['gallery_count'] == 200 and metrics['relevant_images_per_query'] == 2
    return dict(commit=commit, source_modules_checked=len(modules),
                source_bodies_identical_after_import_relocation=True,
                synthetic_train_selection_images=800, canonical_ccd_validation=True,
                router_top2_power_normalization=True, synthetic_readout_metrics=True,
                data_evaluated=False, models_loaded=False, hardware_touched=False,
                runtime_assets_modified=False, imported_sources=importer.loaded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repository, args.commit), indent=2))
