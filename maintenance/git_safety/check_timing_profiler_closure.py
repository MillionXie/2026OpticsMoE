"""CPU import/shape audit of historical timing scripts, not a benchmark."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run(repository, commit):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    def blob(path):
        return subprocess.check_output(['git', '-C', str(repository), 'show', commit + ':' + path])
    helper = 'maintenance/git_safety/check_t08_reverse_backend.py'
    namespace = {'__name__': 'git_tree_importer'}
    exec(compile(blob(helper), helper, 'exec'), namespace)
    importer = namespace['GitImporter'](repository, commit, Path('/__git_memory_only__'))
    sys.meta_path.insert(0, importer)
    import torch
    torch.set_num_threads(2)
    manifest = json.loads(blob('maintenance/storage/TIMING_PROFILER_ADDITIONS_20261004.json'))
    for row in manifest['paths']:
        raw = blob(row['path'])
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        compile(raw, row['path'], 'exec')
    full = importlib.import_module('LightGenV2.scripts.profile_latest_optical_electronics_a100')
    narrow = importlib.import_module('LightGenV2.scripts.profile_narrow_optical_electronics_a100')
    assert narrow.full is full and not narrow.RAW and not full.RAW_ROWS
    assert abs(full.PHYSICAL_PASS_MS - 1.0447) < 1e-12
    with torch.inference_mode():
        energy = torch.tensor([[1., 2., 3., 4.]])
        weights = narrow.standard_router_core(energy)
        assert weights.shape == (1, 4) and (weights > 0).sum() == 2
        head = full.LatestOpenMojiHead().eval()
        condition = torch.randn(1, 192)
        outputs = head(torch.randn(1, 192, 14, 14), condition)
        assert all(torch.isfinite(v).all() for v in outputs)
        position = full.PositionReadout(64)
        tokens = [torch.randn(7, 192), torch.randn(13, 192)]
        assert position(tokens).shape == (2, 192)
    imported = list(importer.loaded)
    assert not narrow.RAW and not full.RAW_ROWS
    sys.meta_path.remove(importer)
    return {'commit': commit, 'exact_source_entries': len(manifest['paths']),
            'imports_from_candidate': imported, 'synthetic_cpu_contracts': 3,
            'head_output_shapes': [list(v.shape) for v in outputs],
            'benchmark_executed': False, 'gpu_used': False,
            'dataset_read': False, 'training': False, 'hardware_touched': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    print(json.dumps(run(a.repository, a.commit), indent=2))
