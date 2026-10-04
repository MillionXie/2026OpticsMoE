"""CPU compatibility audit of sealed T07 Git source; never capture or train."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run(repository, commit, checkpoint):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    def blob(path):
        return subprocess.check_output(['git', '-C', str(repository), 'show', commit+':'+path])
    namespace = {'__name__': 'git_tree_importer'}
    helper = 'maintenance/git_safety/check_t08_reverse_backend.py'
    exec(compile(blob(helper), helper, 'exec'), namespace)
    importer = namespace['GitImporter'](repository, commit, Path('/__git_memory_only__'))
    sys.meta_path.insert(0, importer)
    import torch
    torch.set_num_threads(2)
    manifest = json.loads(blob('maintenance/storage/T07_RANK72_SOURCE_ADDITIONS_20261004.json'))
    for row in manifest['paths']:
        raw = blob(row['path'])
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
        assert len(raw) == row['bytes']
        compile(raw, row['path'], 'exec')
    expected = '25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    module = importlib.import_module('LightGenV2.tasks.t07_abo_image_retrieval.standalone.model')
    model = module.OpticalRetrieval(payload['metadata'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    audit = model.audit()
    assert audit['late_rgb_trainable_parameters'] == 92168
    assert audit['late_rgb_adapter'] == 'frozen_patch_7x7_half_rank72'
    assert audit['capture_count'] == 6 and audit['top_k'] == 2
    assert audit['attention_modules'] == 0 and audit['native_transformer_modules'] == 0
    assert all(.44 < x < .45 for values in audit['alpha'].values() for x in values)
    # Synthetic tokens exercise the original bypass and readout, not any dataset.
    generator = torch.Generator().manual_seed(72)
    patches = torch.randn(2, 49, 1024, generator=generator)
    with torch.inference_mode():
        bypass = model.late_rgb(patches)
        positions = torch.ones(2, 49, dtype=torch.bool)
        physical_latent = torch.randn(2, 49, 192, generator=generator)
        result = model.readout(.5*physical_latent + .5*bypass, positions)
    assert result.shape == (2,64) and torch.isfinite(result).all()
    assert torch.allclose(result.norm(dim=-1), torch.ones(2), atol=1e-6)
    # Importing CLI checks the historical preparation/training dependency closure.
    importlib.import_module('LightGenV2.tasks.t07_abo_image_retrieval.standalone.cli')
    sys.meta_path.remove(importer)
    return dict(commit=commit, checkpoint_sha256=expected, strict_load=True,
                sources_checked=len(manifest['paths']), audit=audit,
                imported_source=importer.loaded, synthetic_descriptor_shape=list(result.shape),
                datasets_evaluated=False, training=False, hardware_touched=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repository, args.commit, args.checkpoint), indent=2))
