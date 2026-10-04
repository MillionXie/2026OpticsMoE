"""CPU-only Git-tree dependency and synthetic-helper audit; never train or acquire."""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


def run(repository, commit):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True

    def blob(path):
        return subprocess.check_output(['git', '-C', str(repository), 'show', commit + ':' + path])

    manifest = json.loads(blob('maintenance/storage/T07_TRAINING_CLOSURE_ADDITIONS_20261004.json'))
    controls = {}
    for row in manifest['paths']:
        raw = blob(row['path'])
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        if row['path'].endswith('.py'):
            compile(raw, row['path'], 'exec')
        else:
            json.loads(raw)
            controls[Path(row['path']).name] = raw.decode()
    namespace = {'__name__': 'git_tree_importer'}
    helper = 'maintenance/git_safety/check_t08_reverse_backend.py'
    exec(compile(blob(helper), helper, 'exec'), namespace)
    importer = namespace['GitImporter'](repository, commit, Path('/__git_memory_only__'))
    sys.meta_path.insert(0, importer)
    try:
        import torch
        torch.set_num_threads(2)
        prefix = 'LightGenV2.tasks.t07_abo_image_retrieval.standalone.'
        modules = {Path(row['path']).stem: importlib.import_module(prefix + Path(row['path']).stem)
                   for row in manifest['paths'] if row['path'].endswith('.py')}
        refine = modules['retrieval_refine']
        for name in ('configure_phase_head_scope', 'attach_train_readout_dropout',
                     'train_ranking_loss', 'configure_alpha_scope',
                     'frozen_except_alpha_digest', 'blend_train_pairs',
                     'training_source_exclusion'):
            assert callable(getattr(refine, name))
        # Paths from the virtual Git importer do not exist on disk. Read only
        # exact Git control blobs, including the historical recursive overlays.
        original_read = Path.read_text
        controls_read = set()

        def control_read(path, *args, **kwargs):
            if str(path).startswith('/__git_memory_only__/') and path.name in controls:
                controls_read.add(path.name)
                return controls[path.name]
            return original_read(path, *args, **kwargs)

        generalization = modules['generalization']
        base = dict(augmentation={}, adapt={}, teacher_alignment_sha256='synthetic',
                    teacher_alignment_origin_checkpoint_sha256='synthetic')
        with patch.object(Path, 'read_text', control_read):
            profiles = {name: generalization.overlay_config(copy.deepcopy(base), name)
                        for name in generalization.PROFILES}
        assert set(controls) == controls_read
        # Synthetic identities preserve the enrolled SKU split contract.
        train = [dict(split='train', product_id=str(sku), sample_id=f'train-{sku}-{view}')
                 for sku in range(200) for view in range(8)]
        query = [dict(split='query', product_id=str(sku), sample_id=f'query-{sku}-{view}')
                 for sku in range(200) for view in range(4)]
        fit, selection, split_audit = modules['robust_holdout'].split_train(
            dict(train=train, query=query))
        assert len(fit['train']) == 1200 and len(selection['query']) == 400
        assert len(split_audit['validation_ids']) == 400
        # Nonself multi-positive TRAIN ranking has finite gradients.
        positive = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0],
                                 [0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
        excluded = torch.eye(4, dtype=torch.bool)
        for kind in ('nll', 'top1_softplus', 'top1_squared_hinge', 'hybrid_nll_top1'):
            logits = torch.randn(4, 4, requires_grad=True)
            loss = refine.train_ranking_loss(logits, positive, excluded, kind)
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
            assert torch.equal(logits.grad.diag(), torch.zeros(4))
        sources = [('a', 'b'), ('c',)]
        mask = refine.training_source_exclusion(sources, ['a', 'b', 'c', 'd'])
        assert mask.tolist() == [[True, True, False, False], [False, False, True, False]]
        # SAM checks helper restoration, not the sealed model or a dataset.
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        before = parameter.detach().clone()
        optimizer = torch.optim.SGD([parameter], lr=.01)
        calls = []

        def closure():
            calls.append(1)
            return dict(loss=parameter.square().sum())

        _, sam = generalization.backward_with_sam(closure, optimizer, rho=.03)
        assert len(calls) == 2 and torch.equal(parameter.detach(), before)
        assert torch.isfinite(parameter.grad).all() and sam['rho'] == .03
        shifted = modules['robust_training'].shift_zero(torch.ones(3, 5, 5), maximum=1)
        assert shifted.shape == (3, 5, 5) and torch.isfinite(shifted).all()
        return dict(commit=commit, sha_checked=len(manifest['paths']),
                    dynamic_profiles_checked=len(profiles), controls_read=sorted(controls_read),
                    imported_source=importer.loaded, synthetic_split=[1200, 400, 800],
                    ranking_gradient_checks=4, sam_restored=True,
                    datasets_evaluated=False, sealed_model_modified=False,
                    training=False, hardware_touched=False,
                    historical_training_reproduced=False)
    finally:
        sys.meta_path.remove(importer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repository, args.commit), indent=2))
