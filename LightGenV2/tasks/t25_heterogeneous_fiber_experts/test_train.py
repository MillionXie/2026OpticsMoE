"""CPU checks for the trainer's optical admission, power archives and interrupted-epoch recovery."""
from argparse import Namespace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import train


class TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_arm_admissions_and_balanced_core_budget(self):
        rng = np.random.default_rng(11)
        raw = {'train': {'images': rng.integers(0, 256, (8, 28, 28, 3), dtype=np.uint8),
                         'labels': np.arange(8), 'ids': np.array([f'train-{i}' for i in range(8)])}}
        initial = set()
        for arm in train.read(train.PROTOCOL)['arms']:
            cfg = train.configuration('kather', 37, arm, 180)
            torch.manual_seed(37)
            model = train.make_model(cfg)
            initial.add(train.paired.state_digest(model))
            identity = {k: cfg[k] for k in ('index_sha256', 'manifest_sha256')}
            with patch.object(train.paired, 'load_raw', return_value=(raw, identity)):
                result = train.smoke(model, cfg, torch.device('cpu'))
            self.assertTrue(result['shared_router_bitwise'])
            self.assertEqual(set(result['labels']), set(range(8)))
            self.assertEqual(list(model.fiber_core_counts), cfg['fiber_core_counts'])
        self.assertEqual(len(initial), 1)
        for seed in (37, 53):
            counts = [train.configuration('blood', seed, arm, 180)['fiber_core_counts'] for arm in ('mixed', 'uniform_half')]
            self.assertEqual([sum(x) for x in counts], [1568, 1568])
            self.assertNotEqual(*counts)

    def test_resume_after_committed_epoch_and_measure_actual_slots(self):
        gen = torch.Generator().manual_seed(739)
        inputs = {s: (torch.rand(n, 224, 224, generator=gen), torch.arange(n) % 8,
                      np.array([f'{s}-{i}' for i in range(n)])) for s,n in [('train',4),('val',2),('test',2)]}
        # Include an actual dark field to test zero transmitted power despite nonzero gate commands.
        inputs['test'][0][0].zero_()
        cfg = train.configuration('kather', 37, 'uniform_half', 2)
        cfg.update(batch=2, expected_counts={s: len(v[1]) for s,v in inputs.items()})
        real_save = train.paired.save_checkpoint
        def interrupted(path, payload):
            real_save(path, payload)
            if path.name == 'last_checkpoint.pt' and payload['epoch'] == 1:
                raise RuntimeError('synthetic interrupt after atomic epoch commit')
        with TemporaryDirectory() as temp, patch.dict(os.environ, {'T25_SOURCE_COMMIT':'cpu-smoke'}), \
                patch.object(train.paired, 'load_inputs', return_value=(inputs, {'synthetic':True})), \
                patch.object(train, 'configuration', return_value=cfg):
            ref = Path(temp)/'reference'
            resumed = Path(temp)/'resumed'
            ref.mkdir()
            resumed.mkdir()
            args = Namespace(out=ref, dataset='kather', seed=37, arm='uniform_half', device='cpu', epochs=2, smoke=False)
            train.run(args)
            args.out = resumed
            with patch.object(train.paired, 'save_checkpoint', side_effect=interrupted), self.assertRaisesRegex(RuntimeError,'synthetic interrupt'):
                train.run(args)
            train.run(args)
            for name in ('last_checkpoint.pt', 'best_checkpoint.pt'):
                a,b = (train.load_pt(p/name) for p in (ref, resumed))
                self.assertTrue(all(torch.equal(t,b['model'][k]) for k,t in a['model'].items()))
            h = train.read(resumed/'history.json')
            self.assertTrue(all(r['train']['n']==4 and r['test']['n']==2 for r in h))
            for split in train.SPLITS:
                with np.load(ref/f'predictions_{split}.npz') as a, np.load(resumed/f'predictions_{split}.npz') as b:
                    self.assertTrue(all(np.array_equal(a[k],b[k]) for k in a.files))
                    train.check_power(b, cfg['fiber_slots'])
            for name, sha in train.read(resumed/'manifest.json').items():
                self.assertEqual(train.paired.sha256(resumed/name), sha)
