"""Pure scope/prior guards; no Torch, SDK or experiment writes."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
TREE = ast.parse((HERE/'lab_caltech20_dvp_layerwise_20261019.py').read_text())
KEYS = {'phase_sets.9', 'electronic.2.local.weight', 'fusion_logits.2', 'linear.weight', 'linear.bias'}


def scope():
    env = dict(np=np, HERE=HERE, SOURCE_CACHE='38b719b8ad935a0170d059d0e4fffa22f6f22640f979603f7f2543fc2d3f2445',
               allowed_keys=lambda model, prefix: KEYS.copy(),
               read=lambda p: json.loads(Path(p).read_text()),
               sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in ('fitting_keys','fitting_TRAIN_prior')]
    assert len(nodes) == 2
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<scope-prior>','exec'),env)
    return env


class FrozenReadoutTests(unittest.TestCase):
    def test_default_scope_identical(self):
        self.assertEqual(scope()['fitting_keys'](None,4,{}),KEYS)

    def test_freeze_only_removes_readout(self):
        result=scope()['fitting_keys'](None,4,dict(freeze_readout=True))
        self.assertEqual(result,KEYS-{'linear.weight','linear.bias'})

    def test_nonboolean_freeze_rejected(self):
        with self.assertRaises(AssertionError):scope()['fitting_keys'](None,4,dict(freeze_readout='False'))

    def test_default_prior_not_applied(self):
        self.assertIsNone(scope()['fitting_TRAIN_prior']({},{}))
        with self.assertRaises(AssertionError):scope()['fitting_TRAIN_prior'](dict(TRAIN_prior_profile_SHA256='orphan'),{})

    def test_exact_published_TRAIN_prior(self):
        env=scope();p=HERE/'lab_caltech20_DVP_conditioned_head_profile_20261020.json'
        cfg=dict(TRAIN_prior_profile=p.name,TRAIN_prior_profile_SHA256=env['sha'](p))
        dm=dict(source_cache_SHA256=env['SOURCE_CACHE']);prior=env['fitting_TRAIN_prior'](cfg,dm)
        counts=np.array(env['read'](p)['original_TRAIN_counts'],dtype=float)
        np.testing.assert_array_equal(prior,counts/counts.mean())
        self.assertEqual(int(counts.sum()),3141)
        with self.assertRaises(AssertionError):env['fitting_TRAIN_prior'](cfg,dict(source_cache_SHA256='different'))
        with self.assertRaises(AssertionError):env['fitting_TRAIN_prior'](dict(cfg,TRAIN_prior_profile_SHA256='different'),dm)

    def test_external_prior_rejected(self):
        env=scope()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AssertionError):env['fitting_TRAIN_prior'](dict(TRAIN_prior_profile=str(Path(td)/'prior.json'),TRAIN_prior_profile_SHA256='x'),{})

    def test_profile_keeps_hard_gate_and_original_graph(self):
        p=json.loads((HERE/'lab_caltech20_DVP_frozen_readout_suffix_profile_20261020.json').read_text())
        self.assertEqual(p['overrides']['target_correct'],314)
        self.assertEqual((p['overrides']['fit_depth'],p['overrides']['fit_prefix']),(4,4))
        self.assertEqual(p['overrides']['middle_fit'],dict(epochs=6,seconds=480))
        self.assertIs(p['overrides']['freeze_readout'],True)
        self.assertEqual(p['overrides']['teacher_weight'],.3)
        fit=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
        text=ast.unparse(fit)
        self.assertIn('fitting_keys(model, prefix, cfg)',text)
        self.assertIn('original_TRAIN_prior_weights',text)
        self.assertIn('if k not in allowed',text)
        self.assertNotIn('label-ridge-fit',text)


if __name__=='__main__':unittest.main()
