"""CPU checks for numerical reparameterization, not new inference capacity."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .refine_measured_ce import contrast_transform, fit_convex, internal_indices, continuation_config, _convex_candidate, enable_readout_only


class MeasuredCETest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(73)

    def test_invertible_same_parameter_count_and_exact_fold(self):
        x = torch.rand(40, 12, dtype=torch.float64) + .1
        x /= x.mean(1, keepdim=True)
        transform = contrast_transform(x)
        self.assertEqual(tuple(transform.shape), (12,12))
        self.assertEqual(int(torch.linalg.matrix_rank(transform)), 12)
        weight = torch.randn(10,12, dtype=torch.float64)
        logits = (x @ transform.T) @ weight.T
        folded = x @ (weight @ transform).T
        self.assertLess(float((logits - folded).abs().max()), 1e-10)
        self.assertLess(float(((x @ transform.T)[:, -1] - 1).abs().max()), 1e-12)

    def test_balanced_internal_TRAIN_only_split(self):
        y = torch.arange(10).repeat_interleave(100)
        fit, hold = internal_indices(y)
        self.assertEqual(len(fit),800);self.assertEqual(len(hold),200)
        self.assertFalse(set(fit.tolist()) & set(hold.tolist()))
        self.assertEqual(torch.bincount(y[hold]).tolist(),[20]*10)

    def test_convex_fit_finite_and_only_original_weight_coordinates(self):
        y = torch.arange(10).repeat(5)
        z = torch.nn.functional.one_hot(y,10).double()
        z = torch.cat((z, torch.ones(50,1,dtype=torch.float64)),1)
        weight, loss = fit_convex(z,y,.01,20)
        self.assertEqual(weight.numel(),10*z.shape[1])
        self.assertTrue(torch.isfinite(weight).all())
        self.assertLess(loss,2.3)
        self.assertEqual(int(((z @ weight.T).argmax(1)==y).sum()),50)

    def test_helper_protocol_uses_explicit_capture_identity(self):
        y=torch.arange(10).repeat_interleave(100)
        x=torch.rand(1000,12,dtype=torch.float64)+.1;x/=x.mean(1,keepdim=True)
        with tempfile.TemporaryDirectory() as folder:
            args=SimpleNamespace(out=Path(folder),parent_sha256='parent')
            weight,strength,loss=_convex_candidate(x,y,np.arange(1000),np.arange(1000),args,{'features_sha256':'features'})
            protocol=json.loads((Path(folder)/'protocol.json').read_text())
            self.assertEqual(protocol['features_sha256'],'features')
            self.assertFalse(protocol['TEST_gradients'])
            self.assertEqual(tuple(weight.shape),(10,12))
            self.assertIn(strength,[.001,.01,.1,1.])

    def test_continuation_is_bounded_and_lower_learning_rate(self):
        config=continuation_config()
        self.assertEqual(config['epochs'],120)
        self.assertLess(config['learning_rate'],.001)
        self.assertLess(config['TRAIN_feature_dropout'],.05)
        self.assertEqual(config['LBFGS_steps'],30)

    def test_loaded_frozen_model_unfreezes_only_original_head(self):
        from .refine_native_readout import fit_head
        model=torch.nn.Module()
        model.shared_head=torch.nn.Linear(784,10,bias=False)
        model.upstream=torch.nn.Linear(2,2)
        model.requires_grad_(False)
        enable_readout_only(model)
        self.assertTrue(model.shared_head.weight.requires_grad)
        self.assertFalse(model.upstream.weight.requires_grad)
        config=continuation_config();config.update(epochs=1,LBFGS_steps=1)
        rows=[]
        fit_head(model.shared_head,torch.rand(20,784),torch.arange(10).repeat(2),
                 None,None,config,lambda *row:rows.append(row))
        self.assertEqual([r[0] for r in rows],['baseline','adamw','lbfgs'])


if __name__ == '__main__':
    unittest.main()
