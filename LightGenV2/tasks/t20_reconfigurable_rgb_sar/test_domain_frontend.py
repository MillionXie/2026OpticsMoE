import unittest
from pathlib import Path
from .domain_frontend_probe import configuration,FRONTEND_SHA


class DomainFrontendTests(unittest.TestCase):
    def test_fixed_shared_frontend_and_training_scope(self):
        c=configuration();self.assertEqual(c['frontend_SHA256'],FRONTEND_SHA);self.assertFalse(c['phase_updates']);self.assertFalse(c['TEST_gradients']);self.assertFalse(c['TEST_teacher']);self.assertFalse(c['selection_uses_TEST'])
    def test_weight_norm_is_single_changed_objective(self):
        base=configuration();other=configuration(Path(__file__).with_name('config_domain_frontend_weightnorm_20261025.json'));self.assertEqual({k:v for k,v in other.items() if k!='weight_regularization'},base);self.assertGreater(other['weight_regularization'],0.)
    def test_no_new_electronic_or_test_training_path(self):
        p=Path(__file__).with_name('domain_frontend_probe.py').read_text();self.assertNotIn('nn.Linear(',p);self.assertIn("model.requires_grad_(False)",p);self.assertLess(p.index("locked_before_TEST.json"),p.index("test=load_arrays(a.data,'test')"))
    def test_cpu_audit_never_saves_checkpoint(self):
        p=Path(__file__).with_name('audit_domain_frontend_CPU.py').read_text();self.assertNotIn('torch.save(',p);self.assertIn('no_checkpoint_reselection=True',p)
    def test_TRAIN_raw_dropout_keeps_feature_strips_and_unit_power(self):
        import torch
        from .robust_readout import raw_region_dropout
        x=torch.ones(2,224,224);y=raw_region_dropout(x)
        self.assertEqual(float(y[:,:84].sum()),0.);self.assertEqual(float(y[:,112:196,:112].sum()),0.);self.assertGreater(float(y[:,84:112,:112].sum()),0.);torch.testing.assert_close(y.square().sum((1,2)),torch.ones(2))


if __name__=='__main__':unittest.main()
