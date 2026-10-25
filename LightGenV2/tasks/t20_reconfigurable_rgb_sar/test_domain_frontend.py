import unittest
from pathlib import Path
from .domain_frontend_probe import configuration,FRONTEND_SHA


class DomainFrontendTests(unittest.TestCase):
    def test_fixed_shared_frontend_and_training_scope(self):
        c=configuration();self.assertEqual(c['frontend_SHA256'],FRONTEND_SHA);self.assertFalse(c['phase_updates']);self.assertFalse(c['TEST_gradients']);self.assertFalse(c['TEST_teacher']);self.assertFalse(c['selection_uses_TEST'])
    def test_no_new_electronic_or_test_training_path(self):
        p=Path(__file__).with_name('domain_frontend_probe.py').read_text();self.assertNotIn('nn.Linear(',p);self.assertIn("model.requires_grad_(False)",p);self.assertLess(p.index("locked_before_TEST.json"),p.index("test=load_arrays(a.data,'test')"))
    def test_cpu_audit_never_saves_checkpoint(self):
        p=Path(__file__).with_name('audit_domain_frontend_CPU.py').read_text();self.assertNotIn('torch.save(',p);self.assertIn('no_checkpoint_reselection=True',p)


if __name__=='__main__':unittest.main()
