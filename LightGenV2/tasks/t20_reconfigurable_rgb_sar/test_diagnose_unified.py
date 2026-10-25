import unittest
from pathlib import Path
import numpy as np
from .diagnose_unified import ridge_weights,class_scores


class DiagnosticTests(unittest.TestCase):
    def test_ridge_is_original_no_bias_linear(self):
        rng=np.random.default_rng(7);y=np.repeat(np.arange(10),20);x=np.eye(10)[y]+rng.normal(0,.01,(200,10));w=ridge_weights(x,y,.01)
        self.assertEqual(w.shape,(10,10));self.assertTrue(np.isfinite(w).all());self.assertEqual((x@w.T).argmax(1).tolist(),y.tolist())
    def test_diagnostic_does_not_load_test_or_save_weights(self):
        source=Path(__file__).with_name('diagnose_unified.py').read_text()
        self.assertNotIn("load_arrays(a.data,'test')",source);self.assertNotIn('torch.save(',source)
    def test_sensor_scores_keep_separate_denominators(self):
        logits=np.eye(10)[[0,1,2,4]];r=class_scores(logits,np.array([0,1,2,3]),np.array([0,0,1,1]))
        self.assertEqual(r['RGB']['accuracy'],1.);self.assertEqual(r['SAR']['accuracy'],.5)
    def test_four_expert_diagnostic_uses_exact_frontend_and_input(self):
        source=Path(__file__).with_name('diagnose_unified.py').read_text()
        self.assertIn('build_static_four if four',source);self.assertIn('neutral_amplitude if four',source)
        self.assertIn("checked_features(data,path,state['frontend_sha256'])",source);self.assertIn('per_process_GPU_memory_fraction=.2',source)
        self.assertIn('TRAIN_pair_alignment',source)
    def test_phase_gradient_diagnostic_never_updates_or_loads_TEST(self):
        source=Path(__file__).with_name('diagnose_phase_gradients.py').read_text()
        self.assertNotIn("load_arrays(a.data,'test')",source);self.assertNotIn('optimizer.step(',source);self.assertNotIn('torch.save(',source)
        self.assertIn('TRAIN_unique_locations=50',source);self.assertIn('parameter_updates=False',source)


if __name__=='__main__':unittest.main()
