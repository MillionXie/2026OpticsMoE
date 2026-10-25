import unittest,numpy as np,torch,json
from pathlib import Path
from .fit_LG_B_TRAIN_readout import task_weights,tensor_labels


class LGBBalancedTests(unittest.TestCase):
    def test_finite_LG_teacher_sources_and_original_student_bounds(self):
        from .fit_stage_balanced_teacher import validate_profile
        cfg=json.loads(Path(__file__).with_name('config_stage_LG_B500_teacher_20261025.json').read_text());validate_profile(cfg)
        self.assertEqual(cfg['sources_per_task'],dict(eurosat=500,clevr=500));self.assertEqual(cfg['architecture'],'moe')
        with self.assertRaises(ValueError):validate_profile(dict(cfg,sources_per_task=dict(eurosat=1000,clevr=1000)))
        for name,lower in cfg['development_floor'].items():self.assertAlmostEqual(cfg['development_ceiling'][name]-lower,.05)
    def test_windows_np_int32_labels_are_explicitly_long_for_CE(self):
        labels=tensor_labels(np.array([0,1],dtype=np.int32));self.assertEqual(labels.dtype,torch.long)
        loss=torch.nn.functional.cross_entropy(torch.zeros(2,10,dtype=torch.float64,requires_grad=True),labels);loss.backward();self.assertTrue(torch.isfinite(loss))
    def test_unique_memory_counts_do_not_dominate_task_mass(self):
        tasks=np.array(['eurosat']*300+['clevr']*1000);weights=task_weights(tasks)
        self.assertAlmostEqual(float(weights[tasks=='eurosat'].sum()),.5);self.assertAlmostEqual(float(weights[tasks=='clevr'].sum()),.5)


if __name__=='__main__':unittest.main()
