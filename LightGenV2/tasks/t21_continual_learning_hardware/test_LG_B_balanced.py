import unittest,numpy as np,torch
from .fit_LG_B_TRAIN_readout import task_weights,tensor_labels


class LGBBalancedTests(unittest.TestCase):
    def test_windows_np_int32_labels_are_explicitly_long_for_CE(self):
        labels=tensor_labels(np.array([0,1],dtype=np.int32));self.assertEqual(labels.dtype,torch.long)
        loss=torch.nn.functional.cross_entropy(torch.zeros(2,10,dtype=torch.float64,requires_grad=True),labels);loss.backward();self.assertTrue(torch.isfinite(loss))
    def test_unique_memory_counts_do_not_dominate_task_mass(self):
        tasks=np.array(['eurosat']*300+['clevr']*1000);weights=task_weights(tasks)
        self.assertAlmostEqual(float(weights[tasks=='eurosat'].sum()),.5);self.assertAlmostEqual(float(weights[tasks=='clevr'].sum()),.5)


if __name__=='__main__':unittest.main()
