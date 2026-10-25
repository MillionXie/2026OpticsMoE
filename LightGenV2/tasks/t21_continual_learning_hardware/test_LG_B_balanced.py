import unittest,numpy as np
from .fit_LG_B_TRAIN_readout import task_weights


class LGBBalancedTests(unittest.TestCase):
    def test_unique_memory_counts_do_not_dominate_task_mass(self):
        tasks=np.array(['eurosat']*300+['clevr']*1000);weights=task_weights(tasks)
        self.assertAlmostEqual(float(weights[tasks=='eurosat'].sum()),.5);self.assertAlmostEqual(float(weights[tasks=='clevr'].sum()),.5)


if __name__=='__main__':unittest.main()
