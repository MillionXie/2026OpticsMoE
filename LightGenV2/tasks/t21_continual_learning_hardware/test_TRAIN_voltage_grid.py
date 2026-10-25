import unittest
import numpy as np
from .probe_TRAIN_voltage_frequency import voltage_grid,LEVELS,CELLS


class VoltageGridTests(unittest.TestCase):
    def test_native_voltage_codes_not_implicit_phase(self):
        for cell in CELLS:
            for level in LEVELS:
                a=voltage_grid(cell,level);self.assertEqual(a.shape,(1016,1016));self.assertEqual(a.dtype,np.uint8)
                self.assertTrue(set(np.unique(a))<={0,level})
        with self.assertRaises(ValueError):voltage_grid(16,128)


if __name__=='__main__':unittest.main()
