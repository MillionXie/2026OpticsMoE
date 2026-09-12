import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from phase_hdmi import load_native,sha

class PhaseTests(unittest.TestCase):
    def test_exact_no_transform(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.bmp';a=np.zeros((1200,1920),np.uint8);a[12,20]=123
            Image.fromarray(a).save(p)
            np.testing.assert_array_equal(load_native(p,sha(p)),a)
            with self.assertRaises(ValueError):load_native(p,'wrong')
    def test_reject_wrong_shape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.bmp';Image.fromarray(np.zeros((1080,1920),np.uint8)).save(p)
            with self.assertRaises(ValueError):load_native(p)

if __name__=='__main__':unittest.main()
