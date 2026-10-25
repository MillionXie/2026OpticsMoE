import unittest
import numpy as np
from .audit_TRAIN_spatial_response import blur


class BlurTests(unittest.TestCase):
    def test_identity(self):
        a=np.arange(64.).reshape(8,8);np.testing.assert_array_equal(blur(a,0),a)
    def test_constant(self):
        np.testing.assert_allclose(blur(np.ones((32,32)),2),1,atol=1e-14)
    def test_lowpass(self):
        a=np.indices((32,32)).sum(0)%2;self.assertLess(blur(a,2).std(),.01)
    def test_reject(self):
        with self.assertRaises(ValueError):blur(np.ones((4,4)),-1)


if __name__=='__main__':unittest.main()
