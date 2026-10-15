import unittest
import numpy as np
from amplitude_response import inverse_field,monotone


class ResponseTest(unittest.TestCase):
    def test_inverse_is_field_not_intensity(self):
        x=np.r_[np.arange(0,256,16),255]
        drive,_=inverse_field(x,7+65*(x/255.)**4)
        recovered=(np.asarray(drive)/255.)**4
        self.assertLess(np.max(np.abs(recovered-(np.arange(256)/255.)**2)),.012)
        self.assertEqual(drive[0],0);self.assertEqual(drive[-1],255)
    def test_pava(self):
        np.testing.assert_allclose(monotone([0,3,1,4]),[0,2,2,4])
    def test_dark_rejected(self):
        with self.assertRaises(AssertionError):inverse_field([0,255],[7,7.1])


if __name__=='__main__':unittest.main()
