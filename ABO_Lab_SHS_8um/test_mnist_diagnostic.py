import unittest
import numpy as np
from mnist_diagnostic import raster,canvas,energy,pcc
class MnistTests(unittest.TestCase):
    def test_physical_raster(self):
        a=np.arange(478*478).reshape(478,478)%256;b=raster(a)
        self.assertEqual(b.shape,(1016,1016))
        # Match the original export's half-open logical bins exactly. At an
        # exact bin boundary floor selects the positive-side pixel; flipping
        # after rasterization is NOT equivalent to flipping before it.
        ix=np.floor((np.arange(1016,dtype=float)+.5-508)*8/17+239).astype(int).clip(0,477)
        self.assertTrue(np.array_equal(b,a[np.ix_(ix,ix)]))
    def test_canvas(self):
        a=np.ones((1016,1016),np.uint8);b=canvas(a,1200)
        self.assertEqual(b.shape,(1200,1920));self.assertEqual(int(b.sum()),1016**2)
        self.assertTrue(np.all(b[92:1108,452:1468]==1))
    def test_raw_energy(self):
        a=np.arange(25).reshape(5,5).astype(float);self.assertEqual(energy(a,[[0,0,2,2]]),[12.0]);self.assertAlmostEqual(pcc(a,2*a+17),1)
