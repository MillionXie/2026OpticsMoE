import copy
import unittest
import numpy as np
from common import ROOT,config
from patterns import raster
from phase_encoding import encode_gray,generated_root


class PhaseEncodingTests(unittest.TestCase):
    def test_exact_inversion_no_spatial_change(self):
        c=config(ROOT/'lab.json')[0];normal=copy.deepcopy(c)
        c['phase_slm']['gray_encoding']='inverted_255_minus_g'
        a=np.linspace(0,6.28,478*478,dtype=np.float32).reshape(478,478)
        x=raster(a,c['phase_slm'],c,phase=True)
        y=raster(a,normal['phase_slm'],normal,phase=True)
        np.testing.assert_array_equal(x,255-y)
        self.assertEqual(x[0,0],255)
        np.testing.assert_array_equal(raster(a,c['amplitude_slm'],c),raster(a,normal['amplitude_slm'],normal))
        self.assertNotEqual(generated_root(ROOT,c),generated_root(ROOT,normal))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):encode_gray(np.zeros((2,2),np.uint8),{'phase_slm':{'gray_encoding':'unknown'}})

    def test_portable_registration_matches_original(self):
        import sys
        sys.path.insert(0,str(ROOT.parent))
        from experiments.hardware_sdk.generators.dual_slm_alignment import _checker,_registered_checker_grating
        from experiments.hardware_sdk.generators.dual_slm_registration_sweep import large_block_mask,single_axis_masked_grating
        from registration import logical_pairs
        for name,a,p,cell,axis in logical_pairs():
            if 'check' in name:
                aa=_checker(478,cell);pp=_registered_checker_grating(478,cell,8,aa,orientation_mode='visible_checker_cells')
            else:
                aa,_=large_block_mask(478,cell);pp=single_axis_masked_grating(aa,8,axis)
            np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(p,pp)


if __name__=='__main__':unittest.main()
