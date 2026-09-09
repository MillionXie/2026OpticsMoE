import unittest
import numpy as np
from common import config,ROOT
from fresnel import four_array,encode,audit_references


class FresnelTests(unittest.TestCase):
    def setUp(self): self.c,_=config(ROOT/'lab.json')

    def test_four_centers_and_no_gap(self):
        bmp,owner,support,m=four_array(self.c)
        self.assertEqual(bmp.shape,(1200,1920)); self.assertEqual(bmp.dtype,np.uint8)
        self.assertTrue(support.all()); self.assertEqual(set(np.unique(owner)),{0,1,2,3})
        np.testing.assert_allclose(m['centers_xy'],[[451.625,91.625],[1467.375,91.625],
                                                  [451.625,1107.375],[1467.375,1107.375]])
        self.assertTrue(m['clipped_by_panel'])
        self.assertGreater(np.count_nonzero(bmp[:,958:962])/4800,.98)
        self.assertGreater(np.count_nonzero(bmp[598:602,:])/7680,.98)
        self.assertTrue(m['sampling']['exceeds_nyquist'])
        self.assertAlmostEqual(m['sampling']['critical_axis_radius_px'],415.625)

    def test_analytic_gray_not_resized_reference(self):
        bmp,owner,_,m=four_array(self.c)
        for x,y in [(70,20),(451,92),(970,710),(1400,1100)]:
            cx,cy=m['centers_xy'][owner[y,x]]
            expected=encode(np.array(((x-cx)**2+(y-cy)**2)*8e-6**2/(2*532e-9*.1)))
            self.assertEqual(bmp[y,x],expected)

    def test_labels_with_vertical_flip(self):
        self.c['phase_slm']['flip_vertical']=True
        *_,m=four_array(self.c)
        self.assertEqual(m['logical_labels_in_physical_order'],['BL','BR','TL','TR'])

    def test_invalid_centers_fail(self):
        self.c['phase_slm']['center_xy']=[100,100]
        with self.assertRaises(ValueError): four_array(self.c)

    def test_original_bmps_when_present(self):
        reports=audit_references(ROOT/'generated/cal/Phase_BMP')
        if not reports: self.skipTest('User reference BMPs are data, not Git source')
        self.assertEqual(len(reports),3)
        for row in reports:
            self.assertEqual(row['max_gray_error'],0)
            self.assertEqual(row['matching_pixel_fraction'],1)


if __name__=='__main__': unittest.main()
