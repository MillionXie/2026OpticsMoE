import unittest
import numpy as np
from exposure_batch import image_stats,recommend
from exposure_scan import select_entries

class ExposureTests(unittest.TestCase):
    def test_raw_before_interpolation(self):
        a=np.zeros((10,10),np.uint8);a[0,0]=255;s=image_stats(a,np.ones_like(a,bool))
        self.assertEqual(s['saturation_fraction'],.01)
    def test_clipped_not_recommended(self):
        rows=[]
        for e in [150,300]:
            a=np.arange(100,dtype=np.uint8).reshape(10,10)+20
            if e==300:a[0,:]=255
            rows.append(dict(exposure_us=e,stats=image_stats(a,np.ones_like(a,bool))))
        self.assertEqual(recommend(rows)['recommended_exposure_us'],150)
    def test_all_inputs_must_pass(self):
        good=dict(exposure_us=150,stats=dict(saturation_fraction=0,p999=100,p99=90,dynamic_range=50))
        bad=dict(exposure_us=150,stats=dict(saturation_fraction=0,p999=10,p99=8,dynamic_range=5))
        self.assertIsNone(recommend([good,bad])['recommended_exposure_us'])
    def test_not_selected_by_result(self):
        rows=[dict(id=f'title_{i:03d}') for i in range(100)]+[dict(id=f'image_{i}') for i in range(4)]
        self.assertEqual([r['id'] for r in select_entries(rows)][-3:],['title_000','title_050','title_099'])

if __name__=='__main__':unittest.main()
