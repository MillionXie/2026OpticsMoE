import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT,config
from patterns import raster
from dual_patterns import logical_pairs,generate

class PairedGratingTests(unittest.TestCase):
    def setUp(self): self.c,_=config(ROOT/'lab.json')
    def test_phase_only_inside_white_cells(self):
        for name,a,p,_,_ in logical_pairs():
            self.assertTrue(np.all(p[a==0]==0),name)
            self.assertEqual(set(np.unique(a)),{0,255})
            self.assertEqual(set(np.unique(p)),{0,128})
    def test_native_edges_coincide_on_both_panels(self):
        for _,a,p,_,_ in logical_pairs():
            ac=raster(a,self.c['amplitude_slm'],self.c,nearest=True)[32:1048,452:1468]
            support=raster(a,self.c['phase_slm'],self.c,nearest=True)[92:1108,452:1468]
            pc=raster(p,self.c['phase_slm'],self.c,nearest=True)[92:1108,452:1468]
            np.testing.assert_array_equal(ac,support)
            self.assertTrue(np.all(pc[ac==0]==0))
    def test_orientation_is_applied_inside_active_field(self):
        _,a,p,_,_=logical_pairs()[1]
        c=copy.deepcopy(self.c); c['phase_slm']['flip_vertical']=True
        normal=raster(p,self.c['phase_slm'],self.c,nearest=True)[92:1108,452:1468]
        flipped=raster(p,c['phase_slm'],c,nearest=True)[92:1108,452:1468]
        np.testing.assert_array_equal(flipped,np.flipud(normal))
    def test_saved_pairs_binary_native_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=generate(self.c,Path(tmp))
            self.assertEqual(len(report['pairs']),4)
            for row in report['pairs']:
                for f,size,values in [('A.bmp',(1920,1080),{0,255}),('P.bmp',(1920,1200),{0,128}),('P_V.bmp',(1920,1200),{0,128})]:
                    with Image.open(Path(tmp)/row['folder']/f) as im:
                        self.assertEqual(im.mode,'L'); self.assertEqual(im.size,size)
                        self.assertEqual(set(np.unique(np.array(im))),values)
                        im.close() # Release Pillow's BMP mmap before Windows temp cleanup.

if __name__=='__main__': unittest.main()
