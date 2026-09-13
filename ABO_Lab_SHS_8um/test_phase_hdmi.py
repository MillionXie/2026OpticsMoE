import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from phase_hdmi import load_native,sha,PhaseHDMI
from types import SimpleNamespace
from phase_display import choose_panel

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
    def test_retained_native_mono8_and_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.bmp';a=np.arange(1200*1920,dtype=np.uint8).reshape(1200,1920);Image.fromarray(a).save(p)
            s=PhaseHDMI(d,p);s.settle_s=0;s.wrapper_sha='unknown';s.info={};calls=[]
            def write(ptr,mode):
                calls.append((np.ctypeslib.as_array(ptr,shape=(a.size,)).copy(),mode));return 1
            s.dll=SimpleNamespace(Write_image=write)
            receipt=s.show(p);s.repeat()
            self.assertEqual([x[1] for x in calls],[1,1]);self.assertTrue(receipt['persistent_buffer'])
            np.testing.assert_array_equal(s.pixels,a);np.testing.assert_array_equal(calls[1][0],a.ravel())
            s.dll.Write_image=lambda *args:0
            with self.assertRaises(RuntimeError):s.repeat()
    def test_display_target_restrictions(self):
        r=dict(ids=['MONITOR\\FNR0002\\x'],flags=1,width=1920,height=1200,hz=60)
        self.assertEqual(choose_panel([r]),r)
        for rows in ([],[r,r],[dict(r,flags=5)],[dict(r,width=1080)]):
            with self.assertRaises(RuntimeError):choose_panel(rows)

if __name__=='__main__':unittest.main()
