import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from generate_phase_response_patterns import generate


class PatternsTest(unittest.TestCase):
    def test_native_size_and_explicit_polarities(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'patterns';report=generate(out)
            self.assertEqual(len(report['files']),7)
            for row in report['files']:
                with Image.open(out/row['name']) as im:
                    self.assertEqual(im.size,(1920,1200))
                    self.assertEqual(im.mode,'L')
                    self.assertEqual(im.format,'BMP')
            for base in ['P_gx_p8','P_gx_p4','P_lens_10cm']:
                a=np.array(Image.open(out/(base+'.bmp')))
                b=np.array(Image.open(out/(base+'_inverse.bmp')))
                np.testing.assert_array_equal(b,255-a)
            a=np.array(Image.open(out/'P_gx_p8.bmp'))
            np.testing.assert_array_equal(a[:,:8],a[:,8:16])
            with self.assertRaises(FileExistsError):generate(out)


if __name__=='__main__':unittest.main()
