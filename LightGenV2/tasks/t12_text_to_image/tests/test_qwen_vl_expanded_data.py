import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from LightGenV2.tasks.t12_text_to_image.qwen_vl_expanded_data import view_quality


class ScreeningTests(unittest.TestCase):
    def test_complete_subject_accepted_and_clipped_subject_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);im=root/'rgb.png';mask=root/'mask.png'
            rgb=np.full((128,128,3),180,dtype=np.uint8);rgb[30:100:3,35:94]=90
            Image.fromarray(rgb).save(im)
            alpha=np.zeros((128,128),dtype=np.uint8);alpha[25:105,35:94]=255
            Image.fromarray(alpha).save(mask)
            row={'image_path':str(im),'mask_path':str(mask),'category':'pillow'}
            q=view_quality(row);self.assertIsNotNone(q);self.assertGreater(q['foreground_edge'],0)
            alpha[:40,35:94]=255;Image.fromarray(alpha).save(mask)
            self.assertIsNone(view_quality(row))
            Image.fromarray(np.zeros_like(alpha)).save(mask)
            self.assertIsNone(view_quality(row))

    def test_thin_edge_view_rejected_for_cushion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);im=root/'rgb.png';mask=root/'mask.png'
            Image.new('RGB',(256,256),'white').save(im)
            alpha=np.zeros((256,256),dtype=np.uint8);alpha[40:220,100:125]=255
            Image.fromarray(alpha).save(mask)
            self.assertIsNone(view_quality({'image_path':str(im),'mask_path':str(mask),'category':'pillow'}))


if __name__=='__main__':unittest.main()
