import unittest
import numpy as np
from PIL import Image,ImageDraw
from ..scene_reference_framing import frame_reference,FRAMING


class ReferenceFraming(unittest.TestCase):
    def test_default_preserves_pixels_and_crop_preserves_all_disconnected_parts(self):
        image=Image.new('RGB',(100,80),'white');d=ImageDraw.Draw(image);d.rectangle((20,25,70,45),fill='black');d.point((10,10),fill=(20,20,20))
        original,meta=frame_reference(image);self.assertTrue(np.array_equal(np.asarray(image),np.asarray(original)))
        cropped,meta=frame_reference(image,FRAMING);self.assertEqual(meta['source_nonwhite_bbox'],[10,10,71,46]);self.assertEqual(cropped.size,(1024,1024))
        self.assertGreater(meta['framed_visible_fraction'],meta['source_visible_fraction']);self.assertFalse(meta['target_or_mask_used']);self.assertFalse(meta['rotation'])
    def test_square_padding_preserves_long_rectangle_aspect(self):
        image=Image.new('RGB',(200,200),'white');ImageDraw.Draw(image).rectangle((60,90,139,109),fill='black')
        framed,meta=frame_reference(image,FRAMING);y,x=np.nonzero(np.asarray(framed).min(axis=2)<100)
        self.assertAlmostEqual((x.max()-x.min()+1)/(y.max()-y.min()+1),4,delta=.08)
    def test_invisible_reference_is_rejected_instead_of_invented(self):
        with self.assertRaises(ValueError):frame_reference(Image.new('RGB',(32,32),'white'),FRAMING)


if __name__=='__main__':unittest.main()
