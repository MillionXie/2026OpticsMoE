import unittest
from PIL import Image,ImageDraw
import numpy as np
from ..white_reference_outline_probe import outline_from_rgb

PRESET={'white_threshold':245,'minimum_component_area':9,'maximum_product_extent':.68,'center_y':.53,'edge_width':2}


class OutlineConditionTests(unittest.TestCase):
    def test_colour_and_internal_pattern_do_not_change_external_condition(self):
        a=Image.new('RGB',(256,256),'white');ImageDraw.Draw(a).rectangle((28,82,226,166),fill='blue')
        b=Image.new('RGB',(256,256),'white');draw=ImageDraw.Draw(b);draw.rectangle((28,82,226,166),fill='red')
        for x in range(50,200,10):draw.line((x,100,x,145),fill='black',width=4)
        x,meta=outline_from_rgb(a,512,PRESET);y,_=outline_from_rgb(b,512,PRESET)
        np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
        self.assertFalse(meta['GT_or_segmentation_model_used']);self.assertFalse(meta['internal_texture_edges_used'])

    def test_aspect_and_space_for_background(self):
        a=Image.new('RGB',(256,256),'white');ImageDraw.Draw(a).rectangle((20,90,235,150),fill='grey')
        edge,_=outline_from_rgb(a,1024,PRESET);y,x=np.where(np.asarray(edge)[:,:,0]>0)
        self.assertAlmostEqual((x.max()-x.min())/(y.max()-y.min()),215/60,delta=.12)
        self.assertGreater(x.min(),100);self.assertLess(x.max(),924)
        self.assertLess(np.asarray(edge)[:128].max(),1)

    def test_thin_arm_is_not_discarded(self):
        a=Image.new('RGB',(256,256),'white');draw=ImageDraw.Draw(a)
        draw.ellipse((90,205,170,225),fill='grey');draw.line((130,205,130,50,190,50),fill='black',width=3)
        draw.rectangle((178,46,210,80),fill='grey')
        edge,meta=outline_from_rgb(a,512,PRESET)
        self.assertLess(meta['source_bbox_xywh'][1],55);self.assertGreater(np.asarray(edge).max(),0)

    def test_blank_or_nonwhite_reference_rejected(self):
        for image in [Image.new('RGB',(256,256),'white'),Image.new('RGB',(256,256),'grey')]:
            with self.assertRaises(ValueError):outline_from_rgb(image,512,PRESET)


if __name__=='__main__':unittest.main()
