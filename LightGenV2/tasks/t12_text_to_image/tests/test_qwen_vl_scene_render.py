import unittest
import numpy as np
from ..qwen_vl_scene_render import light_field,unmatte_white,recolor_component


class SceneDataRenderTests(unittest.TestCase):
    def test_white_unmatting_avoids_double_alpha_edge(self):
        rgb=np.array([[[255,128,128]]],dtype=np.uint8);alpha=np.array([[128]],dtype=np.uint8)
        fg,a=unmatte_white(rgb,alpha)
        self.assertGreater(fg[0,0,0],254)
        self.assertLess(fg[0,0,1],2)
        self.assertAlmostEqual(float(a[0,0]),128/255,places=6)

    def test_direction_tone_exposure_are_real_pixel_controls(self):
        right=light_field((8,8),'neutral','bright','right')
        left=light_field((8,8),'neutral','bright','left')
        np.testing.assert_allclose(right[:,::-1],left)
        self.assertGreater(right[0,-1,0],right[0,0,0])
        warm=light_field((8,8),'warm','bright','right')
        self.assertGreater(warm[0,-1,0],warm[0,-1,2])
        dim=light_field((8,8),'neutral','softly lit','right')
        self.assertTrue((dim<right).all())

    def test_declared_component_color_changes_and_rejects_missing_component(self):
        fg=np.zeros((64,64,3),dtype=np.float32);fg[:40]=[20,110,35]
        alpha=np.ones((64,64),dtype=np.float32)
        changed=recolor_component(fg,alpha,{'part':'shade','color':'red'})
        self.assertGreater(float(changed[10,10,0]),float(changed[10,10,1]))
        np.testing.assert_array_equal(changed[50],fg[50])
        with self.assertRaises(ValueError):
            recolor_component(np.ones_like(fg)*128,alpha,{'part':'shade','color':'red'})


if __name__=='__main__':
    unittest.main()
