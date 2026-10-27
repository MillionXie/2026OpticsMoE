import unittest
import numpy as np
from ..qwen_vl_scene_render import light_field,unmatte_white,recolor_component


class SceneDataRenderTests(unittest.TestCase):
    def test_hsv_component_repairs_dark_green_without_flat_clipping_or_brass_edits(self):
        fg=np.zeros((64,64,3),dtype=np.float32);alpha=np.zeros((64,64),dtype=np.float32)
        for x in range(10,54):fg[8:36,x]=np.array([.2,1.,.3])*(20+x*2);alpha[8:36,x]=1
        fg[16:20,24:28]=[140,90,30];fg[24:27,36:39]=[220,220,220]
        new=recolor_component(fg,alpha,{'part':'shade','color':'red','recipe':'hsv_component_hull_v2'})
        self.assertGreater(float(new[12,11,0]),float(new[12,11,1]))
        self.assertGreater(float(new[12,51,0]),float(new[12,51,1]))
        np.testing.assert_allclose(new.max(2),fg.max(2),atol=1e-5)
        np.testing.assert_array_equal(new[16:20,24:28],fg[16:20,24:28])
        np.testing.assert_array_equal(new[24:27,36:39],fg[24:27,36:39])

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
