import unittest
import numpy as np
from PIL import Image
from ..qwen_vl_scene_curriculum_data import view_quality


class SceneScreenTests(unittest.TestCase):
    def test_reject_empty_clipped_or_edge_on_products(self):
        for category in ('lamp', 'pillow'):
            self.assertIsNone(view_quality(Image.new('L', (64, 64), 0), category))
        a=np.zeros((64,64),dtype=np.uint8);a[0:55,5:40]=255
        self.assertIsNone(view_quality(Image.fromarray(a), 'lamp'))
        a[:]=0;a[5:55,29:32]=255
        self.assertIsNone(view_quality(Image.fromarray(a), 'pillow'))

    def test_accept_full_bulky_views_without_model_scores(self):
        a=np.zeros((64,64),dtype=np.uint8);a[8:56,12:52]=255
        for category in ('lamp', 'pillow'):
            result=view_quality(Image.fromarray(a),category)
            self.assertEqual(result['bbox'],[12,8,52,56])
            self.assertGreater(result['margin'],1)


if __name__ == '__main__':
    unittest.main()
