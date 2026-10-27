import unittest
from ..qwen_vl_scene_backgrounds import specifications


class BackgroundDataContract(unittest.TestCase):
    def test_balanced_disjoint_teacher_assets_without_heldout(self):
        rows=specifications();self.assertEqual(len(rows),108)
        self.assertEqual(sum(r['split']=='train' for r in rows),72)
        self.assertEqual(sum(r['split']=='val' for r in rows),36)
        self.assertEqual({r['split'] for r in rows},{'train','val'})
        train={r['seed'] for r in rows if r['split']=='train'};dev={r['seed'] for r in rows if r['split']=='val'}
        self.assertFalse(train&dev)
        self.assertEqual(rows,specifications())
        self.assertTrue(all(r['direction']+' window light' in r['prompt'] for r in rows))


if __name__ == '__main__':
    unittest.main()
