import unittest
from ..qwen_vl_scene_backgrounds import specifications
from ..qwen_vl_scene_holdout import contexts


class SealedScenesTest(unittest.TestCase):
    def test_ten_distinct_reference_contexts_without_training_assets(self):
        pool=specifications(layout='tabletop',evaluation_assets=True)
        chosen=contexts(pool,'pillow-example')
        self.assertEqual(len({r['seed'] for r in chosen}),10)
        self.assertEqual(chosen,contexts(pool,'pillow-example'))
        with self.assertRaises(ValueError):contexts(specifications(),'pillow-example')


if __name__=='__main__':unittest.main()
