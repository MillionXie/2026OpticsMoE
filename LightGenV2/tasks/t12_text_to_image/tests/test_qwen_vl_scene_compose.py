import unittest
from ..qwen_vl_scene_compose import description,pick_conditions


class RichScenePairTests(unittest.TestCase):
    def test_description_keeps_real_shade_base_attributes_without_long_arm_list(self):
        text='a table lamp with a dark-blue cylindrical ceramic base, brass-colored foot and a beige cylindrical fabric shade, entirely visible'
        result=description('lamp',text)
        self.assertIn('dark-blue cylindrical ceramic base',result)
        self.assertIn('beige cylindrical fabric shade',result)
        self.assertNotIn('foot',result);self.assertNotIn('entirely visible',result)

    def test_scene_counterfactuals_are_distinct_and_include_single_attribute_changes(self):
        from itertools import product
        pool=[dict(zip(('room','tone','brightness','direction'),p)) for p in product(('study','bedroom','lounge'),('warm','cool','neutral'),('bright','soft'),('left','right'))]
        source=pool[0];rows=pick_conditions(source,pool,8,'source')
        self.assertEqual(len(rows),8);self.assertEqual(len({tuple(r.values()) for r in rows}),8)
        self.assertTrue(all(r!=source for r in rows))
        self.assertTrue(any(sum(r[k]!=source[k] for k in source)==1 for r in rows))
        self.assertEqual(rows,pick_conditions(source,pool,8,'source'))
        self.assertEqual(len(pick_conditions(source,pool,2,'dev-source')),2)


if __name__=='__main__':unittest.main()
