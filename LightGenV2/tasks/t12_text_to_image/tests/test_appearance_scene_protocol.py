import unittest
from ..appearance_scene_protocol import conditions,seed_indices,PROFILE
from ..white_reference_metrics import reward_caption


class AppearanceSceneContractTests(unittest.TestCase):
    def test_requests_and_scoring_do_not_reintroduce_recolor(self):
        colors={'red','blue','green','grey','gray','beige','terracotta','charcoal','cream'}
        for category in ['lamp','pillow']:
            rows=conditions(category);self.assertEqual(len(rows),8)
            self.assertEqual(len({r['prompt'] for r in rows}),8)
            for row in rows:
                for value in [row['prompt'],row['native_caption'],reward_caption(category,row['condition_id'],PROFILE)]:
                    self.assertFalse(colors & set(value.lower().replace(',','').replace('.','').split()))
                self.assertFalse(row['requested_attributes']['change_product_color'])
                self.assertIn('foreground product',row['prompt']);self.assertIn('furnished',row['prompt'])

    def test_real_stochastic_500_not_duplicated_identical_requests(self):
        self.assertEqual(sum(len(seed_indices('test',i)) for i in range(8))*10,500)
        self.assertEqual(sum(len(seed_indices('train',i)) for i in range(8))*76,1216)
        self.assertEqual(sum(len(seed_indices('dev',i)) for i in range(8))*16,128)
        for split in ['train','dev','test']:
            for i in range(8):self.assertEqual(len(seed_indices(split,i)),len(set(seed_indices(split,i))))

    def test_legacy_metric_caption_unchanged(self):
        self.assertIn('blue',reward_caption('lamp','0'))
        self.assertIn('broad cream stripes',reward_caption('pillow','1'))


if __name__=='__main__':unittest.main()
