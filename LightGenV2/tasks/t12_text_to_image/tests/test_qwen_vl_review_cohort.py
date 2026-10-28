import unittest

from LightGenV2.tasks.t12_text_to_image.qwen_vl_scene_review import review_ids


class ReviewCohortTests(unittest.TestCase):
    def test_train_probe_keeps_counterfactuals_but_excludes_dev_and_extra_views(self):
        rows=[]
        for category in ('lamp','pillow'):
            for split in ('train','val'):
                for product in range(3):
                    for view in range(2):
                        for mode in ('background','object','joint'):
                            for variant in range(2):
                                rows.append({'split':split,'category':category,'mode':mode,
                                    'source':{'product_id':str(product),'sample_id':f'{category}:{product}:{view}'},
                                    'prompt':str(variant)})
        selected=[rows[i] for i in review_ids(rows,'train_probe')]
        self.assertEqual(len(selected),24)
        self.assertTrue(all(r['split']=='train' and r['source']['sample_id'].endswith(':0') for r in selected))
        self.assertEqual({r['source']['product_id'] for r in selected},{'0','1'})
        self.assertEqual({r['prompt'] for r in selected},{'0','1'})
        self.assertEqual(review_ids(rows,'development'),[i for i,r in enumerate(rows) if r['split']=='val'])

    def test_incomplete_train_probe_is_rejected(self):
        with self.assertRaises(ValueError):review_ids([],'train_probe')
        with self.assertRaises(ValueError):review_ids([],'test')
