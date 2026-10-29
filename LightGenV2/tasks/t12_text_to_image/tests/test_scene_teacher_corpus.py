import unittest
from LightGenV2.tasks.t12_text_to_image.appearance_scene_protocol import PROFILE,COARSE_PROFILE,SHORT_PROFILE,conditions
from LightGenV2.tasks.t12_text_to_image.scene_teacher_corpus import requests_for


class CoarseSceneDataTest(unittest.TestCase):
    def test_two_sentence_contract_no_extra_native_requests(self):
        for category in ['lamp','pillow']:
            for row in conditions(category,SHORT_PROFILE):
                self.assertEqual(row['prompt'].count('.'),2)
                self.assertLessEqual(len(row['prompt'].split()),45)
                self.assertEqual(row['prompt'],row['native_caption'])
                self.assertFalse(row['requested_attributes']['retain_original_appearance'])
    def sources(self):
        return [{'sample_id':category+str(i),'product_id':category+str(i),'split':'train' if i<4 else 'dev',
                 'category':category,'white_reference_sha256':category+str(i),'rgba_file_sha256':'rgba'+category+str(i)}
                for category in ['lamp','pillow'] for i in range(5)]

    def test_old_contract_is_preserved_new_contract_does_not_force_detail(self):
        old=conditions('lamp');new=conditions('lamp',COARSE_PROFILE)
        self.assertIn('original appearance',old[0]['prompt'])
        self.assertNotIn('original appearance',new[0]['prompt'])
        self.assertEqual(old[0]['prompt_profile'],PROFILE)
        self.assertTrue(new[0]['requested_attributes']['fine_details_free'])
        self.assertFalse(new[0]['requested_attributes']['retain_original_appearance'])

    def test_pilot_is_fixed_train_only(self):
        rows=requests_for(self.sources(),True)
        self.assertEqual(len(rows),32)
        self.assertEqual(len({r['source_product_id'] for r in rows}),8)
        self.assertTrue(all(r['split']=='train' for r in rows))

    def test_seed_depends_on_source_and_index_not_scene(self):
        rows=requests_for(self.sources())
        group=[r for r in rows if r['reference_id']==rows[0]['reference_id'] and r['seed_index']==0]
        self.assertEqual(len(group),8)
        self.assertEqual(len({r['seed'] for r in group}),1)
        bad=self.sources();bad[0]['split']='test'
        with self.assertRaises(AssertionError):requests_for(bad)

    def test_small_budget_still_covers_all_rooms_and_products(self):
        rows=requests_for(self.sources(),False,[],2)
        train=[r for r in rows if r['split']=='train']
        self.assertEqual(len(train),32)
        self.assertEqual(len({r['source_product_id'] for r in train}),8)
        self.assertEqual({r['condition_id'] for r in train},set(map(str,range(8))))


if __name__=='__main__':unittest.main()
