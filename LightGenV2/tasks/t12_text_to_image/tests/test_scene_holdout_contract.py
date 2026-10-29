import unittest
from LightGenV2.tasks.t12_text_to_image.scene_two_sentence_holdout import test_inventory as inventory
from LightGenV2.tasks.t12_text_to_image.appearance_scene_protocol import conditions,SHORT_PROFILE,seed_indices


class SceneHoldoutContract(unittest.TestCase):
    def rows(self):
        return [dict(split='test',category='lamp' if i<5 else 'pillow',source_product_id=str(i),
            source_family_id=str(i),image_sha256='test'+str(i),license='CC BY 4.0',source_member=str(i)+'.png') for i in range(10)]

    def test_ten_disjoint_products(self):
        self.assertEqual(len(inventory(self.rows(),[dict(source_product_id='train',reference_sha256='train')])),10)

    def test_reject_product_or_family_overlap(self):
        with self.assertRaises(AssertionError):inventory(self.rows(),[dict(source_product_id='0',reference_sha256='train')])
        rows=self.rows()+[dict(split='train',source_family_id='0')]
        with self.assertRaises(AssertionError):inventory(rows,[])

    def test_two_sentences_and_five_hundred_draws(self):
        n=0
        for category in ['lamp','pillow']:
            scenes=conditions(category,SHORT_PROFILE)
            self.assertEqual(len(scenes),8)
            for r in scenes:
                self.assertEqual(r['prompt'].count('.'),2)
                self.assertEqual(r['prompt'],r['native_caption'])
                self.assertFalse(r['requested_attributes']['change_product_color'])
                n+=5*len(seed_indices('test',int(r['condition_id'])))
        self.assertEqual(n,500)


if __name__=='__main__':unittest.main()
