import unittest
from LightGenV2.tasks.t12_text_to_image.scene_omnigen_peer import inputs_for


class OmniGenInputTest(unittest.TestCase):
    def test_only_dev_source_text_and_source_seeds(self):
        source={'sample_id':'lamp-a','split':'dev','category':'lamp','white_reference':'white/a.png',
                'white_reference_sha256':'abc','target':'forbidden.png','source_product_id':'not_for_prompt'}
        rows=inputs_for([source,dict(source,split='train',sample_id='train')])
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['seed'],rows[1]['seed'])
        for row in rows:
            self.assertNotIn('target',row);self.assertNotIn('source_product_id',row)
            self.assertIn('Generate',row['prompt'])


if __name__=='__main__':unittest.main()
