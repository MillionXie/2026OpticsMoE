import unittest
from LightGenV2.tasks.t12_text_to_image.scene_omnigen_peer import inputs_for


class OmniGenInputTest(unittest.TestCase):
    def test_corpus_keeps_roles_and_shared_counterfactual_seeds(self):
        sources=[{'sample_id':str(i),'split':'train' if i<8 else 'dev','category':'lamp',
            'white_reference_sha256':str(i),'product_id':str(i),'rgba_file_sha256':'r'+str(i)} for i in range(10)]
        rows=inputs_for(sources,True)
        self.assertEqual(sum(r['split']=='train' for r in rows),32)
        self.assertEqual(sum(r['split']=='dev' for r in rows),4)
        self.assertEqual({r['condition_id'] for r in rows if r['split']=='train'},set(map(str,range(8))))
        group=[r for r in rows if r['reference_id']=='0' and r['seed_index']==0]
        self.assertEqual(len({r['seed'] for r in group}),1)
        with self.assertRaises(AssertionError):inputs_for([dict(sources[0],split='test')],True)

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
