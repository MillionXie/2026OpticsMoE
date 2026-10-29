import unittest
from ..scene_generation_data import bank,condition_subset,request_seed,reference_catalog


class SceneRequestContractTests(unittest.TestCase):
    def test_prompt_coverage_no_brand_no_lighting(self):
        for category in ['lamp','pillow']:
            rows=bank(category);self.assertEqual(len(rows),64)
            self.assertEqual(len({r['prompt'] for r in rows}),64)
            for r in rows:
                self.assertTrue(r['prompt'].startswith('Generate'))
                self.assertGreater(len(r['prompt'].split()),32)
                self.assertNotIn('light from',r['prompt']);self.assertNotIn('B075',r['prompt'])
                self.assertIn('room',r['prompt'])

    def test_known_factors_new_combinations(self):
        for c in ['lamp','pillow']:
            train=condition_subset(c,'train');dev=condition_subset(c,'dev');test=condition_subset(c,'test')
            self.assertEqual((len(train),len(dev),len(test)),(16,8,50))
            self.assertFalse({r['condition_id'] for r in train}&{r['condition_id'] for r in dev})
            self.assertEqual(len({r['scene_index'] for r in train}),8)
            self.assertEqual(len({r['color_index'] for r in train}),4)
            self.assertEqual(len({r['detail_index'] for r in train}),2)

    def test_noise_not_derived_from_caption_or_target(self):
        self.assertEqual(request_seed('raw_source_pixels',0),request_seed('raw_source_pixels',0))
        self.assertNotEqual(request_seed('raw_source_pixels',0),request_seed('raw_source_pixels',1))
        self.assertNotEqual(request_seed('raw_source_pixels',0),request_seed('other_source_pixels',0))

    def test_both_roles_and_family_overlap_rejected(self):
        def item(i,c='lamp'):return dict(sample_id=i,product_id=i,family_id=i,category=c)
        held=[dict(source=item('held'+str(i),'lamp' if i<5 else 'pillow'),target=item('held'+str(i),'lamp' if i<5 else 'pillow')) for i in range(10)]
        rows=[dict(split='train',source=item('a'),target=item('b')),dict(split='val',source=item('c'),target=item('d'))]
        self.assertEqual(len(reference_catalog(rows,held)['train']),2)
        held[0]['source']['family_id']='b'
        with self.assertRaises(ValueError):reference_catalog(rows,held)


if __name__=='__main__':unittest.main()
