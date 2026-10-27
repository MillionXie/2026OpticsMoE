import collections,unittest

from LightGenV2.tasks.t12_text_to_image.qwen_vl_curated import reviewed_pairs


def fixture():
    groups={};descriptions={}
    for split in ('train','val'):
        groups[split]=[]
        for cat in ('lamp','table','pillow'):
            for product in range(2):
                for view in range(2):
                    sku=f'{split}-{cat}-{product}';sid=f'{sku}-{view}'
                    r={'product_id':sku,'family_id':sku,'sample_id':sid,'category':cat,'rgba_sha256':sid}
                    groups[split].append(r);descriptions[sid]=f'a plain {cat} with a clear complete shape'
    return groups,descriptions,{'seed':1042}


class ReviewedPairsTests(unittest.TestCase):
    def test_balanced_roles_and_generic_prompts(self):
        groups,descriptions,profile=fixture();rows=reviewed_pairs(groups,descriptions,profile)
        counts=collections.Counter((r['split'],r['category'],r['mode']) for r in rows)
        self.assertEqual(set(counts.values()),{8})
        for r in rows:
            self.assertNotIn(r['target']['product_id'],r['prompt'])
            if r['mode']=='background':self.assertEqual(r['source'],r['target'])
            else:self.assertNotEqual(r['source']['product_id'],r['target']['product_id'])
            if r['mode']=='object':self.assertEqual(r['source_scene'],r['target_scene'])
            else:self.assertNotEqual(r['source_scene'],r['target_scene'])
        self.assertEqual(rows,reviewed_pairs(groups,descriptions,profile))

    def test_cross_split_target_leakage_is_rejected(self):
        groups,descriptions,profile=fixture()
        groups['val'][0]['product_id']=groups['train'][0]['product_id']
        with self.assertRaises(ValueError):reviewed_pairs(groups,descriptions,profile)

    def test_missing_alternative_product_is_rejected(self):
        groups,descriptions,profile=fixture()
        groups['val']=[r for r in groups['val'] if r['product_id'].endswith('-0')]
        with self.assertRaises(ValueError):reviewed_pairs(groups,descriptions,profile)


if __name__=='__main__':unittest.main()
