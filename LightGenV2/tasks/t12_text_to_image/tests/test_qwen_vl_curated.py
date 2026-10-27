import collections,unittest

from LightGenV2.tasks.t12_text_to_image.qwen_vl_curated import reviewed_pairs,prior_exposure


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
        target_views=collections.defaultdict(set)
        for r in rows:
            if r['mode']!='background':target_views[r['target']['product_id']].add(r['target']['sample_id'])
        self.assertEqual({len(v) for v in target_views.values()},{1})
        # With just two VAL products, repeated target identity still has a
        # different input background: no contradictory identical input/prompt.
        signatures={}
        for r in rows:
            key=(r['source']['sample_id'],tuple(r['source_scene']),r['prompt'])
            value=(r['target']['sample_id'],tuple(r['target_scene']))
            self.assertEqual(signatures.setdefault(key,value),value)

    def test_cross_split_target_leakage_is_rejected(self):
        groups,descriptions,profile=fixture()
        groups['val'][0]['product_id']=groups['train'][0]['product_id']
        with self.assertRaises(ValueError):reviewed_pairs(groups,descriptions,profile)

    def test_missing_alternative_product_is_rejected(self):
        groups,descriptions,profile=fixture()
        groups['val']=[r for r in groups['val'] if r['product_id'].endswith('-0')]
        with self.assertRaises(ValueError):reviewed_pairs(groups,descriptions,profile)

    def test_multi_source_parent_exposure_is_rejected_across_categories(self):
        groups,descriptions,profile=fixture();rows=reviewed_pairs(groups,descriptions,profile)
        audit=prior_exposure(rows,rows);self.assertFalse(audit['new_val_parent_train_overlap'])
        historical=[dict(r) for r in rows]
        victim=next(r for r in rows if r['split']=='val')['target']
        historical[0]={**historical[0],'target':{**victim,'category':'different_upstream_alias'}}
        with self.assertRaises(ValueError):prior_exposure(rows,historical)

    def test_balanced_four_train_edits_with_new_category(self):
        groups,descriptions,profile=fixture()
        for split,items in groups.items():
            for row in items:
                if row['category']=='table':row['category']='ottoman'
        profile['edits_per_mode']={'train':4,'val':2}
        rows=reviewed_pairs(groups,descriptions,profile)
        counts=collections.Counter((r['split'],r['category'],r['mode']) for r in rows)
        self.assertEqual({n for (s,c,m),n in counts.items() if s=='train'},{16})
        self.assertEqual({n for (s,c,m),n in counts.items() if s=='val'},{8})


if __name__=='__main__':unittest.main()
