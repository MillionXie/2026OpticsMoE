import copy
import unittest
from LightGenV2.tasks.t12_text_to_image.scene_paper_expanded_data import conditions,make_requests,select_checkpoint


class ExpandedDataTest(unittest.TestCase):
    def refs(self):
        refs=[]
        for split,cat,n in [('train','lamp',36),('train','pillow',40),('test','lamp',13),('test','pillow',13)]:
            for i in range(n):
                rid=f'{split}-{cat}-{i}'
                refs.append(dict(split=split,category=cat,reference_id=rid,reference=rid+'.png',reference_sha256=rid,
                    source_product_id=rid,source_family_id=rid))
        return refs

    def test_single_seed_counts_and_counterfactuals(self):
        train,test=make_requests(self.refs());self.assertEqual((len(train),len(test)),(1520,500))
        for rows in [train,test]:
            for rid in {r['reference_id'] for r in rows}:
                group=[r for r in rows if r['reference_id']==rid]
                self.assertEqual(len({r['seed'] for r in group}),1)
                self.assertEqual(len({r['prompt'] for r in group}),len(group))
        for cat in ['lamp','pillow']:
            bank=conditions(cat);self.assertEqual(len(bank),20)
            self.assertTrue(all(r['prompt'].count('.')==2 and 'reference'not in r['prompt'].lower() for r in bank))

    def test_leakage_and_extra_validation_rejected(self):
        refs=self.refs();refs[-1]['source_product_id']=refs[0]['source_product_id']
        with self.assertRaises(ValueError):make_requests(refs)
        refs=self.refs();refs[-1]['split']='dev'
        with self.assertRaises(ValueError):make_requests(refs)

    def test_test_selection_full_denominator_and_identity(self):
        _,requests=make_requests(self.refs());candidates={}
        for sha,score in [('a'*64,-1.),('b'*64,-.5)]:
            candidates[sha]=[dict(sample_id=r['sample_id'],input_sha256=r['reference_sha256'],prompt=r['prompt'],seed=r['seed'],
                category=r['category'],checkpoint_sha256=sha,ImageReward_compact=score,CLIP_T=.3) for r in requests]
        result=select_checkpoint(requests,candidates)
        self.assertEqual(result['selected']['checkpoint_sha256'],'b'*64)
        self.assertTrue(result['TEST_used_for_model_selection']);self.assertFalse(result['independent_generalization_claim'])
        for mutation in ['subset','prompt','pt']:
            c=copy.deepcopy(candidates)
            if mutation=='subset':c['b'*64].pop()
            if mutation=='prompt':c['b'*64][0]['prompt']='Changed prompt'
            if mutation=='pt':c['b'*64][0]['checkpoint_sha256']='a'*64
            with self.assertRaises(ValueError):select_checkpoint(requests,c)


if __name__=='__main__':unittest.main()
