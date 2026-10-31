import copy
import unittest
from LightGenV2.tasks.t12_text_to_image.scene_expanded_paired_data import checked_candidate_rows,checked_fast_coverage
from LightGenV2.tasks.t12_text_to_image.scene_paper_expanded_data import make_requests


class FastAdmission(unittest.TestCase):
    def inputs(self):
        refs=[dict(reference_id=f'{c}-{i}',reference=f'{c}-{i}.png',reference_sha256=f'{c}-{i}',split='train',category=c,
                   source_product_id=f'{c}-product-{i%22}',source_family_id=f'{c}-family-{i%22}')for c,n in [('lamp',36),('pillow',40)]for i in range(n)]
        refs += [dict(reference_id=f'test-{c}-{i}',reference=f'test-{c}-{i}.png',reference_sha256=f'test-{c}-{i}',split='test',category=c,
                      source_product_id=f'test-{c}-{i}',source_family_id=f'test-{c}-{i}')for c in ['lamp','pillow']for i in range(13)]
        rows,_=make_requests(refs)
        return refs,rows

    def test_prefix_is_explicit_and_cannot_change_prompts(self):
        _,expected=self.inputs();rows=expected[:700];report=dict(partial_TRAIN_prefix=True,original_TRAIN_request_count=1520,candidate_count=700)
        checked_candidate_rows(report,rows,expected,True)
        with self.assertRaises(ValueError):checked_candidate_rows(report,rows,expected)
        wrong=copy.deepcopy(rows);wrong[0]['prompt']='relabel old pixels'
        with self.assertRaises(ValueError):checked_candidate_rows(report,wrong,expected,True)
        with self.assertRaises(ValueError):checked_candidate_rows(report,list(reversed(rows)),expected,True)

    def test_usable_coverage_cannot_be_faked_with_duplicates(self):
        refs,rows=self.inputs();selected=[r for r in rows if int(r['condition_id'])<8]
        # Supply every template group across at least one source, not just eight easy templates.
        selected+= [r for r in rows if int(r['condition_id'])>=8 and r['reference_id'] in ['lamp-0','pillow-0']]
        result=checked_fast_coverage(selected,refs);self.assertEqual(result['TRAIN_views'],76);self.assertEqual(result['TRAIN_products'],44)
        self.assertEqual(result['template_groups'],40)
        with self.assertRaises(ValueError):checked_fast_coverage(selected+[selected[0]],refs)
        with self.assertRaises(ValueError):checked_fast_coverage(selected[:599],refs)
        with self.assertRaises(ValueError):checked_fast_coverage([r for r in selected if r['reference_id']!='pillow-39'],refs)
        with self.assertRaises(ValueError):checked_fast_coverage([r for r in selected if r['condition_id']!='19'],refs)


if __name__=='__main__':unittest.main()
