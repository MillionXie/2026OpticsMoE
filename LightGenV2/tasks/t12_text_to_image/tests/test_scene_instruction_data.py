import copy
import unittest
import tempfile
import json
from pathlib import Path
from ..scene_instruction_data import LAMP,PILLOW,condition,checked_annotations,PROFILE
from ..scene_instruction_data import SAFE,sha,write
from ..scene_instruction_render import checked_revision


class VisibleFacts(unittest.TestCase):
    def test_two_sentence_concrete_contract(self):
        prompts=[]
        for category,codes in [('lamp',LAMP),('pillow',PILLOW)]:
            for code in codes:
                r=condition(category,code);p=r['prompt'];prompts.append(p)
                self.assertEqual(p.count('.'),2);self.assertLessEqual(len(p.split()),27)
                self.assertNotIn('reference',p);self.assertNotIn('fine-tun',p)
                self.assertEqual(p,r['native_caption']);self.assertEqual(p,r['reward_caption'])
                self.assertGreaterEqual(len(r['required_visible_facts']),3)
                self.assertEqual(r['prompt_profile'],PROFILE)
        self.assertEqual(len(prompts),len(set(prompts)))
    def fixture(self):
        rows=[dict(sample_id='a',category='lamp',target_sha256='abc')]
        review=dict(schema='scene_visible_facts_review_v1',all_candidates_reviewed=True,
                    annotations=[dict(sample_id='a',target_sha256='abc',code='B')])
        return rows,review
    def test_annotation_bound_to_pixels(self):
        rows,review=self.fixture();self.assertIn('a',checked_annotations(rows,review))
        review['annotations'][0]['target_sha256']='different'
        with self.assertRaises(ValueError):checked_annotations(rows,review)
    def test_missing_or_duplicate_review_not_complete(self):
        rows,review=self.fixture();review['annotations']=[]
        with self.assertRaises(ValueError):checked_annotations(rows,review)
        rows,review=self.fixture();review['annotations']*=2
        with self.assertRaises(ValueError):checked_annotations(rows,review)
    def test_wrong_category_and_unknown_anchor_rejected(self):
        with self.assertRaises(ValueError):condition('pillow','B')
        rows,review=self.fixture();review['annotations'][0]['code']='pretend'
        with self.assertRaises(ValueError):checked_annotations(rows,review)
    def test_rejection_is_not_auto_approval(self):
        rows,review=self.fixture();review['annotations'][0]['code']='X'
        self.assertEqual(checked_annotations(rows,review)['a']['code'],'X')
    def revision_fixture(self,root):
        data=root/'data';data.mkdir();folder=root/'revision';folder.mkdir();original=[];actual=[]
        for i in range(16):
            category='lamp'if i<8 else'pillow';rid=str(i);(data/(rid+'.png')).write_bytes(('pixels'+rid).encode())
            row=dict(sample_id=rid,reference=rid+'.png',reference_id=rid,reference_sha256=sha(data/(rid+'.png')),
                category=category,condition_id='0',prompt='old',native_caption='old',reward_caption='old',prompt_profile='old',seed=i)
            original.append(row)
        for row in sorted(original,key=lambda r:r['reference_id']):
            for code in (['B','D','W']if row['category']=='lamp'else['S','A','D']):
                d=condition(row['category'],code);r={k:row[k]for k in SAFE};r.update({k:d[k]for k in ['condition_id','prompt','native_caption','reward_caption','prompt_profile','category']})
                r['sample_id']=row['reference_id']+'-visible-'+code+'-draw2';actual.append(r)
        write(folder/'inference_inputs.json',actual)
        write(folder/'report.json',dict(complete=True,schema='scene_visible_anchor_dataset_v1',prompt_profile=PROFILE,
            TEST_opened=False,role_disjoint=True,source_pairs_sha256='old',pairs_sha256='revised',request_sha256=sha(folder/'inference_inputs.json')))
        return data,folder,original,actual
    def test_exact_target_free_all48_cohort(self):
        with tempfile.TemporaryDirectory()as temp:
            data,folder,original,_=self.revision_fixture(Path(temp));rows,protocol=checked_revision(folder,data,{'pair_sha256':'old'},original)
            self.assertEqual(len(rows),48);self.assertFalse(protocol['TEST_opened'])
            self.assertTrue(all(set(r)==set(SAFE)for r in rows))
            with self.assertRaises(ValueError):checked_revision(folder,data,{'pair_sha256':'unrelated'},original)
    def test_no_easy_subset_or_target_fields(self):
        with tempfile.TemporaryDirectory()as temp:
            data,folder,original,rows=self.revision_fixture(Path(temp));rows[0]['target']='forbidden';write(folder/'inference_inputs.json',rows)
            report=json.loads((folder/'report.json').read_text());report['request_sha256']=sha(folder/'inference_inputs.json');write(folder/'report.json',report)
            with self.assertRaises(ValueError):checked_revision(folder,data,{'pair_sha256':'old'},original)
    def test_public_prompt_cannot_be_swapped_after_freeze(self):
        with tempfile.TemporaryDirectory()as temp:
            data,folder,original,rows=self.revision_fixture(Path(temp));rows[0]['prompt']='Generate whatever looks easy.';write(folder/'inference_inputs.json',rows)
            report=json.loads((folder/'report.json').read_text());report['request_sha256']=sha(folder/'inference_inputs.json');write(folder/'report.json',report)
            with self.assertRaises(ValueError):checked_revision(folder,data,{'pair_sha256':'old'},original)


if __name__=='__main__':unittest.main()
