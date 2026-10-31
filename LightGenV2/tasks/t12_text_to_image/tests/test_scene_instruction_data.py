import copy
import unittest
import tempfile
import json
from pathlib import Path
from ..scene_instruction_data import LAMP,PILLOW,condition,checked_annotations,PROFILE
from ..scene_instruction_data import SAFE,sha,write,checked_parent_transfer
from ..scene_instruction_render import checked_revision,checked_balanced_input_identity


class VisibleFacts(unittest.TestCase):
    def test_balanced_raw_identity_is_full_input_only_and_bound_to_actual_training(self):
        report=dict(complete=True,cache_sha256='actual_cache',DEV_input_identity_sha256='input_only')
        state=dict(cache_sha256='actual_cache',Qwen_input_side=256)
        identity=dict(contains_targets_or_masks=False,Qwen_input_side=256,features={str(i):{}for i in range(48)})
        self.assertTrue(checked_balanced_input_identity(report,state,identity,'input_only'))
        with self.assertRaises(ValueError):checked_balanced_input_identity(report,dict(state,cache_sha256='other_cache'),identity,'input_only')
        with self.assertRaises(ValueError):checked_balanced_input_identity(report,state,dict(identity,contains_targets_or_masks=True),'input_only')
        with self.assertRaises(ValueError):checked_balanced_input_identity(report,state,dict(identity,features={}), 'input_only')
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
    def test_balanced_dataset_cannot_render_with_old_model_or_missing_scene_coverage(self):
        with tempfile.TemporaryDirectory()as temp:
            data,folder,original,_=self.revision_fixture(Path(temp));report=json.loads((folder/'report.json').read_text())
            from ..scene_instruction_native_prior import visible_caption
            actual=json.loads((folder/'inference_inputs.json').read_text())
            for row in actual:
                code=({'0':'B','1':'D','2':'W'}if row['category']=='lamp'else{'0':'S','1':'A','2':'D'})[row['condition_id']];row['native_caption']=visible_caption(row['category'],code)
            write(folder/'inference_inputs.json',actual)
            report.update(schema='scene_balanced_visible_anchor_dataset_v1',eligible_for_training=True,all_reference_conditions_covered=True,TRAIN_products=44,TRAIN_reference_views=44,TRAIN_condition_groups=132,native_caption_style='public_fact_equivalent_description_v1',request_sha256=sha(folder/'inference_inputs.json'));write(folder/'report.json',report)
            rows,protocol=checked_revision(folder,data,{'pair_sha256':'revised'},original);self.assertEqual(len(rows),48);self.assertTrue(protocol['balanced_counterfactual_training_data'])
            with self.assertRaises(ValueError):checked_revision(folder,data,{'pair_sha256':'old'},original)
            report['all_reference_conditions_covered']=False;write(folder/'report.json',report)
            with self.assertRaises(ValueError):checked_revision(folder,data,{'pair_sha256':'revised'},original)
    def test_public_prompt_cannot_be_swapped_after_freeze(self):
        with tempfile.TemporaryDirectory()as temp:
            data,folder,original,rows=self.revision_fixture(Path(temp));rows[0]['prompt']='Generate whatever looks easy.';write(folder/'inference_inputs.json',rows)
            report=json.loads((folder/'report.json').read_text());report['request_sha256']=sha(folder/'inference_inputs.json');write(folder/'report.json',report)
            with self.assertRaises(ValueError):checked_revision(folder,data,{'pair_sha256':'old'},original)
    def test_reviewed_parent_transfer_keeps_pixels_and_roles(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            prior=dict(sample_id='a',split='train',source_product_id='p',source_RGBA_sha256='rgba',reference_sha256='ref',target_sha256='target',prompt='original')
            write(old/'pairs.json',[prior]);oldsha=sha(old/'pairs.json')
            revised=dict(prior,original_sample_id='a',original_prompt='original',prompt='new factual request')
            write(new/'pairs.json',[revised]);newsha=sha(new/'pairs.json')
            write(old/'report.json',dict(complete=True,role_disjoint=True,TEST_opened=False,pairs_sha256=oldsha))
            report=dict(complete=True,role_disjoint=True,TEST_opened=False,schema='scene_visible_anchor_dataset_v1',prompt_profile=PROFILE,source_pairs_sha256=oldsha,pairs_sha256=newsha,review_sha256='review')
            write(new/'report.json',report)
            self.assertTrue(checked_parent_transfer(new,old,oldsha,newsha)['old_pixels_and_roles_preserved'])
            revised['target_sha256']='different_target';write(new/'pairs.json',[revised]);newsha=sha(new/'pairs.json');report['pairs_sha256']=newsha;write(new/'report.json',report)
            with self.assertRaises(ValueError):checked_parent_transfer(new,old,oldsha,newsha)


if __name__=='__main__':unittest.main()
