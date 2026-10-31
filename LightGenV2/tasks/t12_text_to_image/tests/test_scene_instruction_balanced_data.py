import tempfile
from pathlib import Path
import unittest
from ..scene_instruction_balanced_data import checked_pool,curate,checked_dev_labels
from ..scene_instruction_data import condition,write,sha
from ..scene_omnigen_peer import REVISION
from ..scene_generation_data import request_seed


class CandidateAdmission(unittest.TestCase):
    def test_DEV_relation_recheck_excludes_wrong_furniture_without_dropping_inputs(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);(p/'ok.png').write_bytes(b'ok');(p/'wrong.png').write_bytes(b'chair without arms')
            rows=[dict(sample_id=k,split='dev',target=k+'.png',target_sha256=sha(p/(k+'.png')))for k in ['ok','wrong']]
            write(p/'pairs.json',rows);review=dict(pairs_sha256=sha(p/'pairs.json'),all_DEV_labels_reviewed=True,rows=[dict(sample_id=r['sample_id'],target_sha256=r['target_sha256'],status='usable_DEV_label'if r['sample_id']=='ok'else'rejected',accepted_for_validation=r['sample_id']=='ok')for r in rows]);write(p/'review.json',review)
            self.assertEqual([r['sample_id']for r in checked_dev_labels(p,p/'review.json')],['ok'])
            with self.assertRaises(ValueError):checked_dev_labels(p,None)
            review['rows'].pop();write(p/'review.json',review)
            with self.assertRaises(ValueError):checked_dev_labels(p,p/'review.json')
    def test_missing_condition_never_emits_training_pairs_or_approval(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);plan=p/'plan';plan.mkdir();rows=[]
            for i in range(44):
                for code in 'BDW':
                    for seed in [0,1]:rows.append(dict(condition('lamp',code),reference_id=str(i),split='train',reference='not_opened.png',reference_sha256=str(i),seed=seed))
            write(plan/'requests.json',rows);write(plan/'report.json',dict(schema='scene_balanced_visible_anchor_plan_v1',request_sha256=sha(plan/'requests.json')))
            output=p/'readiness';curate(plan,p/'no_DEV_access_needed',[],output)
            import json
            report=json.loads((output/'readiness.json').read_text())
            self.assertFalse(report['ready_for_training']);self.assertEqual(report['missing_reference_conditions'],132)
            self.assertFalse((output/'pairs.json').exists());self.assertFalse((output/'visual_review.json').exists())
    def test_uncertain_not_accepted_and_pixels_roles_and_prompts_are_bound(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);(p/'ref.png').write_bytes(b'input');(p/'target.png').write_bytes(b'target')
            row=dict(condition('lamp','B'),sample_id='one',reference_id='r',split='train',reference='ref.png',reference_sha256=sha(p/'ref.png'),target='target.png',target_sha256=sha(p/'target.png'),source_product_id='p',source_RGBA_sha256='rgba',seed_index=0,seed=request_seed(sha(p/'ref.png'),0))
            write(p/'candidates.json',[row]);write(p/'report.json',dict(complete=True,prompt_profile=row['prompt_profile'],DEV_or_TEST_forwarded=False,candidate_manifest_sha256=sha(p/'candidates.json')));write(p/'model_assets.json',dict(official_revision=REVISION))
            review=dict(candidate_manifest_sha256=sha(p/'candidates.json'),rows=[dict(sample_id='one',target_sha256=row['target_sha256'],status='uncertain',accepted_for_training=False)]);write(p/'review.json',review)
            self.assertEqual(checked_pool(p,p/'review.json',{'r':row}),[])
            review['rows'][0].update(status='usable_TRAIN_example',accepted_for_training=True);write(p/'review.json',review)
            self.assertEqual(len(checked_pool(p,p/'review.json',{'r':row})),1)
            (p/'target.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):checked_pool(p,p/'review.json',{'r':row})
            (p/'target.png').write_bytes(b'target');row['split']='test';write(p/'candidates.json',[row]);write(p/'report.json',dict(complete=True,prompt_profile=row['prompt_profile'],DEV_or_TEST_forwarded=False,candidate_manifest_sha256=sha(p/'candidates.json')));review['candidate_manifest_sha256']=sha(p/'candidates.json');write(p/'review.json',review)
            with self.assertRaises(ValueError):checked_pool(p,p/'review.json',{'r':row})


if __name__=='__main__':unittest.main()
