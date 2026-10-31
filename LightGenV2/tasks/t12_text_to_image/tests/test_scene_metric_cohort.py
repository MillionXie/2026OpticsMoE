import unittest
from ..scene_metric_cohort import expanded_identity


class MetricCohort(unittest.TestCase):
    def report(self):return dict(expanded_DEV_protocol=dict(schema='scene_visible_anchor_DEV_preflight_v1',prompt_profile='two_sentence_visible_anchor_v5',contexts=48,reference_views=16,TEST_opened=False,balanced_counterfactual_training_data=True,request_sha256='a'*64))
    def rows(self):return[dict(reference_id=str(i),category='lamp'if i<8 else'pillow',condition_id=str(j),input_sha256=str(i),seed=i)for i in range(16)for j in range(3)]
    def test_full_new_cohort_and_old_identity(self):
        self.assertEqual(expanded_identity(self.report(),self.rows()),(48,('visible_anchor_v5','a'*64)))
        self.assertEqual(expanded_identity(dict(expanded_DEV_protocol=dict(manifest_sha256='b'*64))),(128,('historical_expanded','b'*64)))
    def test_no_subset_or_seed_change(self):
        for kind in ['subset','seed','condition']:
            rows=self.rows()
            if kind=='subset':rows.pop()
            if kind=='seed':rows[1]['seed']=999
            if kind=='condition':rows[1]['condition_id']='0'
            with self.assertRaises(ValueError):expanded_identity(self.report(),rows)
    def test_wrong_profile_or_test_rejected(self):
        for key,value in [('prompt_profile','old'),('TEST_opened',True),('balanced_counterfactual_training_data',False),('contexts',128)]:
            r=self.report();r['expanded_DEV_protocol'][key]=value
            with self.assertRaises(ValueError):expanded_identity(r)
    def test_informative_prompts_are_a_separate_zero_shot_cohort(self):
        r=self.report();p=r['expanded_DEV_protocol'];p.update(schema='scene_paper_scene_DEV_preflight_v1',prompt_profile='two_sentence_product_scene_v7',prior_model_on_new_prompts_not_retrained_model=True,model_trained_on_revised_dataset=False,old_results_not_relabelled=True,balanced_counterfactual_training_data=False)
        self.assertEqual(expanded_identity(r,self.rows()),(48,('paper_scene_v7_zero_shot','a'*64)))
        p['model_trained_on_revised_dataset']=True
        with self.assertRaises(ValueError):expanded_identity(r)


if __name__=='__main__':unittest.main()
