import copy
import unittest
from LightGenV2.tasks.t12_text_to_image.scene_paper_target_candidates import checked_train_rows,selected_rows
from LightGenV2.tasks.t12_text_to_image.scene_paper_prompt_protocol import condition,PROFILE,PARENT_PAIRS
from LightGenV2.tasks.t12_text_to_image.scene_generation_data import request_seed


class TargetRoleTest(unittest.TestCase):
    def fixture(self):
        report=dict(schema='scene_paper_request_plan_v1',complete=True,prompt_profile=PROFILE,source_pairs_sha256=PARENT_PAIRS,TEST_opened=False,role_disjoint=True,targets_created=False,eligible_for_training=False)
        rows=[]
        for k in range(44):
            rid='source'+str(k);digest=format(k,'064x')
            for i in range(3):
                d=condition('lamp'if k<20 else'pillow',i)
                for j in [0,1]:rows.append(dict(sample_id=rid+'-paper-scene'+str(i)+'-draw'+str(j),reference='references/'+rid+'.png',reference_id=rid,reference_sha256=digest,seed=request_seed(digest,j),**{x:d[x]for x in ['category','condition_id','prompt','native_caption','reward_caption','prompt_profile']}))
        dev=[dict(reference_sha256='f'*64)]*48
        return report,rows,dev

    def test_complete_counterfactuals(self):
        r,rows,dev=self.fixture();self.assertEqual(len(checked_train_rows(r,rows,dev)),264)
        first=selected_rows(rows,[0]);self.assertEqual(len(first),132);self.assertEqual(len({r['reference_id']for r in first}),44)
        self.assertTrue(all(r['sample_id'].endswith('-draw0')for r in first))
        with self.assertRaises(ValueError):selected_rows(rows,[0,0])

    def test_role_prompt_and_seed_guards(self):
        for field,value in [('reference_sha256','f'*64),('seed',-1),('prompt','A changed instruction.')]:
            r,rows,dev=self.fixture();rows[0][field]=value
            with self.assertRaises(ValueError):checked_train_rows(r,rows,dev)
        r,rows,dev=self.fixture();rows[0]['target']='hidden.png'
        with self.assertRaises(ValueError):checked_train_rows(r,rows,dev)


if __name__=='__main__':unittest.main()
