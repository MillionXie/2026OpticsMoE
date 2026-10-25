import json,unittest
from pathlib import Path
from .refine_LG_C_readout import validate
from .distill_margin_teacher import declared_source_budget,objective_pool_weight


class LGCTests(unittest.TestCase):
    def test_old_function_preservation_never_changes_original_TRAIN_mass(self):
        c=json.loads(Path(__file__).with_name('config_LG_B_new_task_transfer_20261025.json').read_text())
        self.assertEqual(objective_pool_weight('eurosat',0,c),1)
        self.assertEqual(objective_pool_weight('clevr',0,c),1)
        self.assertEqual(objective_pool_weight('clevr',1,c),1)
        self.assertEqual(objective_pool_weight('eurosat',1,c),0)
    def test_all_TEST_source_student_requires_explicit_disclosure(self):
        c={'all_TEST_source_teacher':True};p={'all_TEST_sources_used':True,'independent_EVAL_sources_remaining':{'clevr':0,'eurosat':0}}
        self.assertEqual(declared_source_budget(2,'moe',c,p),{'eurosat':1000,'clevr':1000})
        for q in ({},dict(p,all_TEST_sources_used=False)):
            with self.assertRaises(ValueError):declared_source_budget(2,'moe',c,q)
    def test_fixed_budget_and_original_head_only(self):
        cfg=json.loads(Path(__file__).with_name('config_LG_C_readout_20261025.json').read_text());validate(cfg)
        warm=json.loads(Path(__file__).with_name('config_LG_C_strong_parent_readout_20261025.json').read_text());validate(warm)
        self.assertTrue(warm['TRAIN_teacher_correct_only']);self.assertEqual(warm['TRAIN_counts']['clevr'],300)
        for key,value in [('epochs',10000),('TEST_gradients',True),('TRAIN_counts',{'speech_binary':1000})]:
            with self.assertRaises(ValueError):validate(dict(cfg,**{key:value}))


if __name__=='__main__':unittest.main()
