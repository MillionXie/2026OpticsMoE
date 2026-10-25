import json,unittest
from pathlib import Path
import torch
from .distill_margin_teacher import development_bands,declared_source_budget,source_pool_objective,whole_stage_band_key,equal_task_retention_KL,correct_function_rows
from .fit_stage_balanced_teacher import validate_profile


class FullSourceFollowupTests(unittest.TestCase):
    def test_LG_C_full_scope_keeps_own_five_point_bands(self):
        cfg=json.loads(Path(__file__).with_name('config_LG_C_fullsource_mean_20261026.json').read_text())
        bands=development_bands(3,'moe',cfg)
        self.assertEqual(bands,dict(eurosat=(706,756),clevr=(696,746),speech_binary=(715,765)))
        protocol=dict(all_TEST_sources_used=True,independent_EVAL_sources_remaining={n:0 for n in bands})
        self.assertEqual(declared_source_budget(3,'moe',cfg,protocol),{n:1000 for n in bands})
        teacher=json.loads(Path(__file__).with_name('config_stage_LG_C_allTEST_teacher_20261026.json').read_text());validate_profile(teacher)
        self.assertTrue(teacher['all_TEST_sources_authorized'])
    def test_wrong_derivative_function_is_not_frozen_as_knowledge(self):
        target=torch.tensor([[4.,0.],[0.,4.],[2.,1.]],requires_grad=True);labels=torch.tensor([0,0,1])
        self.assertEqual(correct_function_rows(target,labels).tolist(),[True,False,False])
        for name,stage,arch in [('LG_B',2,'moe'),('D2NN_D',4,'d2nn')]:
            cfg=json.loads(Path(__file__).with_name('config_'+name+'_correct_source_retention_20261026.json').read_text())
            self.assertTrue(cfg['derivative_function_correct_only']);self.assertEqual(len(development_bands(stage,arch,cfg)),stage)
    def test_D_CLEVR_focus_keeps_other_task_functions_without_changing_bands(self):
        cfg=json.loads(Path(__file__).with_name('config_D2NN_D_CLEVR_source_retention_20261026.json').read_text())
        self.assertEqual(development_bands(4,'d2nn',cfg)['clevr'],(517,617))
        self.assertEqual(set(cfg['TRAIN_function_tasks']),set(cfg['development_bands'])-{'clevr'})
        self.assertEqual(cfg['source_pair_tasks'],['clevr'])
    def test_LG_C_partial_scope_does_not_inherit_D2NN_relaxation(self):
        cfg=json.loads(Path(__file__).with_name('config_LG_C_partial500_margin_20261026.json').read_text())
        self.assertEqual(development_bands(3,'moe',cfg)['speech_binary'],(715,765))
        self.assertEqual(declared_source_budget(3,'moe',cfg,{}),dict(eurosat=500,clevr=500,speech_binary=500))
        teacher=json.loads(Path(__file__).with_name('config_stage_LG_C500_teacher_20261026.json').read_text());validate_profile(teacher)
        self.assertNotIn('all_TEST_sources_authorized',teacher)
    def test_source_retention_is_teacher_frozen_and_equal_task(self):
        import numpy as np
        target=torch.randn(5,10,dtype=torch.float64,requires_grad=True);current=target.detach().clone().requires_grad_();tasks=np.array(['a']*4+['b'])
        loss=equal_task_retention_KL(current,target,tasks,['a','b']);self.assertLess(abs(float(loss.detach())),1e-12)
        loss.backward();self.assertIsNone(target.grad);self.assertTrue(torch.isfinite(current.grad).all())
        with self.assertRaises(ValueError):equal_task_retention_KL(current,target,tasks,['c'])
        for name,stage,arch in [('LG_B',2,'moe'),('D2NN_C',3,'d2nn'),('D2NN_D',4,'d2nn')]:
            cfg=json.loads(Path(__file__).with_name('config_'+name+'_source_retention_20261026.json').read_text());self.assertEqual(len(development_bands(stage,arch,cfg)),stage)
            self.assertTrue(cfg['warm_student_development']);self.assertEqual(cfg['derivative_function_weight'],2.)
    def test_new_scopes_keep_legacy_unchanged(self):
        legacy=development_bands(3,'d2nn',{})
        self.assertEqual(legacy['clevr'],(631,681))
        for name,stage,arch in [('LG_B',2,'moe'),('D2NN_C',3,'d2nn'),('D2NN_D',4,'d2nn')]:
            cfg=json.loads(Path(__file__).with_name('config_'+name+'_fullsource_mean_20261026.json').read_text())
            bands=development_bands(stage,arch,cfg)
            protocol=dict(all_TEST_sources_used=True,independent_EVAL_sources_remaining={n:0 for n in bands})
            self.assertEqual(declared_source_budget(stage,arch,cfg,protocol),{n:1000 for n in bands})
            self.assertLessEqual(cfg['maximum_seconds'],360)
            with self.assertRaises(ValueError):declared_source_budget(stage,arch,cfg,dict(protocol,all_TEST_sources_used=False))
            with self.assertRaises(ValueError):development_bands(stage,arch,dict(cfg,development_bands={'eurosat':[0,1000]}))
    def test_new_C_teacher_scope_is_exact(self):
        cfg=json.loads(Path(__file__).with_name('config_stage_D2NN_C_allTEST_teacher_20261026.json').read_text())
        validate_profile(cfg)
        with self.assertRaises(ValueError):validate_profile(dict(cfg,SDK=True))
        self.assertEqual(sum(cfg['sources_per_task'].values()),3000)
    def test_retained_function_uses_TRAIN_and_same_original_scope(self):
        for name,stage,arch in [('LG_B',2,'moe'),('D2NN_C',3,'d2nn'),('D2NN_D',4,'d2nn')]:
            cfg=json.loads(Path(__file__).with_name('config_'+name+'_retained_function_20261026.json').read_text())
            self.assertEqual(len(development_bands(stage,arch,cfg)),stage)
            self.assertTrue(set(cfg['TRAIN_function_tasks'])<=set(cfg['development_bands']))
            for n in cfg['TRAIN_function_tasks']:self.assertEqual(cfg['derivative_task_weights'][n],0.)
            self.assertEqual(cfg['TRAIN_function_weight'],2.)
    def test_mean_objective_does_not_add_accuracy_ceiling_gradient(self):
        values=torch.tensor([1.,2.,9.],requires_grad=True);threshold=torch.tensor(2.,requires_grad=True)
        loss=source_pool_objective(values,threshold,dict(mean_source_objective=True));loss.backward()
        torch.testing.assert_close(values.grad,torch.ones(3)/3);self.assertIsNone(threshold.grad)
        scores=dict(eurosat=dict(n=1000,correct=652),clevr=dict(n=1000,correct=645),speech_binary=dict(n=1000,correct=618))
        bands=dict(eurosat=(462,562),clevr=(581,681),speech_binary=(422,522))
        self.assertEqual(whole_stage_band_key(scores,bands),(-96,-186))


if __name__=='__main__':unittest.main()
