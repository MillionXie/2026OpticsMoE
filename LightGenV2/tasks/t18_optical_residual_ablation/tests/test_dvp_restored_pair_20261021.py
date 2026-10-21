import ast
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parents[1]
SOURCE=HERE/'lab_caltech20_dvp_restored_pair_20261021.py'


class RestoredPair(unittest.TestCase):
    def test_scope_and_budget(self):
        cfg=json.loads((HERE/'lab_caltech20_DVP_restored_pair_profile_20261021.json').read_text())
        self.assertEqual(cfg['capture_order'],[4,6]);self.assertEqual(cfg['TRAIN_response_views'],3200)
        self.assertEqual(cfg['maximum_epochs'],12);self.assertEqual(cfg['maximum_fit_seconds'],900)
        self.assertFalse(cfg['TEST_gradient']);self.assertFalse(cfg['automatic_retry'])
        self.assertEqual(sum(cfg['original_TRAIN_counts']),3141)
        self.assertNotIn('pythonw.exe',cfg['desktop_graphics_process_allowlist'])
    def test_all_stimuli_and_frozen_masks(self):
        text=SOURCE.read_text();ast.parse(text)
        for token in ('assert checked==2011',"a.get('input_SHA256',a.get('BMP_SHA256'))==b['BMP_SHA256']",
            'assert not allowed&loop.mask_keys',"loop.validate_state(m,model,before,allowed)",
            'teacher.grad is None','used==1600','new_mask_proposal=False','TEST_teacher=False'):
            self.assertIn(token,text)
        for forbidden in ('.propose(','.physical_adjoint(','.settings(','.rotate(','.warpPerspective('):
            self.assertNotIn(forbidden,text)
    def test_real_pair_train_loss_only(self):
        tree=ast.parse(SOURCE.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='paired_loss')
        names={x.id for x in ast.walk(fn) if isinstance(x,ast.Name)}
        self.assertFalse(any('test' in x.lower() for x in names))
        self.assertIn('teacher_correct',names);self.assertIn('.detach()',ast.unparse(fn))
    def test_single_sdk_and_cpu_independent_audit(self):
        text=SOURCE.read_text()
        self.assertIn('ThreadPoolExecutor(max_workers=1)',text)
        self.assertIn("env=dict(os.environ,CUDA_VISIBLE_DEVICES='')",text)
        self.assertIn('all_original_CPU_TRAIN_and_TEST_verified=True',text)
        self.assertIn('next_layer_started=False',text)


if __name__=='__main__':unittest.main()
