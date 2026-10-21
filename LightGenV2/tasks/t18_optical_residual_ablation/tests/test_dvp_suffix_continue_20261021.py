from pathlib import Path
import ast
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_suffix_continue_20261021.py'


class SuffixContinuation(unittest.TestCase):
    def test_full_TRAIN_existing_suffix_only(self):
        text=SOURCE.read_text();ast.parse(text)
        for token in ("raw['selected_epoch']==raw['actual_epochs']==12", "['accuracy']<.85",
            'suffix_epochs=24,suffix_fit_seconds=600','loop.refine(',
            'current_E2_mask_and_all_upstream_frozen=True', 'new_CCD=0',
            'all1600TRAIN_and411TEST_original_CPU_verified=True','TEST_gradient=False'):
            self.assertIn(token,text)
        for bad in ('.Bench(','.capture(','.settings(','.backward(','.step('):self.assertNotIn(bad,text)
    def test_no_success_without_actual_gate(self):
        text=SOURCE.read_text();self.assertIn('actual_gate_passed=correct>=314',text)
        self.assertIn('next_layer_started=False',text)


if __name__=='__main__':unittest.main()
