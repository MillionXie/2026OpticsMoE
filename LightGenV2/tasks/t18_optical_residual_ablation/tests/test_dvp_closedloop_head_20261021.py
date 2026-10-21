import ast
from pathlib import Path
import unittest

class FullTRAINRowContract(unittest.TestCase):
    def test_no_small_selected_training_set_or_hardware(self):
        text=(Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_closedloop_head_20261021.py').read_text();ast.parse(text)
        for banned in ('.cuda(','.Bench(',"'test'].backward",'randperm(', 'batch_indices('):self.assertNotIn(banned,text)
        for required in ('full_TRAIN_each_update=True','gradient_batch=1600',"assert np.all(sample>0)","gradient=x.T@",'all_nonhead_frozen=True',"folder/'independent_round_audit.json'",'TEST_gradient=False'):
            self.assertIn(required,text)
    def test_primal_is_small_original_row_system(self):
        text=(Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_closedloop_head_20261021.py').read_text()
        self.assertIn('torch.eye(x.shape[1],dtype=x.dtype)',text);self.assertIn('published_curvature_equivalence',text)

if __name__=='__main__':unittest.main()
