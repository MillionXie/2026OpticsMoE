import ast
from pathlib import Path
import unittest

class IndependentClosedLoopAudit(unittest.TestCase):
    def test_CPU_only_and_no_mutations_or_automatic_admission(self):
        path=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_closedloop_audit_20261021.py';text=path.read_text();ast.parse(text)
        for banned in ('m.Bench(','.cuda(','.backward(','.step(','.load_state_dict(', 'torch.save('):self.assertNotIn(banned,text)
        for required in ("os.environ.get('CUDA_VISIBLE_DEVICES')==''",'next_layer_permission=False','original_replay(current,data,fields,role)',"validate_state(m,current,before,allowed)","np.array_equal(prob.argmax(1),bank['probabilities'].argmax(1))",'SDK_opened=False','TEST_gradient=False'):
            self.assertIn(required,text)
    def test_no_core_or_training_code_touched(self):
        path=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_closedloop_audit_20261021.py';tree=ast.parse(path.read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='save' for n in calls),1)
        self.assertIn('independent_round_audit.json',path.read_text())

if __name__=='__main__':unittest.main()
