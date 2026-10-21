import ast
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'lab_caltech20_dvp_closedloop_20261021.py'

class ClosedLoopContract(unittest.TestCase):
    def test_gates_and_fixed_hardware(self):
        tree=ast.parse(SOURCE.read_text());env={}
        nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('SPECS','HARDWARE') for t in n.targets)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<contract>','exec'),env)
        self.assertEqual([env['SPECS'][d]['target'] for d in (4,6)],[314,322])
        self.assertEqual([env['SPECS'][d]['final'] for d in (4,6)],[342,350])
        self.assertEqual([env['SPECS'][d]['stage'] for d in (4,6)],['2','0'])
        text=SOURCE.read_text()
        for forbidden in ('settings(exposure','coordinates.write','lut.write','torch.optim.SGD'):
            self.assertNotIn(forbidden,text)
        for node in ast.walk(tree):
            if isinstance(node,ast.keyword) and node.arg=='TEST_gradient':
                self.assertIsInstance(node.value,ast.Constant);self.assertIs(node.value.value,False)
        for token in ("cfg['exposure_us']==60006.","cfg['wait_ms']==240","cfg['orientation']=='flip_v'",'assert count==2011','assert checked==2011','assert white<2'):
            if token=='assert white<2':self.assertIn('if white>=2:',text)
            else:self.assertIn(token,text)

    def test_full_TRAIN_gradient_then_fresh_capture(self):
        tree=ast.parse(SOURCE.read_text());fn={n.name:ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertIn("put_fields(work, data, fields, 'train'",fn['propose'])
        self.assertIn('used == list(range(1600))',fn['propose'])
        self.assertNotIn('opt.step()',fn['propose'])
        self.assertIn('simulated - simulated.detach()',fn['physical_adjoint'])
        self.assertIn("for role in ('train', 'test')",fn['acquire'])
        self.assertNotIn('Bench(',fn['audit_capture'])
        self.assertIn('newfields = acquire(',fn['run'])
        self.assertIn('actual_gate_passed=passed',fn['run'])
        self.assertNotIn('capture_worker(',fn['run'])

    def test_finite_profile(self):
        p=json.loads((ROOT/'lab_caltech20_DVP_closedloop_profile_20261021.json').read_text())
        self.assertEqual(p['rounds_per_depth'],2)
        self.assertEqual(p['phase_step_max_radians'],.04)
        self.assertEqual(p['suffix_epochs'],12)
        self.assertEqual(p['suffix_fit_seconds'],600)

if __name__=='__main__':unittest.main()
