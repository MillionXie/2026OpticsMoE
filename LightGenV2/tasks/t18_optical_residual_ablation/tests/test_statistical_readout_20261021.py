"""Static boundaries; numerical Torch selftest is run on the lab host."""
import ast,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]

class StatisticalReadout(unittest.TestCase):
    def test_profile(self):
        c=json.loads((HERE/'lab_caltech20_DVP_statistical_readout_profile_20261021.json').read_text())
        self.assertEqual(c['targets'],{'4':288,'6':322})
        self.assertEqual(sum(c['original_TRAIN_counts']),3141)
        self.assertEqual(c['parents']['4']['correct'],280)
        self.assertEqual(c['parents']['6']['correct'],285)
        self.assertLessEqual(c['fit_seconds'],120)

    def test_scalar_fit_has_no_test_argument(self):
        t=ast.parse((HERE/'lab_caltech20_dvp_statistical_readout_20261021.py').read_text())
        f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='calibrate')
        self.assertFalse(any('test' in a.arg.lower() for a in f.args.args))
        names={n.id for n in ast.walk(f) if isinstance(n,ast.Name)}
        self.assertNotIn('te',names)
        self.assertNotIn('SDK',names)

    def test_protected_gradient_has_no_test_argument(self):
        t=ast.parse((HERE/'lab_caltech20_dvp_statistical_readout_20261021.py').read_text())
        f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='protected_direction')
        self.assertFalse(any('test' in a.arg.lower() or 'query' in a.arg.lower() for a in f.args.args))
        c=json.loads((HERE/'lab_caltech20_DVP_protected_Huber_profile_20261021.json').read_text())
        self.assertEqual(c['max_classes']*c['steps_each_class'],2560)
        self.assertLessEqual(c['fit_seconds'],120)

    def test_targeted_methods_gradient_boundaries(self):
        t=ast.parse((HERE/'lab_caltech20_dvp_statistical_readout_20261021.py').read_text())
        for name in ('coupled_huber_direction','regularized_mixup','regularized_sam','joint_huber'):
            f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertFalse(any('test' in a.arg.lower() or 'query' in a.arg.lower() for a in f.args.args))
        for depth,name in ((4,'coupled_Huber'),(6,'mixup')):
            c=json.loads((HERE/f'lab_caltech20_DVP_L{depth}_{name}_profile_20261021.json').read_text())
            self.assertEqual(c['parents'][str(depth)]['correct'],284 if depth==4 else 287)
            self.assertEqual(c['targets'],{'4':288,'6':322})
            self.assertLessEqual(c['fit_seconds'],120)

if __name__=='__main__':unittest.main()
