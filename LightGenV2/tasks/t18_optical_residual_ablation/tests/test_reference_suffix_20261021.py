import ast,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
class ReferenceSuffix(unittest.TestCase):
    def test_loss_no_test_argument(self):
        t=ast.parse((HERE/'lab_caltech20_dvp_reference_suffix_20261021.py').read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='teacher_loss')
        self.assertFalse(any('test' in a.arg.lower() for a in f.args.args))
    def test_bounded_profile(self):
        c=json.loads((HERE/'lab_caltech20_DVP_reference_suffix_profile_20261021.json').read_text());self.assertEqual(c['targets'],{'4':288,'6':322});self.assertEqual(c['maximum_epochs'],3);self.assertEqual(c['parents']['4']['correct'],285);self.assertEqual(c['parents']['6']['correct'],287);self.assertEqual(c['dropout_location'],'CCD_pixels');self.assertEqual(c['feature_dropout_p'],0)
    def test_frozen_readout_SAM_scope(self):
        c=json.loads((HERE/'lab_caltech20_DVP_L6_frozen_readout_phase_SAM_profile_20261021.json').read_text());self.assertTrue(c['freeze_readout']);self.assertEqual(c['lr_head'],0);self.assertEqual(c['teacher_mode'],'current_measured_parent');self.assertEqual(c['feature_alignment'],0);self.assertEqual(c['phase_SAM_max'],.04);self.assertEqual(c['targets'],{'6':322})
if __name__=='__main__':unittest.main()
