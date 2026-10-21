import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_coherent_floor_20261021.py'


class CoherentFloor(unittest.TestCase):
    def test_old_fit_and_two_observation_epochs_held_out(self):
        text=SOURCE.read_text();ast.parse(text)
        for token in ("train=np.flatnonzero(mask)","'current_L6_TRAIN':np.arange(296,336)",
            'zero_frames_fit=False','labels_used=False','TRAIN_response_records=216',
            "results[k]['loss']<=.8*baseline[k]['loss']",'new_classifier_or_mask_generated=False'):
            self.assertIn(token,text)
        self.assertNotIn('.Bench(',text);self.assertNotIn('.settings(',text)
    def test_black_floor_and_nominal_function_contract(self):
        text=SOURCE.read_text()
        self.assertIn('(amplitude+floor)/(1+floor)',text)
        self.assertIn('torch.equal(actual,old.response(',text)
        self.assertIn('abs(finite-gradient)<max(1e-7,.005*abs(gradient))',text)


if __name__=='__main__':unittest.main()
