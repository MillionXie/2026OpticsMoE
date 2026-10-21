import ast
from pathlib import Path
import unittest

class CoherentResponseContract(unittest.TestCase):
    def test_no_device_or_classifier_training(self):
        text=(Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_coherent_response_20261021.py').read_text();ast.parse(text)
        for banned in ('.Bench(','.camera.',"data['test']",'linear.weight','fusion_logits'):
            self.assertNotIn(banned,text)
        for required in ('TEST_used=False','original_deployed_graph_unchanged=True','TRAIN_response_records=216','next_mask_or_capture_started=False','best_state'):
            self.assertIn(required,text)

if __name__=='__main__':unittest.main()
