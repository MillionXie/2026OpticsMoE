import ast
from pathlib import Path
import unittest

class SingleTRAINLDAContract(unittest.TestCase):
    def test_statistics_only_TRAIN_no_new_layer_or_hardware(self):
        text=(Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_fullTRAIN_lda_20261021.py').read_text();ast.parse(text)
        segment=text.split('def fit_head(x,y):',1)[1].split('def selftest():',1)[0]
        for value in ('test_features','test_labels','.cuda(','.Bench(','.backward(','.Parameter('):self.assertNotIn(value,text if value.startswith('.') else segment)
        for value in ('TEST_fit_records=0','new_inference_layers=0','all_nonhead_frozen=True','original_full_CPU_TRAIN1600_TEST411_verified=True','OAS_shrinkage=shrinkage'):self.assertIn(value,text)

if __name__=='__main__':unittest.main()
