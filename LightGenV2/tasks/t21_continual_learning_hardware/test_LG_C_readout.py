import json,unittest
from pathlib import Path
from .refine_LG_C_readout import validate


class LGCTests(unittest.TestCase):
    def test_fixed_budget_and_original_head_only(self):
        cfg=json.loads(Path(__file__).with_name('config_LG_C_readout_20261025.json').read_text());validate(cfg)
        for key,value in [('epochs',10000),('TEST_gradients',True),('TRAIN_counts',{'speech_binary':1000})]:
            with self.assertRaises(ValueError):validate(dict(cfg,**{key:value}))


if __name__=='__main__':unittest.main()
