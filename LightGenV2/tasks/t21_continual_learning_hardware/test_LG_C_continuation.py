import unittest,json
from pathlib import Path
from .continue_LG_C_optics import validate


class CContinuationTests(unittest.TestCase):
    def test_finite_budget_and_original_memory(self):
        c=json.loads(Path(__file__).with_name('config_LG_C_optical_continuation_20261025.json').read_text());validate(c)
        for key,value in [('epochs',20),('memory_records_per_old_task',1000),('TEST_gradients',True)]:
            with self.assertRaises(ValueError):validate(dict(c,**{key:value}))


if __name__=='__main__':unittest.main()
