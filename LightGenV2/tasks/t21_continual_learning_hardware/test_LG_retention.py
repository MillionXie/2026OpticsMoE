import unittest,json
from pathlib import Path
from .refine_LG_retention import retention_selection,validate_profile

class TestRetention(unittest.TestCase):
    def test_fixed_profile(self):
        c=json.loads(Path(__file__).with_name('config_LG_retention_20261025.json').read_text());validate_profile(c)
        for key,value in [('architecture','d2nn'),('freeze_A',False),('memory_records_per_old_task',1000),('TEST_gradients',True)]:
            with self.assertRaises(ValueError):validate_profile(dict(c,**{key:value}))
    def test_selection_balances_whole_stage_not_spliced_tasks(self):
        ref={'old':.8,'new':.8}
        scores=lambda a,b:{'old':dict(balanced_accuracy=a),'new':dict(balanced_accuracy=b)}
        self.assertGreater(retention_selection(scores(.78,.78),ref),retention_selection(scores(.7,.95),ref))
        self.assertGreater(retention_selection(scores(.9,.9),ref),retention_selection(scores(.8,.8),ref))
        with self.assertRaises(ValueError):retention_selection(scores(.7,.8),{'old':.8})

if __name__=='__main__':unittest.main()
