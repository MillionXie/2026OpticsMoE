import unittest
from ..scene_fixed_condition_recovery import checked_freeze


class FixedConditionRecovery(unittest.TestCase):
    def test_historical_default_and_correct_parent(self):
        self.assertFalse(checked_freeze({},None,1200,'qwen'))
        p=dict(condition_correction=dict(UNet_LoRA_unchanged=True),image_condition_correction=dict(text_pooled_projection_unchanged=True))
        self.assertTrue(checked_freeze(dict(freeze_corrected_condition_head=True),p,0,'qwen'))
    def test_wrong_or_unverified_lineage_rejected(self):
        profile=dict(freeze_corrected_condition_head=True)
        for parent,warmup,variant in[(None,0,'qwen'),({},0,'qwen'),({},10,'qwen'),({},0,'native')]:
            with self.assertRaises(ValueError):checked_freeze(profile,parent,warmup,variant)


if __name__=='__main__':unittest.main()
