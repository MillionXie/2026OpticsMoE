import unittest

from LightGenV2.tasks.t12_text_to_image.qwen_vl_hybrid_workflow import verify_feature_count


class WorkflowTests(unittest.TestCase):
    def test_reuse_count_follows_recorded_cohort(self):
        for train in (960, 1824):
            self.assertEqual(verify_feature_count({'complete': True, 'count': train + 96},
                             {'pairs': {'train': train, 'val': 96}}), train + 96)
        with self.assertRaises(ValueError):
            verify_feature_count({'complete': True, 'count': 1056},
                                 {'pairs': {'train': 1824, 'val': 96}})
        with self.assertRaises(ValueError):
            verify_feature_count({'complete': False, 'count': 1920},
                                 {'pairs': {'train': 1824, 'val': 96}})
