import unittest
from LightGenV2.tasks.t12_text_to_image.scene_acceptance import metric_comparison


class Acceptance(unittest.TestCase):
    def test_latest_ten_percent_positive_scores(self):
        self.assertTrue(metric_comparison(.91,1)['passed'])
        self.assertFalse(metric_comparison(.89,1)['passed'])
        self.assertTrue(metric_comparison(1.1,1)['passed'])
    def test_negative_preference_never_reverses_better_and_worse(self):
        self.assertTrue(metric_comparison(-2.19,-2)['passed'])
        self.assertFalse(metric_comparison(-2.21,-2)['passed'])
        self.assertLess(metric_comparison(-1.8,-2)['signed_drop_over_abs_reference'],0)
    def test_invalid_percentage_is_not_fabricated(self):
        for a,b in [(1,0),(float('nan'),1),(1,float('inf'))]:
            with self.assertRaises(ValueError):metric_comparison(a,b)


if __name__=='__main__':unittest.main()
