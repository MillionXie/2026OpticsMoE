import unittest
from ..scene_middle_delivery import threefold_table


class MiddleTiming(unittest.TestCase):
    def test_threefold_transport_budget_not_fourfold(self):
        rows=[dict(role='qwen',label='E',mode='software',mean_ms=3000.),dict(role='light_lg',label='LG',mode='ideal_CCD_fixed_wait',mean_ms=900.)]
        text=threefold_table(rows)
        self.assertEqual(rows[1]['extra_hardware_H_budget_for3x_ms'],100.)
        self.assertTrue(rows[1]['above3x_before_unknown_H'])
        self.assertIn('100.000 ms',text);self.assertIn('3倍目标',text)
    def test_slower_candidate_is_not_approved(self):
        rows=[dict(role='qwen',label='E',mode='software',mean_ms=3000.),dict(role='light_lg',label='LG',mode='ideal_CCD_fixed_wait',mean_ms=1100.)]
        threefold_table(rows);self.assertEqual(rows[1]['extra_hardware_H_budget_for3x_ms'],-100.)
        self.assertFalse(rows[1]['above3x_before_unknown_H'])
    def test_no_reference_cannot_create_speed_claim(self):
        with self.assertRaises(ValueError):threefold_table([dict(role='light_lg',label='LG',mode='ideal_CCD_fixed_wait',mean_ms=900.)])


if __name__=='__main__':unittest.main()
