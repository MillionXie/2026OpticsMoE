import unittest
from .adam_block_probe import admits_router,group_contains


class PolicyTests(unittest.TestCase):
    def test_groups_never_move_unselected_tensors(self):
        self.assertTrue(group_contains('router_phase','router'));self.assertFalse(group_contains('expert_phases.0','router'))
        self.assertTrue(group_contains('global_phases.1','global'));self.assertFalse(group_contains('router_phase','global'))

    def test_requires_both_batches_views_gain_above_noise_and_guard_preservation(self):
        row=dict(repair_margin_delta=.0002,guard_new_wrong=0,guard_delta_q10=-.000001)
        self.assertTrue(admits_router([[dict(row),dict(row)],[dict(row),dict(row)]],0))
        cases=[[dict(row),dict(row)],[dict(row),dict(row)]];cases[1][1]['guard_new_wrong']=1;self.assertFalse(admits_router(cases,0))
        self.assertFalse(admits_router([[dict(row),dict(row)],[dict(row),dict(row)]],.001))


if __name__=='__main__':unittest.main()
