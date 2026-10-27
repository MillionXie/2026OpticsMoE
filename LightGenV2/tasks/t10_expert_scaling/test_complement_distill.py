import unittest
import numpy as np
from .complement_distill import Auxiliary,complement_masks,pick_extra
from .reheat_refine import rate


class PolicyTests(unittest.TestCase):
    def test_masks_use_same_view_parent_and_teacher(self):
        y=np.array([0,1]);q=np.eye(2)[np.array([[0,1],[1,1]])];p=np.eye(2)[np.array([[1,1],[0,0]])]
        repair,retain=complement_masks(q,p,y)
        np.testing.assert_array_equal(repair,[[True,False],[False,True]])
        np.testing.assert_array_equal(retain,[[False,True],[True,False]])

    def test_extra_sampling_avoids_repeated_and_main_ids(self):
        pool=np.array([[v,i] for v in range(2) for i in range(10)])
        result=pick_extra(pool,18,[0,1,9]);ids=[i for code,i in result]
        self.assertEqual(len(set(ids)),4);self.assertTrue(set(ids).isdisjoint([0,1,9]))
        for code,index in result:self.assertIn([code,index],pool.tolist())

    def test_fixed_sixteen_round_schedule_and_input_budget(self):
        c=dict(schedule_type='distill',rounds=16,peak_lr=.0003,end_lr=.00003)
        self.assertAlmostEqual(rate(1,c),.0001);self.assertAlmostEqual(rate(3,c),.0003);self.assertAlmostEqual(rate(16,c),.00003)
        self.assertEqual(16*64,1024);self.assertEqual(16*1024,16384);self.assertEqual(16*64*4,4096)

    def test_calibration_extra_rows_repeat_exactly_and_are_disjoint_from_all_main_ids(self):
        from types import SimpleNamespace
        from .cached_distill import calibration_indices
        a=Auxiliary.__new__(Auxiliary);a.y=np.repeat(np.arange(8),437);a.ids=np.array([str(i) for i in range(3496)])
        a.teacher=np.broadcast_to(np.eye(8)[a.y],(8,3496,8)).copy();a.parent=a.teacher.copy()
        a.parent[:,::3]=np.eye(8)[(a.y[::3]+1)%8]
        a.pool_mask,a.parent_correct=complement_masks(a.teacher,a.parent,a.y)
        a.calibration_view0=True;a.output=None;a.start_round(0)
        batches=calibration_indices(a.y,a.teacher[0],a.parent[0]);main=set(np.concatenate(batches));extra=[]
        for batch in batches:
            ix=SimpleNamespace(numpy=lambda batch=batch:batch)
            a.start_batch(ix,0);first=a.extra_indices.copy();a.start_batch(ix,0)
            np.testing.assert_array_equal(first,a.extra_indices);self.assertTrue((a.extra_codes==0).all())
            self.assertTrue(a.pool_mask[a.extra_codes,a.extra_indices].all());extra.extend(first.tolist())
        self.assertEqual(len(set(extra)),8);self.assertTrue(set(extra).isdisjoint(main))


if __name__=='__main__':unittest.main()
