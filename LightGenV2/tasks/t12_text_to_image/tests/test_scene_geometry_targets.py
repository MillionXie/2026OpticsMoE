import unittest
import torch
from ..scene_geometry_targets import regrid


class AdaptiveGridEndpoints(unittest.TestCase):
    def test_every_nonuniform_interval_ends_at_the_saved_teacher_state(self):
        sigmas=[float(30-i)for i in range(31)];x=torch.stack([torch.full((4,64,64),.5*s*s+2.)for s in sigmas])
        saved=dict(paths={'TRAIN':x},sigmas=sigmas,timesteps=[float(999-i*30)for i in range(30)])
        edges=[0,3,9,16,22,27,30];rows=regrid(saved,edges)
        self.assertEqual(len(rows),6)
        for row,stop in zip(rows,edges[1:]):
            actual=row['x']+row['target_epsilon']*(sigmas[stop]-row['sigma'])
            torch.testing.assert_close(actual,x[stop],atol=1e-5,rtol=1e-5)
            self.assertEqual(row['split'],'train')
        torch.testing.assert_close(saved['paths']['TRAIN'],x,atol=0,rtol=0)

    def test_reordered_or_long_teacher_intervals_are_rejected(self):
        for edges in [[0,9,10,14,20,25,30],[0,3,9,16,22,30,27],[0,5,10,15,20,25,29]]:
            with self.assertRaises(ValueError):regrid({},edges)


if __name__=='__main__':unittest.main()
