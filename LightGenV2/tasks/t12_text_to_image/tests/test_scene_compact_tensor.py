import unittest
import torch
from ..scene_compact_candidates import regrid_paths


class CompactTensorEndpoints(unittest.TestCase):
    def fixture(self):
        sigmas=[float(30-i) for i in range(31)]
        path=torch.stack([torch.full((4,64,64),2.*s+3.)for s in sigmas])
        return dict(teacher_steps=30,sigmas=sigmas,timesteps=[float(999-30*i)for i in range(30)],paths={'TRAIN':path})

    def test_actual_tensor_endpoints_reconstruct_teacher_for_both_grids(self):
        for steps in (6,8):
            saved=self.fixture();rows,edges=regrid_paths(saved,steps)
            self.assertEqual(len(rows),steps)
            for r,(start,stop) in zip(rows,zip(edges,edges[1:])):
                end=r['x']+r['target_epsilon']*(saved['sigmas'][stop]-saved['sigmas'][start])
                torch.testing.assert_close(end,saved['paths']['TRAIN'][stop],rtol=0,atol=0)
                self.assertEqual(r['split'],'train')

    def test_nonfinite_or_incomplete_teacher_paths_cannot_make_targets(self):
        saved=self.fixture();saved['paths']['TRAIN'][1,0,0,0]=float('nan')
        with self.assertRaises(ValueError):regrid_paths(saved,6)
        saved=self.fixture();saved['sigmas'][-1]=1.
        with self.assertRaises(ValueError):regrid_paths(saved,8)


if __name__=='__main__':unittest.main()
