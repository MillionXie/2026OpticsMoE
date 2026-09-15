import copy,unittest
import torch
from .core import grouped_weights,merge_states,state_digest

class MergeContract(unittest.TestCase):
    def test_group_energy(self):
        w=grouped_weights(torch.tensor([0,1,0,1]));torch.testing.assert_close(w.square().sum(1),torch.ones(4))
        self.assertTrue(bool((w[0,2:]==0).all()));self.assertTrue(bool((w[1,:2]==0).all()))
        with self.assertRaises(ValueError):grouped_weights(torch.tensor([2]))

    def test_copy_only_experts_and_reject_different_shared_interface(self):
        shared={'backbone_metadata':{'frozen_stem_sha256':'same'},'vision_optical':{f'core.experts.{i}.raw_phase':torch.zeros(2,2) for i in range(4)},'classification_head':{'weight':torch.ones(3)}}
        shared['vision_optical']['global.raw_phase']=torch.zeros(2,2)
        a=copy.deepcopy(shared);b=copy.deepcopy(shared)
        for i in (0,1):a['vision_optical'][f'core.experts.{i}.raw_phase'].add_(i+1)
        for i in (2,3):b['vision_optical'][f'core.experts.{i}.raw_phase'].add_(i+1)
        merged=merge_states(shared,a,b)
        for i in range(4):self.assertTrue(torch.equal(merged['vision_optical'][f'core.experts.{i}.raw_phase'],torch.full((2,2),float(i+1))))
        self.assertEqual(state_digest(shared,True),state_digest(merged,True))
        b['classification_head']['weight'][0]+=1
        with self.assertRaises(RuntimeError):merge_states(shared,a,b)

if __name__=='__main__':unittest.main()
