import unittest,torch
from .distill_B_active_TRAIN import project_box_mean_one,active_queries
from .fit_augmented_residual_teacher import ResidualTeacher


class ActiveTRAINTests(unittest.TestCase):
    def test_projection_box_positivity_and_mean(self):
        torch.manual_seed(6);base=torch.rand(7,12);base/=base.mean(1,keepdim=True);std=base.std(0,unbiased=False);values=base+torch.randn_like(base)*3
        q=project_box_mean_one(values,base,std)
        self.assertTrue(bool((q>=(base-.02*std).clamp_min(0)-1e-6).all()));self.assertTrue(bool((q<=base+.02*std+1e-6).all()));self.assertTrue(bool((q>=0).all()))
        torch.testing.assert_close(q.mean(1),torch.ones(7),atol=1e-6,rtol=0)
    def test_teacher_frozen_query_inputs_bounded(self):
        torch.manual_seed(4);base=torch.rand(5,12)+.1;base/=base.mean(1,keepdim=True);transform=torch.eye(12);teacher=ResidualTeacher(torch.randn(10,12)).eval()
        for p in teacher.parameters():p.requires_grad_(False)
        before={n:v.clone() for n,v in teacher.state_dict().items()};labels=teacher(base).argmax(1);std=base.std(0,unbiased=False)
        q=active_queries(base,labels,teacher,torch.randn(10,12),transform,std,steps=2)
        self.assertFalse(q.requires_grad);self.assertTrue(torch.isfinite(q).all());torch.testing.assert_close(q.mean(1),torch.ones(5),atol=1e-6,rtol=0)
        self.assertTrue(bool(((q-base).abs()<=.02*std+1e-6).all()))
        for n,v in teacher.state_dict().items():self.assertTrue(torch.equal(before[n],v))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
    def test_nonfinite_projection_rejected(self):
        with self.assertRaises(ValueError):project_box_mean_one(torch.full((2,4),float('nan')),torch.ones(2,4),torch.ones(4))


if __name__=='__main__':unittest.main()
