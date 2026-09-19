import unittest
import numpy as np
import torch
from LightGenV2.tasks.t11_lifelong_optics.model import OpticalMoE,loss
from LightGenV2.tasks.t11_lifelong_optics.data import balanced_indices,domain

class Contract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg=dict(seed=17,expert_size=24,gap=4,border=4,wavelength_m=5.32e-7,pixel_size_m=1.7e-5,distance_m=.1,router_detector_size=8,detector_size=8)
        self.m=OpticalMoE(self.cfg)
        self.x=torch.randint(1,255,(3,30,30,3),dtype=torch.uint8)
        self.y=torch.tensor([0,3,7])

    def test_power_masks_and_geometry(self):
        m=self.m; geometry=(m.height,m.width,m.slots.copy())
        for stage in ('A','warmup','B'):
            m.configure(stage); out=m(self.x,warmup=stage=='warmup')
            self.assertTrue(torch.allclose(out['output_power'],torch.ones(3),atol=2e-6))
            self.assertTrue(torch.allclose(out['routes'].sum(1),torch.ones(3)))
            self.assertTrue(torch.all(out['routes'][:,int(m.active_count):]==0))
            if stage=='warmup':
                self.assertTrue(torch.equal(out['routes'][:,:4],torch.zeros(3,4)))
                self.assertTrue(torch.equal(out['routes'][:,4:8],torch.full((3,4),.25)))
            self.assertEqual(geometry,(m.height,m.width,m.slots))
        out=m(self.x,mask=[True]*4+[False]*8)
        self.assertTrue(torch.equal(out['routes'][:,4:],torch.zeros(3,8)))
        with self.assertRaises(ValueError): m(self.x,mask=[False]*12)

    def test_freezing_with_adam(self):
        m=self.m
        for stage in ('A','warmup','B'):
            m.configure(stage)
            before={n:p.detach().clone() for n,p in m.named_parameters()}
            optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True); loss(m(self.x,warmup=stage=='warmup'),self.y).backward()
                for n,p in m.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad,n); self.assertTrue(torch.isfinite(p.grad).all(),n)
                        self.assertGreater(p.grad.abs().sum().item(),0,n)
                optimizer.step()
            for n,p in m.named_parameters():
                if p.requires_grad: self.assertFalse(torch.equal(p,before[n]),n)
                else: self.assertTrue(torch.equal(p,before[n]),n)

    def test_balanced_replay_and_domain(self):
        labels=np.repeat(np.arange(8),437)
        ids=balanced_indices(labels,256,17)
        self.assertEqual(np.bincount(labels[ids]).tolist(),[32]*8)
        self.assertEqual(len(set(ids)),256)
        self.assertTrue(torch.equal(domain(self.x,'A'),self.x))
        self.assertFalse(torch.equal(domain(self.x,'B'),self.x))
        self.assertEqual(domain(self.x,'B').dtype,torch.uint8)
        with self.assertRaises(ValueError): self.m(self.x.float()/255)

if __name__=='__main__': unittest.main()
