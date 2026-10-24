"""CPU admission tests, not a physical performance claim."""
import unittest
import torch
from .model import build_model
from .subset import memory_indices


class TestNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_paired_memory(self):
        m=dict(split='train',records_per_group=2,group_strata=[str(i%10) for i in range(500)])
        ix=memory_indices(m)
        self.assertEqual(len(set(ix)),300)
        self.assertTrue((ix[1::2]==ix[::2]+1).all())
        with self.assertRaises(ValueError):memory_indices(dict(m,split='test'))

    def test_payload_replay_and_gradient(self):
        for architecture in ('moe','d2nn'):
            torch.manual_seed(73);model=build_model(architecture);model.configure_stage(1)
            x=torch.rand(1,224,224);x=x/x.square().sum((-2,-1),keepdim=True).sqrt()
            out=model(x,return_debug=True)
            replay=model(x,measured={k:v.detach() for k,v in out['ccd'].items()})
            torch.testing.assert_close(out['logits'],replay['logits'],atol=0,rtol=0)
            prefix={}
            for stage in (model.STAGES if architecture=='moe' else model.STAGES[1:]):
                amplitude,phase,_=model.payload(x,stage,prefix)
                torch.testing.assert_close(amplitude.detach()*255,(amplitude.detach()*255).round(),atol=2e-5,rtol=0)
                self.assertTrue(bool((amplitude>=0).all() and (amplitude<=1).all()))
                torch.testing.assert_close(phase.detach()*256,(phase.detach()*256).round(),atol=0,rtol=0)
                prefix[stage]=out['ccd'][stage]
            out['logits'].square().mean().backward()
            self.assertTrue(bool(torch.isfinite(model.global_phase.grad).all()))
            self.assertGreater(float(model.global_phase.grad.abs().max()),0)
            self.assertEqual(sum(p.numel() for p in model.shared_head.parameters()),7840)
            if architecture=='moe':
                active=set(model.active_indices[:4].tolist())
                self.assertEqual({i for i,p in enumerate(model.first_phase) if p.grad is not None},active)
                model.configure_stage(2)
                self.assertEqual({i for i,p in enumerate(model.first_phase) if p.requires_grad},set(model.active_indices[4:8].tolist()))


if __name__=='__main__':unittest.main()
