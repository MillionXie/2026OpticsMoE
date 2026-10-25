import unittest
import torch
from .export_canonical_phase import canonicalize,phase_codes
from .unified_native import build_unified


class CanonicalPhaseTests(unittest.TestCase):
    def test_all_256_codes_and_nonphase_weights_preserved(self):
        raw=torch.logit((torch.arange(256).float()+.001)/256);head=torch.randn(10,784);s={'router_phase':raw,'shared_head.weight':head};c,p=canonicalize(s)
        torch.testing.assert_close(phase_codes(raw),phase_codes(c['router_phase']),atol=0,rtol=0);torch.testing.assert_close(c['shared_head.weight'],head,atol=0,rtol=0);self.assertEqual(len(p),1);self.assertTrue(torch.equal(s['router_phase'],raw))
    @torch.no_grad()
    def test_original_graph_CPU_outputs_exact(self):
        torch.set_num_threads(2);model=build_unified('moe');state,proof=canonicalize(model.state_dict());other=build_unified('moe');other.load_state_dict(state,strict=True);x=torch.rand(1,224,224);x=x/x.square().sum().sqrt()
        torch.testing.assert_close(model(x)['logits'],other(x)['logits'],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
