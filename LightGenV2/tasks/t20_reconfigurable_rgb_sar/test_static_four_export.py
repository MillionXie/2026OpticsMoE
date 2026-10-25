import unittest
import torch
from .static_four import build_static_four,geometry
from .export_static_four_phase import canonicalize,phase_codes,compare


class StaticExportTests(unittest.TestCase):
    @torch.no_grad()
    def test_exact_existing_CPU_graph_and_nonphase_state(self):
        torch.set_num_threads(2);original=build_static_four('moe');state,proof=canonicalize(original.state_dict());export=build_static_four('moe');export.load_state_dict(state,strict=True)
        self.assertEqual(geometry(original),geometry(export));self.assertEqual(len(proof),6)
        for name,value in original.state_dict().items():
            if 'phase' in name:self.assertTrue(torch.equal(phase_codes(value),phase_codes(state[name])))
            else:self.assertTrue(torch.equal(value,state[name]))
        x=torch.rand(2,224,224);x=x/x.square().sum((1,2),keepdim=True).sqrt()
        torch.testing.assert_close(original(x)['logits'],export(x)['logits'],rtol=0,atol=0)

    def test_prediction_gate_is_not_relaxed_when_accuracy_matches(self):
        a=dict(ids=['a'],labels=[0],domains=[1],predictions=[1],probabilities=[[.1,.4501,.4499]])
        b=dict(a,predictions=[2],probabilities=[[.1,.4499,.4501]])
        self.assertFalse(compare(a,b,.002)['passed']);self.assertTrue(compare(a,a,0.)['passed'])


if __name__=='__main__':unittest.main()
