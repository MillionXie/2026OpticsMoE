import unittest
import torch
from torch import nn

from TransferFromElectricity.tasks.t01_object_retrieval.models.generator import StaticGenerator, LoRALinear, install_lora
from TransferFromElectricity.tasks.t01_object_retrieval.models.injection import ExpertInjection
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import PhaseLayer, AngularSpectrumPropagator, phase_dc_loss


class StaticExpertsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_initial_anchor_update_and_compact_roundtrip(self):
        model = StaticGenerator(size=16, patch=4)
        initial = model().detach().clone()
        torch.testing.assert_close(initial, model.anchor, rtol=0, atol=1e-7)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model().square().mean().backward()
        self.assertGreater(model.tokens.grad.norm().item(), 0)
        opt.step()
        self.assertGreater((model()-initial).abs().max().item(), 0)
        other = StaticGenerator(size=16, patch=4)
        other.load_compact_state(model.compact_state())
        torch.testing.assert_close(other(), model(), rtol=0, atol=0)

    def test_injection_preserves_optics_gradient_dc_and_export(self):
        planes = nn.ModuleList([PhaseLayer(16, init='small_normal') for _ in range(8)])
        raw = torch.stack([p.raw_phase.detach() for p in planes]).reshape(2,4,16,16).requires_grad_()
        prop = AngularSpectrumPropagator(532e-9, 17e-6, 16, 0.1)
        field = torch.rand(3,16,16).to(torch.complex64)
        baseline = prop(planes[0](field))
        injection = ExpertInjection(planes)
        injection.bind(raw)
        result = prop(planes[0](field))
        torch.testing.assert_close(result, baseline, rtol=0, atol=0)
        loss = result.abs().square()[:, :5, :7].sum() + phase_dc_loss(planes)
        loss.backward()
        self.assertGreater(raw.grad.norm().item(), 0)
        self.assertTrue(all(p.parametrizations.raw_phase.original.grad is None for p in planes))
        with torch.no_grad():
            injection.materialize()
            torch.testing.assert_close(prop(planes[0](field)), result, rtol=0, atol=0)
            # Accompanying samples cannot affect a static phase / linear optics.
            torch.testing.assert_close(prop(planes[0](field[:1])), result[:1])

    def test_lora_identity_and_updates_inside_tiny_qwen(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        cfg = Qwen3VLTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, rope_scaling={'rope_type':'default','mrope_section':[1,1,2]})
        model = Qwen3VLTextModel(cfg).eval().requires_grad_(False)
        ids = torch.tensor([[1,2,3,4],[1,2,5,6]])
        before = model(input_ids=ids).last_hidden_state.detach()
        installed = install_lora(model, 2)
        self.assertEqual(len(installed), 4)
        result = model(input_ids=ids).last_hidden_state
        torch.testing.assert_close(result, before)
        result[:,-1,0].sum().backward()
        b = [p for n,p in model.named_parameters() if n.endswith('lora_b')]
        self.assertGreater(sum(p.grad.abs().sum().item() for p in b), 0)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n))


if __name__ == '__main__':
    unittest.main()
