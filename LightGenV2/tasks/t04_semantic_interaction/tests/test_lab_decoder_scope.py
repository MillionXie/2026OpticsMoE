"""CPU checks for the physical decoder-only adaptation boundary."""
import copy,unittest
import torch
from torch import nn
from LightGenV2.tasks.t04_semantic_interaction.lab_tune_exp05_decoder import protected

class ScopeTests(unittest.TestCase):
    def test_decoder_change_allowed_but_upstream_detected(self):
        model=nn.Module();model.upstream=nn.Linear(3,3);model.decoder=nn.Linear(3,2)
        before=protected(model,'decoder')
        with torch.no_grad():model.decoder.weight.add_(1)
        self.assertEqual(before,protected(model,'decoder'))
        with torch.no_grad():model.upstream.weight.add_(1)
        self.assertNotEqual(before,protected(model,'decoder'))

    def test_only_decoder_has_gradients(self):
        model=nn.Module();model.upstream=nn.Linear(3,3);model.decoder=nn.Linear(3,2)
        model.requires_grad_(False);model.decoder.requires_grad_(True)
        feature=model.upstream(torch.ones(2,3)).detach();model.decoder(feature).square().sum().backward()
        self.assertIsNone(model.upstream.weight.grad);self.assertIsNotNone(model.decoder.weight.grad)

    def test_best_snapshot_is_not_mutated_by_last(self):
        head=nn.Linear(3,2);best=copy.deepcopy(head.state_dict());old=best['weight'].clone()
        with torch.no_grad():head.weight.add_(1)
        torch.testing.assert_close(best['weight'],old)

if __name__=='__main__':unittest.main()
