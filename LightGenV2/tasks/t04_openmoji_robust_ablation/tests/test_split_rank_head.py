"""Training-host torch contract test: compression never alters decoder capacity."""
import unittest
import torch
from torch import nn
from unittest.mock import patch
from LightGenV2.tasks.t04_semantic_interaction.shared_readout import SharedGridReadout
from LightGenV2.tasks.t04_openmoji_robust_ablation import split_rank_head


class SplitRankTest(unittest.TestCase):
    def test_counts_strict_warmstart_and_unchanged_decoder(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        source = {'shared_readout.'+k: v for k,v in SharedGridReadout().state_dict().items()}
        reference = None
        for rank, count in ((64, 271384), (32, 209944), (48, 240664)):
            model = nn.Module()
            model.shared_readout = SharedGridReadout(variant='lowrank64')
            cfg = type('Cfg', (), {'shared_readout_variant': 'lowrank64', 'editor_rank': rank})()
            with patch.object(split_rank_head.t, 'build_model', return_value=model):
                model = split_rank_head.build_model(cfg, torch.device('cpu'))
            mapped = split_rank_head.adapted_source_state(source, model.state_dict(), rank)
            model.load_state_dict(mapped, strict=True)
            self.assertEqual(sum(p.numel() for p in model.shared_readout.parameters()), count)
            self.assertEqual(sum(p.numel() for p in model.shared_readout.decoder.parameters()), 30162)
            decoder = model.shared_readout.decoder.state_dict()
            if reference is None:
                reference = decoder
            else:
                self.assertTrue(all(torch.equal(v, reference[k]) for k,v in decoder.items()))
            output = model.shared_readout(torch.randn(2,192,14,14),torch.randn(2,192))
            self.assertTrue(torch.isfinite(output['category_logits']).all())
            model.load_state_dict(model.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
