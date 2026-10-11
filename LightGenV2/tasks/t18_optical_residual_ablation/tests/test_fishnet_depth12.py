"""Depth extension preserves the old graph and uses the frozen training loop."""
import copy
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='Verified Linux Torch; preserve Windows environment')
TASK = Path(__file__).resolve().parents[1]


def test_depth12_recipe_and_pair_initialization(monkeypatch):
    sys.path.insert(0, str(TASK))
    import fishnet_depth12 as d
    import torch
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *args, **kwargs: self)
    base = copy.deepcopy(d.f.r.CONFIGS['moe_L6_oeo_relu_softsign'])
    graph = d.graph_config(base)
    assert d.f.r.CONFIGS['moe_L6_oeo_relu_softsign'] == base
    assert graph['model']['num_cycles'] == graph['global_fc']['num_masks'] == 6
    assert graph['expert_bank']['d2nn']['nonlinear_schedule'] == [True] * 6
    for key in ['optics', 'prompt', 'optical_router', 'nonlinearity', 'global_oeo', 'detector', 'readout', 'regularization']:
        assert graph[key] == base[key]
    monkeypatch.setitem(d.f.r.CONFIGS, 'moe_L12_oeo_relu_softsign', graph)
    cfg = d.f.t.config(0.)
    d.f.r.setseed(17)
    left = d.f.t.original_build('moe', 12, cfg)
    left_sha = d.f.tensor_identity(left)
    assert len(d.install_depth12_residual(left, d.f.t.PhaseLayer, 0.)) == 60
    assert d.f.tensor_identity(left) == left_sha
    d.f.r.setseed(17)
    right = d.f.t.original_build('moe', 12, cfg)
    assert len(d.install_depth12_residual(right, d.f.t.PhaseLayer, .3)) == 60
    assert d.f.tensor_identity(right) == left_sha
    assert sum(p.numel() for p in right.parameters()) == d.PARAMETERS == 2649088
    assert not any(isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) for module in right.modules())
    for name, module in right.named_modules():
        if 'router' in name:
            assert not hasattr(module, 'residual_rho')
