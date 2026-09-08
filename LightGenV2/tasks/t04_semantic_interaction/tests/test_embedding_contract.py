from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
from LightGenV2.tasks.t04_semantic_interaction.embedding_model import PositionReadout, ScaleOnlyCCD, PoolOnlyReadout, position_encoding
from LightGenV2.tasks.t04_semantic_interaction.embedding_data import identity

TASK = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name', ['embedding_alpha40', 'embedding_alpha40_lean', 'embedding_d2nn_alpha40'])
def test_profiles(name):
    cfg = load_settings(TASK / 'configs' / (name + '.yaml'))
    assert cfg.embedding_only
    assert .4 < cfg.fusion_alpha_minimum < cfg.optical_fusion_initial < cfg.fusion_alpha_maximum
    assert cfg.epochs == 100
    assert cfg.prompt_cache_path.name == 'token_embeddings_v1.pt'

def test_linear_position_readout_not_concat_or_max():
    readout = PositionReadout(4)
    values = torch.arange(6.).reshape(2, 3).requires_grad_()
    result = readout([values])
    torch.testing.assert_close(result[0], values[0] * readout.weight[0] + values[1] * readout.weight[1])
    assert not torch.allclose(result, readout([values.flip(0)]))
    assert result.shape == (1, 3)
    result.sum().backward()
    assert values.grad is not None and readout.weight.grad is not None
    with pytest.raises(ValueError):
        readout([torch.zeros(5, 3)])

def test_ccd_scalar_and_linear_pool():
    norm = ScaleOnlyCCD()
    value = torch.arange(1., 17.).reshape(1, 4, 4)
    torch.testing.assert_close(norm(value), norm(3 * value))
    assert float(norm(value)[0, 3, 3] / norm(value)[0, 0, 0]) == 16.
    aperture = SimpleNamespace(height=4, width=4, x0=0, y0=0, x1=4, y1=4)
    readout = PoolOnlyReadout(SimpleNamespace(geometry=SimpleNamespace(detector_aperture=aperture), output_size=2))
    torch.testing.assert_close(readout.forward_intensity(value * 2)[0], 2 * readout.forward_intensity(value)[0])
    assert not list(readout.parameters())

def test_position_and_identity():
    code = position_encoding(5, 192, 'cpu')
    assert code.shape == (5, 192) and not torch.allclose(code[0], code[1])
    assert identity({'source_grid': [[1, 0]], 'instruction': 'move'}) == identity({'source_grid': [[1, 0]], 'instruction': 'move'})
