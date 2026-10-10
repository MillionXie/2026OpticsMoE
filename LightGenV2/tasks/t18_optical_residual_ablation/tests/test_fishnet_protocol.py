"""Data leakage and record-preservation contracts for the new dataset."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_fishnet import grouped_split


def examples():
    return [dict(label=k, original_hash=f'original-{k}-{i}', cache_hash=f'cache-{k}-{i}')
            for k in range(8) for i in range(20)]


def test_both_pixel_spaces_group_transitively_without_dropping_records():
    rows = examples()
    rows[1]['original_hash'] = rows[0]['original_hash']
    rows[2]['cache_hash'] = rows[1]['cache_hash']
    split, count = grouped_split(rows, 20261011)
    owner = {i: key for key, indices in split.items() for i in indices}
    assert len(owner) == len(rows) == 160
    assert owner[0] == owner[1] == owner[2] and count == 158
    assert all({rows[i]['label'] for i in indices} == set(range(8)) for indices in split.values())
    assert grouped_split(rows, 20261011)[0] == split


@pytest.mark.parametrize('kind', ['original_hash', 'cache_hash'])
def test_conflicting_class_for_identical_pixels_is_rejected(kind):
    rows = examples()
    rows[20][kind] = rows[0][kind]
    with pytest.raises(AssertionError, match='conflicting labels'):
        grouped_split(rows, 20261011)


@pytest.mark.skipif(sys.platform == 'win32', reason='Native Torch check uses the verified Linux training environment')
def test_uneven_microbatch_preserves_weighted_loss_and_regularizer_gradient():
    import torch
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 8).double()
    x = torch.randn(7, 3, dtype=torch.double)
    y = torch.tensor([0, 7, 1, 2, 3, 4, 6])
    weights = torch.arange(1, 9, dtype=torch.double) / 4

    def loss(indices):
        term = torch.nn.functional.cross_entropy(model(x[indices]), y[indices], reduction='none', label_smoothing=.02)
        return (term * weights[y[indices]]).mean() + .02 * model.weight.square().mean()

    loss(torch.arange(7)).backward()
    full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for indices in torch.arange(7).split(2):
        (loss(indices) * len(indices) / 7).backward()
    for p, reference in zip(model.parameters(), full):
        torch.testing.assert_close(p.grad, reference, rtol=1e-12, atol=1e-12)
