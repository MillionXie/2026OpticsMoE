import copy
import random

import pytest
import torch
from PIL import Image

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import (
    PROFILES, blend_train_pairs, training_source_exclusion)


def batch():
    rows = [dict(product_id=str(i % 2), sample_id=str(i), image_path=f'{i}.png',
                 split='train' if i < 2 else 'gallery', source_split='train') for i in range(4)]
    images = [Image.new('RGB', (8, 8), (i * 60,) * 3) for i in range(4)]
    return images, rows


def test_blend_is_same_sku_deterministic_and_nonmutating():
    images, rows = batch()
    saved_rows = copy.deepcopy(rows)
    original = [im.tobytes() for im in images]
    a, sources, weights = blend_train_pairs(images, rows, 2, random.Random(12), 1.)
    b, sources2, weights2 = blend_train_pairs(images, rows, 2, random.Random(12), 1.)
    assert sources == sources2 == [('0', '2'), ('1', '3'), ('2', '0'), ('3', '1')]
    assert weights == weights2 and all(.05 <= w <= .15 for w in weights)
    for i, im in enumerate(a):
        expected = Image.blend(images[i], images[(i + 2) % 4], weights[i])
        assert im.tobytes() == b[i].tobytes() == expected.tobytes()
    assert original == [im.tobytes() for im in images] and rows == saved_rows


def test_zero_probability_exact_input_and_rng_no_change():
    images, rows = batch()
    rng = random.Random(12); state = rng.getstate()
    output, sources, weights = blend_train_pairs(images, rows, 2, rng, 0.)
    assert all(a is b for a, b in zip(output, images))
    assert weights == [0.] * 4 and rng.getstate() == state
    assert sources == [('0',), ('1',), ('2',), ('3',)]


@pytest.mark.parametrize('field,value', [('product_id', 'wrong'), ('sample_id', '0'),
    ('image_path', '0.png'), ('split', 'query'), ('source_split', 'test')])
def test_blend_rejects_wrong_identity_or_nontrain(field, value):
    images, rows = batch(); rows[2][field] = value
    with pytest.raises(ValueError):
        blend_train_pairs(images, rows, 2, random.Random(1))


def test_gallery_masks_exclude_every_source_with_remaining_positives():
    ids = [str(i) for i in range(16)]
    mask = training_source_exclusion([('0', '2'), ('1',)], ids)
    assert mask.dtype == torch.bool and mask.sum(1).tolist() == [2, 1]
    same_sku = torch.arange(16) % 2 == 0
    assert (same_sku & ~mask[0]).sum() == 6
    assert ((~same_sku) & ~mask[1]).sum() == 7
    for bad in ([('unknown',)], [('0', '0')], [()]):
        with pytest.raises(ValueError):
            training_source_exclusion(bad, ids)


def test_viewblend_profile_changes_training_augmentation_only():
    p = dict(PROFILES['sku_phase_head_viewblend'])
    assert p.pop('same_sku_blend_probability') == .3
    assert p.pop('same_sku_blend_range') == (.05, .15)
    assert p == PROFILES['sku_phase_head_top1']
