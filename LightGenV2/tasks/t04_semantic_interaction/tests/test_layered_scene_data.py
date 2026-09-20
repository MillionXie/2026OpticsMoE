from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('resvg_py')

from LightGenV2.tasks.t04_semantic_interaction.layered_scene_data import (
    MIN_VISIBLE, generate_example, render_grid,
)

SVG = Path(__file__).resolve().parents[1] / 'assets/openmoji-17.0.0-svg'


@pytest.mark.parametrize('task', ['add','replace','move','remove'])
def test_layered_examples_keep_anchor_contract_and_are_visible(task):
    for index in range(8):
        example=generate_example(task,73_000_000+index,SVG)
        assert example['source_grid'].shape==example['target_grid'].shape==(6,6)
        assert np.array_equal(example['edit_grid'],example['source_grid']!=example['target_grid'])
        assert np.count_nonzero(example['edit_grid']) in (1,2)
        for grid in (example['source_grid'],example['target_grid']):
            image,objects=render_grid(grid,SVG)
            assert image.size==(224,224)
            assert min(o['visible_alpha_fraction'] for o in objects)>=MIN_VISIBLE


def test_render_is_deterministic_and_contains_real_occlusion():
    example=generate_example('move',73_000_123,SVG)
    a,objects=render_grid(example['source_grid'],SVG)
    b,_=render_grid(example['source_grid'],SVG)
    assert np.array_equal(a,b)
    # Search a deterministic short prefix; scenes are designed to include,
    # but not force, mild occlusion.
    fractions=[]
    for seed in range(73_001_000,73_001_040):
        _,rows=render_grid(generate_example('add',seed,SVG)['source_grid'],SVG)
        fractions.extend(o['visible_alpha_fraction'] for o in rows)
    assert any(MIN_VISIBLE <= value < .995 for value in fractions)
