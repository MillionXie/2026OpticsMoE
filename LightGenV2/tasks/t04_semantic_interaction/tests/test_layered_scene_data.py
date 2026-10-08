import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('resvg_py')

from LightGenV2.tasks.t04_semantic_interaction.layered_scene_data import (
    MIN_VISIBLE, _save, generate_example, render_grid,
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


def test_saved_scene_metadata_is_json_serializable(tmp_path):
    example=generate_example('move',73_000_123,SVG)
    metadata=_save(example,tmp_path/'sample',SVG)
    loaded=json.loads((tmp_path/'sample'/'scene.json').read_text(encoding='utf-8'))
    assert loaded['program']==metadata['program']
    assert all(isinstance(value,int) for value in loaded['program']['new_anchor'])


def test_layered_gallery_uses_saved_scene_images(tmp_path):
    from LightGenV2.tasks.t04_semantic_interaction.layered_scene_gallery import save_layered_gallery

    example=generate_example('move',73_000_123,SVG)
    sample_id='test_000000'
    _save(example,tmp_path/'data'/'test'/sample_id,SVG)
    galleries={'move':[{
        'sample_id':sample_id,
        'task':'move',
        'instruction':example['instruction'],
        'target':example['target_grid'],
        'prediction':example['target_grid'],
    }]}
    settings=SimpleNamespace(data_dir=tmp_path/'data',svg_asset_dir=SVG)
    save_layered_gallery(tmp_path/'gallery',galleries,settings)
    assert (tmp_path/'gallery'/'move_examples.png').is_file()
    assert (tmp_path/'gallery'/'index.html').is_file()


def test_focused_profile_keeps_inference_contract_and_changes_training_only():
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings

    task_dir=Path(__file__).resolve().parents[1]
    formal=load_settings(task_dir/'configs'/'layered_scene_formal.yaml')
    focused=load_settings(task_dir/'configs'/'layered_scene_focus_changed_iou.yaml')
    assert focused.layout_version==formal.layout_version
    assert focused.lightgen_model_variant==formal.lightgen_model_variant
    assert focused.electronic_width==formal.electronic_width
    assert focused.editor_depth==formal.editor_depth
    assert focused.fusion_alpha_minimum==formal.fusion_alpha_minimum
    assert focused.fusion_alpha_maximum==formal.fusion_alpha_maximum
    assert focused.changed_cell_weight>formal.changed_cell_weight
    assert focused.edit_loss_weight>formal.edit_loss_weight
    assert focused.phase_dropout_p>formal.phase_dropout_p
