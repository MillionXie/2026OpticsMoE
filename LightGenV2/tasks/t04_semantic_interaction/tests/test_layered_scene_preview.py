import numpy as np
import pytest

from LightGenV2.tasks.t04_semantic_interaction.layered_scene_preview import (
    Object, examples, render, validate_pair,
)


@pytest.mark.parametrize('example', examples(), ids=lambda e:e['id'])
def test_only_requested_object_changes_and_no_excessive_occlusion(example):
    validate_pair(example)
    source,rows=render(example['source'])
    target,_=render(example['target'])
    assert source.size==target.size==(224,224)
    assert np.any(np.asarray(source)!=np.asarray(target))
    assert min(r['visible_alpha_fraction'] for r in rows)>=.7


def test_foreground_really_occludes_background_and_removal_reveals_it():
    scene=[Object('house','house',112,180,154,0), Object('dog','dog',105,185,29,1)]
    image,rows=render(scene)
    clean,_=render(scene[:1])
    assert rows[0]['visible_alpha_fraction']<1
    assert rows[1]['visible_alpha_fraction']==pytest.approx(1.)
    assert not np.array_equal(image,clean)
    assert np.array_equal(render(scene[:1])[0],clean)


def test_invalid_geometry_fails_instead_of_silently_clipping():
    with pytest.raises(ValueError,match='Clipped'):
        render([Object('house','house',0,180,154,0)])
    with pytest.raises(ValueError,match='unique depth'):
        render([Object('house','house',112,180,154,0), Object('dog','dog',112,185,29,0)])
