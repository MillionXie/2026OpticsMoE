import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_eval import dev_requests


def rows():
    return [dict(sample_id=str(i),reference='ref'+str(i//8)+'.png',reference_sha256=format(i//8,'064x'),prompt='Prompt '+str(i%8),seed=i//8,category='lamp'if i<64 else'pillow',prompt_profile='two_sentence_natural_room_v4')for i in range(128)]


def test_complete_dev_keeps_seeds_pixels_prompts_and_role():
    source=rows();selected=dev_requests(source,[dict(reference_sha256='f'*64)])
    assert len(selected)==128 and all(r['split']=='dev'for r in selected)
    assert [(r['prompt'],r['seed'],r['reference_sha256'])for r in selected]==[(r['prompt'],r['seed'],r['reference_sha256'])for r in source]


def test_dev_not_an_arbitrary_subset_overlap_or_target_container():
    source=rows()
    with pytest.raises(ValueError):dev_requests(source[:16],[])
    with pytest.raises(ValueError):dev_requests(source,[dict(reference_sha256=source[0]['reference_sha256'])])
    source[0]['target']='forbidden.png'
    with pytest.raises(ValueError):dev_requests(source,[])
