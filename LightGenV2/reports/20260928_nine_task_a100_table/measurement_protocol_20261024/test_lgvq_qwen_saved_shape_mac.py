import hashlib
import json
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from test_native_shape_mac import config
from lgvq_qwen_saved_shape_mac import quality_head_mac,request_mac,group_paths


def test_full_vocabulary_is_only_last_pooled_vector_then_five_trained_rows():
    c=config();c['text_config']['vocab_size']=20
    assert quality_head_mac(c)==dict(native_full_vocabulary_last_pooled_vector=160,
        trained_five_quality_rows=40,quality_probability_score_dot=5)


def evidence():
    small=dict(input_ids=[[1,2,3]],attention_mask=[[1,1,1]],video_grid_thw=[[2,2,2]])
    return dict(sample_id='unit_fixture_not_measurement',small_cpu_values=small,
        small_values_sha256=hashlib.sha256(json.dumps(small,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        tensors={k:dict(shape=s) for k,s in dict(input_ids=[1,3],attention_mask=[1,3],video_grid_thw=[1,3],pixel_values_videos=[8,24]).items()},
        effective_input_tokens=3)


def test_actual_video_time_chunks_and_missing_shape_are_not_guessed():
    c=config();c['text_config']['vocab_size']=20;e=evidence();r=request_mac(c,e)
    assert r['actual_attention_chunks']==[4,4] and r['MAC_terms']['vision_QK_and_AV']==2*2*4*(16+16)
    e['tensors']['pixel_values_videos']['shape']=[4,24]
    with pytest.raises(ValueError):request_mac(c,e)


def test_original_retained_points_and_supplementary_platform_stay_separate():
    groups=group_paths()
    assert len(groups)==3 and all(len(v)==3 for v in groups.values())
    assert 'qwen100_repaired01' in str(groups['laboratory40GB_cap100'][0])
    assert all('rental_qwen_anchor_20261025_02' in str(p) for p in groups['supplementary80GB_cap100'])
