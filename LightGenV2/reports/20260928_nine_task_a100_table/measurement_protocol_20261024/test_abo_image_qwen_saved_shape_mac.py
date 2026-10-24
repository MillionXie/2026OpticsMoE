import json
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from abo_image_qwen_saved_shape_mac import request_terms
from test_native_shape_mac import config


def fixture():
    return dict(input_ids='[[1,2,3]]',attention_mask='[[1,1,1]]',valid_input_tokens='3',
        image_grid_thw='[[1,2,2]]',input_shapes=str(dict(input_ids=[1,3],attention_mask=[1,3],
            image_grid_thw=[1,3],pixel_values=[4,24])))


def test_original_graph_counts_search_not_unused_lm_head():
    terms,shape=request_terms(config(),fixture())
    assert terms['original_FP32_100_title_search']==100*8
    assert 'lm_head' not in terms and shape['premerger_tokens']==4
    assert terms['language_QK_and_AV']==2*2*4*2*3*3


@pytest.mark.parametrize('key,value',[
    ('attention_mask','[[0,1,1]]'),('valid_input_tokens','4'),
    ('image_grid_thw','[[2,2,2]]'),('input_shapes',str(dict(input_ids=[1,3],attention_mask=[1,3],
        image_grid_thw=[1,3],pixel_values=[8,24])))])
def test_missing_or_inconsistent_record_is_not_invented(key,value):
    row=fixture();row[key]=value
    with pytest.raises(ValueError):request_terms(config(),row)
