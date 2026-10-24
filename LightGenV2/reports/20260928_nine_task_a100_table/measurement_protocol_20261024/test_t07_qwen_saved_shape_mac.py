from pathlib import Path
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from t07_qwen_saved_shape_mac import request_terms,CONFIG


def test_dynamic_aspect_grid_and_original64D_gallery_dot():
    c=json.loads(CONFIG.read_text());row=dict(input_ids=str([[1]*68]),attention_mask=str([[1]*68]),
        image_grid_thw='[[1,16,10]]',pixel_values_shape='[160,1536]',input_ids_shape='[1,68]',valid_input_tokens='68')
    terms,shapes=request_terms(c,row)
    assert shapes['premerger_tokens']==160
    assert terms['GPU_FP32_gallery1600_descriptor64_dot']==102400
    assert not any('vocabulary' in k for k in terms)
    row['pixel_values_shape']='[196,1536]'
    with pytest.raises(ValueError):request_terms(c,row)
