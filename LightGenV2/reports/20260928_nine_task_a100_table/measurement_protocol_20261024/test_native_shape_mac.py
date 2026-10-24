import json
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from native_shape_mac import native_mac,openmoji_standard_head_mac


def config():
    return dict(text_config=dict(hidden_size=8,num_attention_heads=4,num_key_value_heads=2,head_dim=2,num_hidden_layers=2,intermediate_size=12),
        vision_config=dict(hidden_size=4,intermediate_size=8,spatial_merge_size=2,deepstack_visual_indexes=[0],
            out_hidden_size=8,depth=2,in_channels=3,temporal_patch_size=2,patch_size=2))


def test_gqa_query_heads_and_no_vision_joint_frame_attention():
    a=native_mac(config(),3,[4,4]);b=native_mac(config(),3,[8])
    assert a['vision_QK_and_AV']*2==b['vision_QK_and_AV']
    assert a['language_QK_and_AV']==2*2*4*2*3*3
    assert a['language_QKVO']==2*3*(8*(4+2*2)*2+4*2*8)


def test_all_actual_mergers_including_deepstack_and_patch_conv3d():
    a=native_mac(config(),3,[4])
    assert a['actual_main_and_deepstack_mergers']==2*(16*16+16*8)
    assert a['patch_embedding']==4*3*2*4*4
    assert a['vision_two_linear_MLP']==2*4*2*4*8


def test_head_has_full_padded_position_and_actual_instruction_adapters():
    a=openmoji_standard_head_mac(4);b=openmoji_standard_head_mac(5)
    assert a['full_padded64_position_linear']==b['full_padded64_position_linear']==64*192
    assert b['text_adapter']-a['text_adapter']==2048*192
    assert a['category_edit_grid_heads']==36*192*18


@pytest.mark.parametrize('length,chunks',[(0,[196]),(64,[]),(64,[195])])
def test_missing_or_incompatible_shapes_are_not_invented(length,chunks):
    with pytest.raises(ValueError):native_mac(config(),length,chunks)
