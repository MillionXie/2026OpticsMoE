from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dense_vision_qwen_saved_shape_mac import pose_head,saliency_head,request_terms,CONFIG,read


def settings():return dict(pose_head_mode='deconv',vision_hidden_size=1024,pose_projection_dim=128,pose_decoder_channels=[128,128],heatmap_size=56)


def test_transposed_convolution_uses_input_not_output_area():
    v=pose_head(settings())
    assert v['upsample14_to28']==14**2*128**2*16
    assert v['upsample28_to56']==4*v['upsample14_to28']
    assert v['joint_predictor']==56**2*128*14


def test_progressive_decoder_convolution_after_each_upsample():
    v=saliency_head()
    assert v['decoder_upsample28']==28**2*(128*9+128*96)
    assert v['decoder_upsample224']==224**2*(32*9+32*16)
    assert v['final_refine']==224**2*(16*9+16**2)


def test_full_visual_mergers_retained_but_no_language():
    c=read(CONFIG);v=request_terms(c,'LSP',settings())
    assert not any(k.startswith('language') for k in v)
    assert v['actual_main_and_deepstack_mergers']>0
    assert v['vision_QKV_and_output']==24*196*4*1024**2
    assert v['head.final_refine']==56**2*128**2*9


def test_wrong_old_bilinear_head_rejected():
    s=settings();s['pose_head_mode']='bilinear'
    with pytest.raises(ValueError):pose_head(s)
