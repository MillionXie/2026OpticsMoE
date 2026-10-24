from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from t07_student_saved_shape_mac import nn_terms


def state():
    shapes={'frontend.patch.weight':[1024,3,2,16,16],'frontend.merger_fc1.weight':[4096,4096],
        'frontend.merger_fc2.weight':[2048,4096],'vision.output_adapter.weight':[1024,192],'readout.projection.weight':[64,1152]}
    for mode,hidden in [('vision',1024),('language',2048)]:
        shapes[mode+'.input_adapter.weight']=[192,hidden]
        shapes[mode+'.optics.input_adapter.weight']=[224,192]
        for name in ('expert_output','global_output'):shapes[mode+'.optics.'+name+'.weight']=[192,224]
        for i in (0,1):
            prefix=f'{mode}.blocks.{i}.'
            shapes[prefix+'token_depthwise.weight']=[192,1,5,5] if mode=='vision' else [192,1,5]
            shapes[prefix+'token_pointwise.weight']=[192,192]
            shapes[prefix+'mlp.0.weight']=[384,192];shapes[prefix+'mlp.3.weight']=[192,384]
    return {k:dict(shape=v) for k,v in shapes.items()}


def test_one_merger_no_language_output_no_rope_and_separate_routing():
    v=nn_terms(state(),77)
    assert v['serial_NN_terms']['one_main_merger']==49*(4096**2+4096*2048)
    assert v['actual_NN_MAC']==v['charged_serial_NN_MAC']+v['parallel_residual_NN_MAC']
    assert v['other_executed_digital_dense_MAC']['GPU_router_two_dense_mask_contractions']==2*4*478**2
    assert v['serial_NN_terms']['spatial2x2_readout']==1152*64
    assert v['parallel_NN_terms']['vision.blocks.0.depthwise']==196*192*25


def test_no_later_bypass_or_invented_language_adapter():
    for name in ('late_rgb.project.weight','language.output_adapter.weight'):
        s=state();s[name]=dict(shape=[192,192])
        with pytest.raises(ValueError):nn_terms(s,77)
