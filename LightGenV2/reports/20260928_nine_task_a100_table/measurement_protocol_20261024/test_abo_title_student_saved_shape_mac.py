from collections import Counter
import hashlib
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from abo_title_student_saved_shape_mac import call_mac,expected_counts,validate_counts,bind_source_bytes


def weights():
    d={}
    for name,shape in {'core.input_adapter.weight':[192,2048],'core.output_adapter.weight':[2048,192],
        'core.optical_branch.core.input_adapter.weight':[224,192],
        'core.optical_branch.core.output_adapter.weight':[192,224],
        'core.optical_branch.expert_output_adapter.weight':[192,224]}.items():d[name]={'shape':shape}
    for i in (0,1):
        for name,shape in {'token_depthwise.weight':[192,1,5],'token_pointwise.weight':[192,192],
            'mlp.0.weight':[96,192],'mlp.3.weight':[192,96]}.items():d[f'core.blocks.{i}.{name}']={'shape':shape}
    return d,{'projection.weight':{'shape':[64,384]}}


@pytest.mark.parametrize('length',[1,53,224])
def test_original_two_stage_adapter_and_causal_conv_counts(length):
    d,h=weights();r=call_mac(length,d,h,192*length)
    assert r['selected_serial_MAC']==length*(3*2048*192+4*224*192)+384*64+2400*64
    assert r['parallel_residual_MAC']==2*length*(192*5+192*192+2*192*96)
    assert r['actual_MAC']==r['selected_serial_MAC']+r['parallel_residual_MAC']+192*length


def test_exact_occurrences_no_vision_no_merger():
    counts=Counter(expected_counts());validate_counts(counts)
    counts['language.output_adapter']=1
    with pytest.raises(ValueError):validate_counts(counts)
    counts=Counter(expected_counts());counts['bridge.native_merger']=1
    with pytest.raises(ValueError):validate_counts(counts)


@pytest.mark.parametrize('length,rope',[(0,0),(225,225*192),(53,53*64)])
def test_unknown_length_or_native_position_is_not_zero(length,rope):
    d,h=weights()
    with pytest.raises(ValueError):call_mac(length,d,h,rope)


def test_original_git_blob_binding_only_accepts_checkout_newlines():
    original=b'x = 1\ny = 2\n';pin=hashlib.sha256(original).hexdigest()
    bind_source_bytes(original,original.replace(b'\n',b'\r\n'),pin)
    with pytest.raises(ValueError):bind_source_bytes(original,b'x = 2\ny = 2\n',pin)
    with pytest.raises(ValueError):bind_source_bytes(b'x = 3\n',original,pin)
