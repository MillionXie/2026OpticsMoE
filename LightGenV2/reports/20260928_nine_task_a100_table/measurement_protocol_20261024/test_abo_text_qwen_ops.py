"""Independent Qwen main-term counting without loading a model or CUDA."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_abo_text_qwen_ops import independent_main_mac
from audit_abo_text_ops import independent_record_mac


def test_separate_projection_attention_search_terms():
    # h4, Q2*2, KV1*2, intermediate6, two layers, lengths3/5.
    # QKVO/token=48; MLP/token=72; QK/AV/sequence=8*S².
    assert independent_main_mac([3,5],4,2,6,2,1,2,candidates=7)==2*(120*8+8*34)+2*7*4


def test_invalid_or_missing_shapes_not_zero_cost():
    for lengths in ([],[0],[-1],[2.5]):
        with pytest.raises(ValueError):independent_main_mac(lengths,4,2,6,2,1,2)
    with pytest.raises(ValueError):independent_main_mac([3],4,2,6,3,2,2)


def test_independent_raw_operator_arithmetic_not_original_counter_code():
    def tensor(shape):return {'shape':shape}
    assert independent_record_mac({'operator':'aten.addmm.default','inputs':[tensor([4]),tensor([2,3]),tensor([3,4])]})==24
    assert independent_record_mac({'operator':'aten._scaled_dot_product_flash_attention.default',
        'inputs':[tensor([1,4,3,4]),tensor([1,2,3,4]),tensor([1,2,3,4])]})==288
    with pytest.raises(ValueError):independent_record_mac({'operator':'aten.not_a_registered_MAC.default','inputs':[]})
