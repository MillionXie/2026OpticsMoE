"""Independent Qwen main-term counting without loading a model or CUDA."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_abo_text_qwen_ops import independent_main_mac


def test_separate_projection_attention_search_terms():
    # h4, Q2*2, KV1*2, intermediate6, two layers, lengths3/5.
    # QKVO/token=48; MLP/token=72; QK/AV/sequence=8*S².
    assert independent_main_mac([3,5],4,2,6,2,1,2,candidates=7)==2*(120*8+8*34)+2*7*4


def test_invalid_or_missing_shapes_not_zero_cost():
    for lengths in ([],[0],[-1],[2.5]):
        with pytest.raises(ValueError):independent_main_mac(lengths,4,2,6,2,1,2)
    with pytest.raises(ValueError):independent_main_mac([3],4,2,6,3,2,2)
