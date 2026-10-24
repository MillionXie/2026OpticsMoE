from pathlib import Path
import sys
import ast
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from rental_original_shape_mac import language_mac,defaults


def config():
    return dict(hidden_size=8,num_attention_heads=2,kv_lora_rank=4,qk_nope_head_dim=2,qk_rope_head_dim=2,
        v_head_dim=2,num_hidden_layers=3,q_lora_rank=None,first_k_dense_replace=1,moe_layer_freq=1,
        intermediate_size=12,num_experts_per_tok=2,moe_intermediate_size=3,n_shared_experts=1,n_routed_experts=4,vocab_size=20)


def test_absorbed_MLA_is_not_expanded_KV_projection():
    terms,padding=language_mac(config(),5,True)
    assert 'language_expanded_KV_B' not in terms and padding==0
    assert terms['language_Q_absorb']==3*2*5*2*4
    assert terms['language_compressed_QK']==3*2*25*(2+4)
    assert terms['language_compressed_AV']==3*2*25*4
    assert terms['language_output_absorb']==3*2*5*4*2


def test_FA2_padding_is_explicit_and_logical_alternative_remains_distinct():
    terms,padding=language_mac(config(),5,False)
    assert terms['language_expanded_KV_B']==3*5*4*2*(2+2)
    assert terms['language_AV_actual_padded']==3*2*25*4
    assert padding==3*2*25*(4-2)


def test_active_experts_and_shared_experts_not_total_parameter_capacity():
    terms,_=language_mac(config(),5,True)
    assert terms['language_dense_SwiGLU']==5*3*8*12
    assert terms['language_routed_expert_SwiGLU']==2*5*2*3*8*3
    assert terms['language_shared_expert_SwiGLU']==2*5*3*8*3
    assert terms['native_full_vocabulary_last_vector']==8*20
    assert terms['trained_five_quality_rows']==8*5


def test_real_constructor_defaults_not_ambiguous_docstring_defaults():
    tree=ast.parse('class C:\n token_pooling:bool=False\n def __init__(self,num_hidden_layers=30,**kw):pass')
    assert defaults(tree,'C')==dict(token_pooling=False,num_hidden_layers=30)
    c=config();c['q_lora_rank']=2
    with pytest.raises(ValueError):language_mac(c,5,True)
