"""Tensor-name identity resolution without importing Torch or loading assets."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bind_lgvq_front import native_key,vision_config_at_original_runtime_dtype,METADATA_PIN,PROMPT_PIN,TEMPORAL_PT


def test_only_original_native_frontend_keys():
    keys=['model.visual.patch_embed.proj.weight','model.visual.pos_embed.weight',
        'model.language_model.embed_tokens.weight','lm_head.weight','model.language_model.layers.0.self_attn.q_proj.weight']
    assert native_key(keys,'vision','patch_embed.proj.weight')==keys[0]
    assert native_key(keys,'vision','pos_embed.weight')==keys[1]
    assert native_key(keys,'language','weight')==keys[2]


def test_missing_or_ambiguous_tensor_rejected():
    with pytest.raises(ValueError):native_key(['lm_head.weight'],'language','weight')
    with pytest.raises(ValueError):native_key(['a.embed_tokens.weight','b.embed_tokens.weight'],'language','weight')
    for pin in (METADATA_PIN,PROMPT_PIN,TEMPORAL_PT):assert len(pin)==64 and not set(pin)-set('0123456789abcdef')


def test_runtime_dtype_binding_never_ignores_other_architecture_fields():
    old={'dtype':'bfloat16','hidden_size':1024}
    assert vision_config_at_original_runtime_dtype({'dtype':None,'hidden_size':1024},old)==old
    assert vision_config_at_original_runtime_dtype(old,old)==old
    for current in ({'dtype':'float32','hidden_size':1024},{'dtype':None,'hidden_size':2048}):
        with pytest.raises(ValueError):vision_config_at_original_runtime_dtype(current,old)
