"""CPU contracts; native tiny-model equality runs only where Torch is installed."""
import ast
import json
import sys
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[1]


def test_bounded_protocol_and_protected_b1():
    c = json.loads((TASK/'configs/baselines/qwen_host_metadata_20261016.json').read_text())
    assert c['batches'] == [2,3,4,6,8]
    assert c['repeats']==3 and c['warmup_seconds']==5 and c['measurement_seconds']==30 and c['queue_depth']==8
    assert c['strict_hidden_logits_mos_equality'] and c['no_B1_execution_or_replacement']
    assert c['maximum_suite_seconds']==3600 and c['maximum_initial_idle_wait_seconds']==1800


def test_no_model_math_copy_or_feature_cache():
    text=(TASK/'qwen_host_metadata_execution.py').read_text()
    tree=ast.parse(text)
    calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
    forbidden={'Linear','Embedding','scaled_dot_product_attention','compile','CUDAGraph','backward','step','load_state_dict','eval'}
    assert not any(isinstance(n,ast.Attribute) and n.attr in forbidden for n in calls)
    assert 'return type(self.wrapped).forward(self, **inputs)' in text
    assert 'positions.to(input_ids.device)' in text
    assert 'self.wrapped.get_rope_index(' in text


@pytest.mark.skipif(sys.platform != 'linux', reason='Run native Torch equality in the original Linux server environment only')
def test_native_tiny_cpu_forward_and_metadata_parity():
    torch=pytest.importorskip('torch')
    pytest.importorskip('transformers.models.qwen3_vl.modeling_qwen3_vl')
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from LightGenV2.tasks.t06_video_quality_assessment.qwen_host_metadata_execution import HostMetadataQwenModel,validate_metadata_unchanged
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,rope_scaling={'rope_type':'default','mrope_section':[1,1,2]}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,in_channels=3,patch_size=2,temporal_patch_size=2,spatial_merge_size=2,num_position_embeddings=16,deepstack_visual_indexes=[0]),
        image_token_id=89,video_token_id=91,vision_start_token_id=90,vision_end_token_id=92)
    cfg._attn_implementation='sdpa'
    native=Qwen3VLForConditionalGeneration(cfg).eval().requires_grad_(False)
    for batch in [2,3]:
        tokens=[1,90]+[91]*4+[92,90]+[91]*4+[92,2]
        inputs=dict(input_ids=torch.tensor([tokens]*batch),attention_mask=torch.ones(batch,len(tokens),dtype=torch.long),video_grid_thw=torch.tensor([[2,4,4]]*batch),pixel_values_videos=torch.randn(32*batch,24))
        opt=HostMetadataQwenModel(native.model,inputs,inputs)
        with torch.inference_mode():
            a=native.model(**inputs,return_dict=True,use_cache=False)
            b=opt(**inputs,return_dict=True,use_cache=False)
        assert torch.equal(a.last_hidden_state,b.last_hidden_state)
        assert torch.equal(a.rope_deltas,b.rope_deltas)
        assert validate_metadata_unchanged(opt,inputs)
        assert {id(p) for p in opt.parameters()}=={id(p) for p in native.model.parameters()}
        with pytest.raises(ValueError,match='identity'):
            opt(**dict(inputs,input_ids=inputs['input_ids'].clone()),use_cache=False)
