"""CPU proof before any independent-vision batching is attempted on A100."""
import ast
import sys
from pathlib import Path
import pytest


def test_no_graph_training_attention_backend_or_context_concatenation():
    path=Path(__file__).resolve().parents[1]/'batched_visual_qwen_execution.py'
    text=path.read_text();tree=ast.parse(text)
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'CUDAGraph','compile','backward','step','load_state_dict','Linear','Embedding'}.intersection(calls)
    assert "ALL_ATTENTION_FUNCTIONS['sdpa']" in text and 'is_causal=False' in text
    assert 'len(set(lengths))!=1' in text and 'self.wrapped(hidden_states,cu_seqlens' in text
    runner=(path.parent/'measure_batched_visual.py').read_text()
    assert 'args.batch!=3' in runner and '<=900' in runner
    assert 'uniform.measure_repeat' in runner and 'range(0,558,3)' in runner
    assert 'normalized_learned_work(oa)' in runner and 'wa!=wb' in runner


@pytest.mark.skipif(sys.platform!='linux',reason='Native Torch CPU proof in original Linux environment')
def test_native_tiny_equal_independent_chunks_hidden_and_rope_exact():
    import torch
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    from LightGenV2.tasks.t06_video_quality_assessment.batched_visual_qwen_execution import BatchedVisualQwen
    torch.set_num_threads(1);torch.manual_seed(42)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,rope_scaling={'rope_type':'default','mrope_section':[1,1,2]}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,in_channels=3,patch_size=2,temporal_patch_size=2,spatial_merge_size=2,num_position_embeddings=16,deepstack_visual_indexes=[0]),image_token_id=89,video_token_id=91,vision_start_token_id=90,vision_end_token_id=92)
    cfg._attn_implementation='sdpa';native=Qwen3VLForConditionalGeneration(cfg).eval().requires_grad_(False)
    for batch in [2,3]:
        tokens=[1,90]+[91]*4+[92,90]+[91]*4+[92,2]
        inputs=dict(input_ids=torch.tensor([tokens]*batch),attention_mask=torch.ones(batch,len(tokens),dtype=torch.long),video_grid_thw=torch.tensor([[2,4,4]]*batch),pixel_values_videos=torch.randn(32*batch,24))
        inputs['input_ids'][1]=torch.tensor([0]+tokens[:-1]);inputs['attention_mask'][1,0]=0
        adapter=BatchedVisualQwen(native.model,inputs,inputs)
        with torch.inference_mode():
            a=native.model(**inputs,return_dict=True,use_cache=False);b=adapter(**inputs,return_dict=True,use_cache=False)
        assert torch.equal(a.last_hidden_state,b.last_hidden_state)
        assert torch.equal(a.rope_deltas,b.rope_deltas)
        assert {id(p) for p in adapter.parameters()}=={id(p) for p in native.model.parameters()}
