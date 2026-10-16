"""CPU contracts and tiny native-model exact equivalence, no GPU admission."""
import ast
import sys
from pathlib import Path
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import measure_residual_sync as m


def test_bounded_original_measurement_and_protected_b1():
    c=m.config()
    assert c['cases']=={'qwen':[2,3,4,6],'clip':[4,8,10]}
    assert (c['warmup_s'],c['measurement_s'],c['queue_depth'],c['repeats'])==(5,30,8,3)
    assert c['strict_actual_learned_operator_dtypes'] and c['full_TEST_videos']==558
    text=Path(m.__file__).read_text()
    assert 'uniform.measure_repeat' in text and 'ops_a!=ops_b' in text
    assert 'range(0,558,batch)' in text and 'prior_metadata.validation_guard' in text
    assert 'no_formal_table_replacement=True' in text


def test_no_new_learning_math_or_feature_cache_in_adapter():
    text=(Path(m.__file__).parent/'residual_sync_execution.py').read_text()
    tree=ast.parse(text)
    names={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'CUDAGraph','compile','backward','step','load_state_dict','scaled_dot_product_attention','Linear','Embedding'}.intersection(names)
    assert 'type(self.wrapped).forward(self,**kwargs)' in text
    assert 'hidden_states[batch_index,token_index,:].clone()+visual_embeds' in text
    assert 'hidden_states[batch_index,token_index,:]=local_this' in text
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IndexedQwenText')
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assert len(forward.body)==1 and isinstance(forward.body[0],ast.Return)
    assert 'non_blocking=True' in text


@pytest.mark.skipif(sys.platform!='linux',reason='Native Torch CPU test runs on original Linux environment')
def test_native_tiny_indexed_cpu_hidden_rope_and_parameter_identity():
    import torch
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    from LightGenV2.tasks.t06_video_quality_assessment.residual_sync_execution import IndexedHostQwen
    torch.set_num_threads(1);torch.manual_seed(42)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,rope_scaling={'rope_type':'default','mrope_section':[1,1,2]}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,in_channels=3,patch_size=2,temporal_patch_size=2,spatial_merge_size=2,num_position_embeddings=16,deepstack_visual_indexes=[0]),image_token_id=89,video_token_id=91,vision_start_token_id=90,vision_end_token_id=92)
    cfg._attn_implementation='sdpa'
    native=Qwen3VLForConditionalGeneration(cfg).eval().requires_grad_(False)
    for padding in [False,True]:
        tokens=[1,90]+[91]*4+[92,90]+[91]*4+[92,2]
        inputs=dict(input_ids=torch.tensor([tokens]*2),attention_mask=torch.ones(2,len(tokens),dtype=torch.long),video_grid_thw=torch.tensor([[2,4,4]]*2),pixel_values_videos=torch.randn(64,24))
        if padding:
            inputs['input_ids'][1]=torch.tensor([0]+tokens[:-1]);inputs['attention_mask'][1,0]=0
        adapter=IndexedHostQwen(native.model,inputs,inputs)
        with torch.inference_mode():
            a=native.model(**inputs,return_dict=True,use_cache=False)
            b=adapter(**inputs,return_dict=True,use_cache=False)
        assert torch.equal(a.last_hidden_state,b.last_hidden_state)
        assert torch.equal(a.rope_deltas,b.rope_deltas)
        assert {id(p) for p in adapter.parameters()}=={id(p) for p in native.model.parameters()}


@pytest.mark.skipif(sys.platform!='linux',reason='Native Torch CPU capture test on Linux')
def test_clip_capture_records_composite_matmul_quality_logits():
    import torch
    from types import SimpleNamespace
    class Visual(torch.nn.Module):
        def __init__(self):super().__init__();self.transformer=torch.nn.Identity()
        def forward(self,x):return self.transformer(x)
    visual=Visual();model=SimpleNamespace(visual=visual)
    head=torch.nn.Linear(2048,5,bias=False).eval().requires_grad_(False)
    inputs=torch.randn(16,512);scores=torch.arange(5).float()
    def compute(x):
        feature=torch.nn.functional.normalize(visual(x).float(),dim=-1).reshape(-1,2048).half().float()
        return (feature@head.weight.t()).softmax(-1)@scores
    with torch.inference_mode():
        result,ops=m.capture(compute,inputs,None,model,head,'clip',True)
    assert set(result)=={'hidden','image_features','quality_logits','mos'}
    assert result['quality_logits'].shape==(4,5)
    assert any(x['op'] in ['aten.mm.default','aten.matmul.default'] for x in ops)
