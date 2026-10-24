from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lgvq_spatial_qwen_saved_shape_mac import validate,HEAD_SHA,CONFIG_SHA,WEIGHT_SHA,MANIFEST_SHA,REFERENCE_SHA,PROMPT


def report():
    return dict(status='complete',system='qwen',task='LGVQ spatial',repeat=1,batch=1,measured_videos=200,
        checkpoint_sha256=HEAD_SHA,manifest_sha256=MANIFEST_SHA,assets={'config.json':CONFIG_SHA,'model.safetensors':WEIGHT_SHA},
        reference_sha256=REFERENCE_SHA,prompt=PROMPT,explicit_warmup_forwards=0,original_full_vocabulary_projection=True,
        processed_frame_size=[448,448],quality_readout_dtypes=['torch.float32'],gpu_uuid='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55',power_limit_w=250)


def test_original_spatial_not_temporal_prompt_or_head():
    v=report();validate(v,1);v['prompt']='temporal'
    with pytest.raises(ValueError):validate(v,1)


@pytest.mark.parametrize('key,value',[('checkpoint_sha256','other'),('original_full_vocabulary_projection',False),('power_limit_w',400)])
def test_reject_changed_actual_graph_or_physical_platform(key,value):
    v=report();v[key]=value
    with pytest.raises(ValueError):validate(v,1)
