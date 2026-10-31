import json
import pytest
from LightGenV2.tasks.t17_router_classification.run_panel_d_queue import queue_profiles


def test_queue_restricts_dataset_k_selection_and_authorized_gpu(tmp_path):
    cfg=dict(panel_d_topk=True,architecture='LG',main_layers=2,top_k=1,dataset='kather2016',selection_source='original_TEST_development',authorized_gpu_uuids=['allowed'])
    p=tmp_path/'one.json';p.write_text(json.dumps(cfg));assert queue_profiles([p],'allowed')[0]['top_k']==1
    with pytest.raises(ValueError,match='GPU'):queue_profiles([p],'other')
    with pytest.raises(ValueError,match='unique'):queue_profiles([p,p],'allowed')
    cfg['main_layers']=4;p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='two-layer'):queue_profiles([p],'allowed')
