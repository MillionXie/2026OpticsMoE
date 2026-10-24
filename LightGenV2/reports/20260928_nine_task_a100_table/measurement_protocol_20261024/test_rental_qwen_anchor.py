import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

path=Path(__file__).with_name('run_rental_qwen_anchor.py')
spec=importlib.util.spec_from_file_location('rental_anchor',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_platform_override_keeps_network_contract_and_original_config():
    original=dict(batch=1,profiles=dict(lab=dict(gpu_uuid='original',power_limit_w=250)),
                  explicit_warmup_forwards=0,resource_caps=[100,50])
    e=SimpleNamespace(config=lambda:original,source_map=lambda:{'original.py':'original'},sha=lambda _: 'wrapper')
    m.bind_platform(e)
    value=e.config()
    assert value['profiles']['lab']==dict(gpu_uuid=m.GPU,power_limit_w=400)
    assert value['batch']==1 and value['explicit_warmup_forwards']==0
    assert original['profiles']['lab']['power_limit_w']==250
    assert e.source_map()['original.py']=='original'


def test_predecessor_missing_partial_and_running_wait(tmp_path):
    assert not m.predecessor_ready(tmp_path)
    p=tmp_path/'suite_status.json';p.write_text('{')
    assert not m.predecessor_ready(tmp_path)
    p.write_text(json.dumps(dict(status='running')))
    assert not m.predecessor_ready(tmp_path)


def test_only_released_finished_predecessor_is_safe(tmp_path):
    import pytest
    p=tmp_path/'suite_status.json'
    p.write_text(json.dumps(dict(status='complete',mps_released=False)))
    with pytest.raises(RuntimeError):m.predecessor_ready(tmp_path)
    for state in ('complete','stopped'):
        p.write_text(json.dumps(dict(status=state,mps_released=True)))
        assert m.predecessor_ready(tmp_path)
