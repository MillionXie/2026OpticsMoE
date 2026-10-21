import importlib
import sys
import numpy as np
import json
import csv
import pytest


def test_offline_metrics_do_not_require_torch_or_cuda():
    torch_preexisting = 'torch' in sys.modules
    module = importlib.import_module('LightGenV2.tasks.t06_video_quality_assessment.reports.audit_mps_batch_activity')
    assert torch_preexisting or 'torch' not in sys.modules
    r = module.metrics(np.array([1.,2.,3.,4.]),np.array([2.,4.,6.,8.]))
    assert abs(r['srcc']-1) < 1e-10 and abs(r['plcc']-1) < 1e-10
    assert abs(r['mae']-2.5) < 1e-10


@pytest.mark.parametrize('corrupt',[None,'time16_ms','energy16_j'])
def test_independent_overlap_integral_keeps_gaps_and_host_once(tmp_path,monkeypatch,corrupt):
    module=importlib.import_module('LightGenV2.tasks.t06_video_quality_assessment.reports.audit_mps_batch_activity')
    monkeypatch.setattr(module.experiment,'REPO',tmp_path)
    calls=[]
    for i,(a,b) in enumerate(module.experiment.batches(10)):
        calls.append(dict(videos=b-a,in_cost_subset=b<=200,preprocess_start=i*3.,
            first_block_start=i*3.+.25,gpu_output=i*3.+.75,legacy_host_ms=500.))
    seconds=20*.5+19*2.25
    report=dict(status='complete',mode='cost',model='clip',batch=10,cap=50,repeat=1,
        explicit_warmup_forwards=0,no_profiling_during_cost=True,visible_multiprocessors=54,
        power_limit_w=250,cost_videos=200,test_videos=558,
        cost=dict(time16_ms=seconds*80,energy16_j=(338.2+50)*seconds*.08))
    if corrupt:report['cost'][corrupt]+=1
    (tmp_path/'report.json').write_text(json.dumps(report))
    (tmp_path/'calls.json').write_text(json.dumps(calls))
    with (tmp_path/'telemetry.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['host_monotonic_s','watts']);writer.writeheader()
        writer.writerows(dict(host_monotonic_s=i*.25,watts=50) for i in range(670))
    if corrupt:
        with pytest.raises(AssertionError):module.independent_cost_replay(tmp_path)
    else:
        result=module.independent_cost_replay(tmp_path)
        assert result['cost_calls']==20 and result['last_test_batch']==8
        assert result['gpu_energy_j']==50*seconds
        assert result['host_energy_j']==338.2*seconds
