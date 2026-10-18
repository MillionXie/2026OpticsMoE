import pytest
import csv
import json
from LightGenV2.tasks.t06_video_quality_assessment import rental_followup_audit as a

def test_tied_rank_and_pearson():
    assert a.ranks([3,1,1,4])==[3,1.5,1.5,4]
    a.near(a.correlation([1,2,3],[2,4,6]),1)
    a.near(a.correlation(a.ranks([1,2,3]),a.ranks([3,2,1])),-1)
    with pytest.raises(ValueError):a.near(.8,.7)


def point_fixture(folder,paced):
    folder.mkdir()
    repeats=[];groups=[];raw=[]
    if paced:groups.append(dict(phase='calibration',start_monotonic_s=-40,compute_end_monotonic_s=-25,end_monotonic_s=-10,forwards=8,duty=.5,requested_idle_s=15))
    for index in range(1,4):
        start=(index-1)*50.;end=start+30
        groups.append(dict(phase=f'warmup{index}',start_monotonic_s=start-5,compute_end_monotonic_s=start-2.5 if paced else start,end_monotonic_s=start,forwards=8,duty=.5 if paced else 1,requested_idle_s=2.5 if paced else 0))
        groups.append(dict(phase=f'measured{index}',start_monotonic_s=start,compute_end_monotonic_s=start+15 if paced else end,end_monotonic_s=end,forwards=8,duty=.5 if paced else 1,requested_idle_s=15 if paced else 0))
        r=dict(start_monotonic_s=start,end_monotonic_s=end,duration_s=30,samples=600,
               mean_gpu_utilization_percent=50,mean_board_power_w=100,forward_calls=8,output=[55],frozen_duty=.5 if paced else 1)
        if paced:r['service_cost']=dict(service_time_16_videos_ms=60000,estimated_energy_16_videos_j=26292,
                                      includes_all_imposed_wait=True,not_dataset_once_latency=True)
        repeats.append(r)
        for i in range(600):raw.append(dict(host_monotonic_s=start+.025+i*.05,utilization_percent=50,watts=100))
    report=dict(status='complete',paced=paced,batch=1,repeats=repeats,total_samples=1800,
                pooled_mean_utilization_percent=50,frozen_duty=.5 if paced else 1,target50_reached=paced)
    if paced:report['pooled_service_cost']=dict(service_time_16_videos_ms=60000,estimated_energy_16_videos_j=26292,measured_same_window_board_w=100)
    (folder/'report.json').write_text(json.dumps(report))
    (folder/'forward_groups.json').write_text(json.dumps(groups))
    with (folder/'telemetry.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    return report


@pytest.mark.parametrize('paced',[False,True])
def test_independent_whole_window_and_service_reconstruction(tmp_path,paced):
    folder=tmp_path/'point';point_fixture(folder,paced)
    result=a.audit_sequential_point(folder)
    assert result['samples']==1800 and result['utilization_percent']==50
    if paced:assert result['service_E16_j']==26292 and result['target50_reached']


def test_wrong_service_energy_rejected(tmp_path):
    folder=tmp_path/'point';report=point_fixture(folder,True)
    report['repeats'][0]['service_cost']['estimated_energy_16_videos_j']=10
    (folder/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):a.audit_sequential_point(folder)


def test_measured_source_uses_exact_git_blob_not_checkout_line_endings():
    from pathlib import Path
    repo=Path(a.__file__).resolve().parents[3]
    assert a.measured_source_sha(repo,'03ab757e8','LightGenV2/tasks/t06_video_quality_assessment/quality_token_common.py')=='bfeec599ffecb105338b6f185ff6d6cbff2730024d0b195704e72c2a7cd082ba'
    with pytest.raises(ValueError):a.measured_source_sha(repo,'HEAD','../outside.py')


@pytest.mark.parametrize('processes,accepted', [([],False),([['42',' GPU-test','100']],True),([['43','GPU-test','100']],False),([['42','GPU-other','100']],False)])
def test_process_identity_rejects_foreign_or_wrong_gpu(tmp_path,processes,accepted):
    (tmp_path/'identity.json').write_text(json.dumps(dict(pid=42,gpu_uuid='GPU-test')))
    (tmp_path/'process_audit.json').write_text(json.dumps([dict(time=1,processes=processes)]))
    if accepted:
        assert a.audit_process_identity(tmp_path)['foreign_observations']==0
    else:
        with pytest.raises(AssertionError):a.audit_process_identity(tmp_path)
