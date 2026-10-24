import pytest
from LightGenV2.tasks.t19_resnet20_baselines.prepare_final_plot_data import retrieval_samples,immutable_copy,immutable_json,sha


def test_query_AP_keeps_all_same_SKU_positives():
    pred=dict(queries=2,candidates=4,query_ids=['query-a','query-b'],query_skus=['a','b'],
              candidate_ids=['a1','b1','a2','b2'],candidate_skus=['a','b','a','b'],
              order=[[1,0,2,3],[1,3,0,2]],first_positive_rank=[2,1])
    _,rows=retrieval_samples(pred)
    assert rows[0][5:8]==[0,1,1]
    assert rows[0][8]==.5
    assert rows[0][9]==pytest.approx((.5+2/3)/2)
    assert rows[1][9:]==[1,2]


def test_archive_preserves_original_and_refuses_overwrite(tmp_path,monkeypatch):
    import LightGenV2.tasks.t19_resnet20_baselines.prepare_final_plot_data as export
    monkeypatch.setattr(export,'TASK',tmp_path)
    original=tmp_path/'original.pt';original.write_bytes(b'original saved checkpoint')
    frozen=tmp_path/'archive/last.pt';immutable_copy(original,frozen)
    assert sha(original)==sha(frozen)
    record=tmp_path/'record.json';immutable_json(record,dict(value=1));immutable_json(record,dict(value=1))
    with pytest.raises(ValueError):immutable_json(record,dict(value=2))
    original.write_bytes(b'different')
    with pytest.raises(ValueError):immutable_copy(original,frozen)
