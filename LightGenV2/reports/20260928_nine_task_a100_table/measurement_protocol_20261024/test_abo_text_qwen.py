"""No Torch import, weights, CUDA or data mutations in protocol unit tests."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_abo_text_qwen import validate_calls
from measure_abo_text_qwen import original_source,ORIGINAL_SHA


def fixture():
    titles=[dict(product_id=str(i),title='real title '+str(i)) for i in range(100)]
    calls=[dict(index=i,query_index=i%100,pass_index=i//100,product_id=t['product_id'],title=t['title'],
        input_start=i*2.,prepare_end=i*2.+.2,model_start=i*2.+.3,cpu_result_return=i*2.+1,
        input_ids='[[1,2]]',attention_mask='[[1,1]]',input_ids_shape='[1,2]',valid_input_tokens=2,
        title_utf8_sha256=hashlib.sha256(t['title'].encode('utf8')).hexdigest()) for i,t in enumerate(titles*2)]
    return calls,titles


def test_complete_window_includes_all199_real_gaps():
    calls,titles=fixture()
    assert validate_calls(calls,titles)==(399.,199.,400)


def test_reject_skip_first_or_drop_second_pass():
    calls,titles=fixture()
    with pytest.raises(ValueError):validate_calls(calls[1:],titles)
    wrong=copy.deepcopy(calls);wrong[100]['product_id']='wrong'
    with pytest.raises(ValueError):validate_calls(wrong,titles)


def test_reject_overlap_and_fake_tokens():
    calls,titles=fixture();calls[1]['input_start']=.5
    with pytest.raises(ValueError):validate_calls(calls,titles)
    calls,titles=fixture();calls[0]['valid_input_tokens']=2048
    with pytest.raises(ValueError):validate_calls(calls,titles)


def test_reference_source_is_immutable_git_blob():
    assert hashlib.sha256(original_source().encode('utf8')).hexdigest()==ORIGINAL_SHA
    assert 'stable=True' in original_source()


def test_title_audit_checks_original_csvs_without_fake_gallery_images(tmp_path,monkeypatch):
    import csv
    import audit_abo_text_qwen as audit
    titles=[dict(product_id=str(i),title='title'+str(i),label=str(i)) for i in range(100)]
    train=[dict(sample_id='train'+str(i)) for i in range(4800)]
    test=[dict(sample_id=f'test{i}_{j}',product_id=str(i),label=str(i),image_path='not_an_online_title_input.jpg')
        for i in range(100) for j in range(24)]
    pins={}
    for name,rows in [('titles.csv',titles),('train.csv',train),('test.csv',test),('manifest.csv',test)]:
        with (tmp_path/name).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        pins[name]=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    monkeypatch.setattr(audit,'CSV_SHA',pins)
    assert audit.original_csv_contract(tmp_path)==(titles,train,test)
    (tmp_path/'titles.csv').write_text('changed',encoding='utf8')
    with pytest.raises(ValueError):audit.original_csv_contract(tmp_path)
