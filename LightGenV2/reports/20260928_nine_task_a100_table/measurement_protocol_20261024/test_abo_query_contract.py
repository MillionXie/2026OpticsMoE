"""Pure CPU workload/direction tests without importing Torch or reading images."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('abo_query_contract', Path(__file__).with_name('abo_query_contract.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture():
    titles = [dict(label=str(i), product_id=f'p{i}', title=f'title{i}') for i in range(100)]
    train = [dict(sample_id=f'train{i}') for i in range(4800)]
    test = [dict(sample_id=f'test{24*i+j}', product_id=f'p{i}', label=str(i)) for i in range(100) for j in range(24)]
    return titles, train, test


def test_image_queries_preserve_original_first200_not_balanced_subset():
    titles, train, test = fixture(); m.validate_rows(titles, train, test)
    calls = m.timed_calls('image_to_text', titles, test)
    assert [x['row']['sample_id'] for x in calls] == [x['sample_id'] for x in test[:200]]
    assert [x['row']['label'] for x in calls[:24]] == ['0']*24


def test_titles_are_two_complete_passes_with100_unique_queries():
    titles, _, test = fixture()
    calls = m.timed_calls('text_to_image', titles, test)
    assert len(calls) == 200
    assert [c['unique_query_index'] for c in calls] == list(range(100))*2
    assert [c['pass_index'] for c in calls] == [0]*100+[1]*100
    assert m.cost_contract('text_to_image')['online_optical_passes'] == 3
    assert m.cost_contract('text_to_image')['offline_candidate_optical_passes'] == 6


def test_frozen_baseline_prompts_and2048_not_t07_prefix():
    assert m.cost_contract('image_to_text')['baseline_descriptor_dimensions'] == 2048
    assert m.BASELINE_QUERY['text_to_image'] != m.DOCUMENT_INSTRUCTION
    assert m.BASELINE_QUERY['image_to_text'] != m.BASELINE_QUERY['text_to_image']
    with pytest.raises(ValueError): m.cost_contract('text_to_image_generation')


def test_mapping_or_overlap_changes_rejected():
    titles, train, test = fixture()
    test[0]['label'] = '1'
    with pytest.raises(ValueError): m.validate_rows(titles, train, test)
    titles, train, test = fixture(); train[0]['sample_id'] = test[0]['sample_id']
    with pytest.raises(ValueError): m.validate_rows(titles, train, test)
