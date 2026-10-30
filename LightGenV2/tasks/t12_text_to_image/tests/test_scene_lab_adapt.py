import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_adapt import match_train


def fixture():
    requests=[dict(sample_id=str(i),reference_sha256=str(i)*64,prompt='Prompt '+str(i),split='train') for i in range(2)]
    records=[dict(reference_sha256=r['reference_sha256'],prompt=r['prompt'],input_key='key'+r['sample_id'],split='train') for r in reversed(requests)]
    return requests,records


def test_pairs_exact_input_identity_not_row_order():
    requests,records=fixture()
    assert match_train(requests,records)==[('0','key0'),('1','key1')]


def test_no_test_changed_prompt_or_duplicate_teacher():
    requests,records=fixture()
    for changed in ([dict(requests[0],split='test'),requests[1]],[dict(requests[0],prompt='different'),requests[1]],requests+[requests[0]]):
        with pytest.raises(ValueError):match_train(changed,records)
    with pytest.raises(ValueError):match_train(requests,records+[records[0]])


def test_feature_export_is_optional_same_original_forward():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'scene_lab_native.py').read_text()
    assert 'export_features=False' in source and "if self.export_features:" in source
    assert source.count('self.qwen.model(**inputs')==1
    assert 'self.head(hidden.float(),native[0][None].float())' in source
