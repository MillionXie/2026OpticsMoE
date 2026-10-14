"""No hardware, SSH, models or datasets are opened by these checks."""
import pytest
from LightGenV2.tasks.t06_video_quality_assessment.export_measured_examples import indexed, sha


def test_duplicates_are_not_silently_overwritten():
    with pytest.raises(ValueError,match='Duplicate'):
        indexed([{'id':'same'},{'id':'same'}],'id')


def test_join_is_by_identity_not_row_order():
    rows=[{'id':'b','prediction':2},{'id':'a','prediction':1}]
    assert indexed(rows,'id')['a']['prediction']==1


def test_sha_records_file_bytes(tmp_path):
    import hashlib
    p=tmp_path/'source.txt';p.write_bytes(b'original measurement')
    assert sha(p)==hashlib.sha256(b'original measurement').hexdigest()
