import pytest
from LightGenV2.tasks.t06_video_quality_assessment.rental_vlm_audit import close,digest


def test_tolerance_rejects_real_difference():
    close(1.,1.+1e-10)
    with pytest.raises(AssertionError):close(1.,1.001)


def test_nonfinite_rejected():
    with pytest.raises(AssertionError):close(float("nan"),1.)


def test_stream_digest(tmp_path):
    import hashlib
    path=tmp_path/"evidence";path.write_bytes(b"evidence")
    assert digest(path)==hashlib.sha256(b"evidence").hexdigest()
