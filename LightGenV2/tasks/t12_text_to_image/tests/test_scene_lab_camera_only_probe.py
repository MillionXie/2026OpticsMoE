from datetime import datetime,timezone,timedelta
import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_camera_only_probe import validate_holder


def test_separate_camera_requires_live_explicit_SLM_only_owner():
    now=datetime.now(timezone.utc);r=dict(SLMs_held=True,camera_SDK_opened=False,deadline_UTC=(now+timedelta(seconds=120)).isoformat());lock=dict(scope='SLMs_only',camera_SDK_owned=False)
    validate_holder(r,lock,now)
    with pytest.raises(ValueError):validate_holder(dict(r,camera_SDK_opened=True),lock,now)
    with pytest.raises(ValueError):validate_holder(r,dict(lock,scope='all_devices'),now)
    with pytest.raises(ValueError):validate_holder(dict(r,deadline_UTC=(now+timedelta(seconds=5)).isoformat()),lock,now)
