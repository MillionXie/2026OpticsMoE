"""Persistent display changes ownership only, not phase bytes or calibration."""
import sys
from types import SimpleNamespace
from LightGenV2.tasks.t12_text_to_image.scene_lab_capture import PersistentPhaseBridge


def test_persistent_bridge_uses_confirmed_owner_without_display_relayout(monkeypatch):
    calls=[]
    class Owner:
        def __init__(self,link,black,lens):calls.append((link,black,lens));self.info={'mock':True}
        def __enter__(self):return self
        def show(self,path):calls.append(path);return {'path':path}
        def __exit__(self,*args):calls.append('closed')
    monkeypatch.setitem(sys.modules,'LightGenV2.hardware_common.shs.phase_owner',SimpleNamespace(PhaseOwner=Owner))
    with PersistentPhaseBridge('sdk','lut','flat')as bridge:
        assert bridge.show('exact_native_BMP',pump=lambda:None)=={'path':'exact_native_BMP'}
        assert bridge.info=={'mock':True}
    link,black,lens=calls[0]
    assert black==lens=='flat'and not link['phase_display_align_top']and link['phase_startup_cycles']==0
    assert link['phase_lut']=='lut'and link['phase_pixel_format']=='rgba'
    assert calls[1:]==['exact_native_BMP','closed']
