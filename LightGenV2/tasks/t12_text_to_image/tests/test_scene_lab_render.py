from pathlib import Path
import pytest


def test_physical_renderer_uses_supported_original_observer_not_a_generator_patch():
    source=(Path(__file__).parents[1]/'scene_lab_render.py').read_text()
    assert 'request_observer=observer' in source and 'render(args)' in source
    assert 'simulation_diagnostic=False' in source and "if tuple(self.order)!=STAGES" in source
    assert 'ideal is not None' in source and 'Physical input identity differs' in source
    assert '.raw_scene_features=' not in source and '.forward=' not in source
    assert "fuse_task_lora=True,unet_precision='bf16'" in source
    assert "ledger.session.get('scope')!='reviewed_physical'" in source
    assert "receipt['metadata']['frame_provenance']!='physical_linear_camera'" in source
    assert 'self.verified={stage:self.ledger.receipt(stage,self.active)for stage in STAGES}'in source


def test_observer_actual_callback_math_and_lineage_against_synthetic_fixture(tmp_path):
    import sys
    if sys.platform=='win32':pytest.skip('Run the numeric Torch adapter test in the designated compute runtime, not the desktop DLL environment')
    try:import torch
    except (ImportError,OSError):pytest.skip('Torch runtime unavailable locally; run this CPU check on the compute runtime')
    import numpy as np
    from LightGenV2.tasks.t12_text_to_image.tests.test_scene_lab_protocol import fixture,runner,accept
    from LightGenV2.tasks.t12_text_to_image.scene_lab_protocol import STAGES
    from LightGenV2.tasks.t12_text_to_image.scene_lab_render import MeasuredObserver
    ledger,dark,gain=fixture(tmp_path/'synthetic_only')
    # Production constructor must reject this non-physical fixture.
    with pytest.raises(ValueError):MeasuredObserver(ledger,'a'*64)
    for stage in STAGES:
        ledger.prepare(stage,runner)
        for request in ledger.requests:accept(ledger,stage,request,dark,gain)
    request=ledger.requests[0]
    # Deliberately bypass the physical-admission constructor FOR THIS UNIT TEST
    # to exercise the exact callback against an explicitly synthetic ledger.
    observer=MeasuredObserver.__new__(MeasuredObserver);observer.ledger=ledger;observer.active=request;observer.order=[];observer.prefix=[];observer.completed=0
    observer.verified={stage:ledger.receipt(stage,request)for stage in STAGES}
    amplitude=np.full((1,478,478),.2,np.float32)
    for stage in STAGES:
        value=observer.provider(stage,torch.from_numpy(amplitude),torch.full((1,478,478),1.7),None)
        np.testing.assert_array_equal(value.numpy(),np.ones_like(amplitude))
        amplitude=np.full_like(amplitude,.1+float(value.mean())*.01)
    observer.end({});assert observer.completed==1 and observer.active is None and observer.verified is None
    observer.active=request;observer.order=[];observer.prefix=[]
    observer.verified={stage:ledger.receipt(stage,request)for stage in STAGES}
    with pytest.raises(ValueError):observer.provider(STAGES[0],torch.full((1,478,478),.2),torch.full((1,478,478),1.8),None)
