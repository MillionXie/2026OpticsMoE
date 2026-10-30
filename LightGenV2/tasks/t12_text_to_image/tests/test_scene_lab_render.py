from pathlib import Path


def test_physical_renderer_uses_supported_original_observer_not_a_generator_patch():
    source=(Path(__file__).parents[1]/'scene_lab_render.py').read_text()
    assert 'request_observer=observer' in source and 'render(args)' in source
    assert 'simulation_diagnostic=False' in source and "if tuple(self.order)!=STAGES" in source
    assert 'ideal is not None' in source and 'Physical input identity differs' in source
    assert '.raw_scene_features=' not in source and '.forward=' not in source
    assert "fuse_task_lora=True,unet_precision='bf16'" in source
    assert "ledger.session.get('scope')!='reviewed_physical'" in source
    assert "receipt['metadata']['frame_provenance']!='physical_linear_camera'" in source
