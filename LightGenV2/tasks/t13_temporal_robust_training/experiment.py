"""Thin adapters around the byte-pinned teacher runtime; keep it unmodified."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace


@contextmanager
def detector_noise_evaluation(settings, *, scale: float):
    """Perturb only CCDs in eval; never enable dropout or router training."""
    if scale < 0:
        raise ValueError("Noise scale must be nonnegative")
    from LightGenV2.tasks.t06_video_quality_assessment.models import multivideo9x4 as optics
    original = optics._perturb_ccd
    noise_settings = replace(
        settings, ccd_noise_enabled=scale > 0,
        ccd_noise_mean_fraction=settings.ccd_noise_mean_fraction * scale,
        ccd_noise_std_fraction=settings.ccd_noise_std_fraction * scale,
        ccd_noise_min_fraction=settings.ccd_noise_min_fraction * scale,
        ccd_noise_max_fraction=settings.ccd_noise_max_fraction * scale,
        ccd_shot_photons_per_mean=settings.ccd_shot_photons_per_mean / max(scale * scale, 1e-12),
    )
    # The same settings are used for every group's testing, independent of training flag.
    optics._perturb_ccd = lambda clean, ignored, *, training: original(clean, noise_settings, training=scale > 0)
    try:
        yield
    finally:
        optics._perturb_ccd = original


def install_common_selection_evaluator(training_module, deployment_settings, *, seed: int):
    """Select all groups on the same 8um/eta validation physics, no camera noise.

    Nonpersistent propagator buffers/settings can be changed without changing
    learned tensor names or optimizer references. Restore them after evaluation.
    """
    import torch
    from LightGenV2.tasks.t06_video_quality_assessment.models.multivideo9x4 import _FullFieldBase
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.modeling import AngularSpectrum
    original = training_module.evaluate

    def evaluate_on_common_device(model, payload, settings, device, **kwargs):
        saved = []
        original_mode = model.training
        for module in model.modules():
            if isinstance(module, _FullFieldBase):
                saved.append((module, module.settings, module.propagation_size, module.propagation))
                module.settings = deployment_settings
                module.propagation_size = deployment_settings.propagation_canvas_size
                module.propagation = AngularSpectrum(
                    deployment_settings, size=module.propagation_size,
                    pixel_pitch_um=deployment_settings.modulator_pixel_pitch_um,
                ).to(device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                return original(model, payload, settings, device, **kwargs)
        finally:
            for module, old_settings, old_size, old_propagation in saved:
                module.settings, module.propagation_size, module.propagation = old_settings, old_size, old_propagation
            model.train(original_mode)

    training_module.evaluate = evaluate_on_common_device
    return original
