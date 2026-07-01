import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import build_model as build_dataset_model
from foundation_distillation.runtime import build_student
from same_input_multitask.scripts.train_same_input_multitask import build_model as build_same_input_model
from single_task.baselines.model_factory import build_model as build_single_model


def _config(model_type="learnable_route_moe"):
    return {
        "model": {"type": model_type, "num_experts": 9, "prompt_type": "complex_amplitude", "routing_type": "learnable"},
        "student": {"model_type": "feature_distilled_optical_moe", "num_experts": 9},
        "layout": {"geometry_profile": "fast120_520", "canvas_height": 520, "canvas_width": 520, "input_size": 120, "expert_size": 120, "expert_pitch": 150, "padding": 35, "prompt_aperture_size": 450},
        "optics": {"num_layers": 1, "global_fc_phase_mode": "center_window", "global_fc_phase_size": 450, "global_fc_padding_mode": "transparent", "distances_m": {"input_to_prompt": 0.01, "prompt_to_expert": 0.01, "inter_layer": 0.01, "layer5_to_fc": 0.01, "fc_to_detector": 0.01}},
        "prompt": {"mode": "complex_order_router", "train_amplitudes": True, "train_phase_biases": True},
        "detector": {"detector_size": 4, "layout": "grid"},
        "readout": {"type": "optical_only", "normalize_detector_energy": True, "input_norm": "none", "norm_affine": False, "hidden_layers": 0, "dropout": 0.0},
        "feature_detector": {"type": "grid_pool", "source_region": "camera_active_window", "grid_size": 30, "feature_dim": 900, "pooling": "sum"},
        "feature_preprocess": {"norm": "layernorm", "norm_affine": True, "activation": "gelu"},
        "classifier": {"input": "semantic_feature", "input_dim": "auto_teacher_dim", "hidden_dim": 8, "hidden_layers": 1, "activation": "gelu", "dropout": 0.0},
        "projector": {"type": "mlp", "input_dim": "auto_feature_dim", "output_dim": "auto_teacher_dim", "hidden_dim": 8, "hidden_layers": 1, "activation": "gelu", "dropout": 0.0, "output_l2_normalize": True},
        "regularization": {"phase_dropout": {"enabled": False}},
    }


def _assert_detector(intermediates):
    assert intermediates["detector_intensity"].shape == (1, 520, 520)


def test_fast_geometry_forwards_across_experiment_families():
    image = torch.rand(1, 1, 120, 120)
    with torch.inference_mode():
        model = build_single_model(_config(), 10)
        logits, intermediates = model(image, return_intermediates=True)
        assert logits.shape == (1, 10)
        _assert_detector(intermediates)
        del model
        gc.collect()

        tasks = ["mnist", "emnist_letters"]
        model = build_dataset_model(_config(), tasks, {"mnist": 10, "emnist_letters": 26})
        logits, intermediates = model(image, task_name="mnist", return_intermediates=True)
        assert logits.shape == (1, 10)
        _assert_detector(intermediates)
        del model
        gc.collect()

        tasks = ["shape", "scale"]
        model = build_same_input_model(_config(), tasks, {"shape": 3, "scale": 6})
        logits, intermediates = model(image, task_name="shape", return_intermediates=True)
        assert logits.shape == (1, 3)
        _assert_detector(intermediates)
        del model
        gc.collect()

        model = build_student(_config(), num_classes=10, teacher_feature_dim=32)
        logits, raw, processed, semantic, semantic_normalized, intermediates = model(
            image, return_intermediates=True
        )
        assert logits.shape == (1, 10)
        assert raw.shape == (1, 900)
        assert processed.shape == (1, 900)
        assert semantic.shape == (1, 32)
        assert semantic_normalized.shape == (1, 32)
        _assert_detector(intermediates)
