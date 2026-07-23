from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9 import (
    TASK_PROMPTS,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.datasets import (
    load_spaq,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.features import (
    multimodal_forward_features,
    pool_answer_hidden_state,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.modeling import (
    build_head,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.optics.geometry import (
    MoEGeometry,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.optics.moe import (
    FullPlaneReadout,
    HomogeneousMoEOpticalCore,
    LanguageDeepStackHomogeneousMoE,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.optics.physical import (
    AngularSpectrumPropagator,
    PhaseLayer,
    SquareDetectionLayerNormReload,
    aperture_linear_indices,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.optics.replacement import (
    DeepStackMultimodalReplacement,
    VisionNativeAttentionPrelude,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.optics.router import (
    ElectronicAmplitudeRouter,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.processor_cache import (
    ProcessorCacheStore,
    collate_processor_samples,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.settings import (
    load_settings,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9.teacher_cache import (
    TeacherCacheStore,
)


ROOT = Path(
    "experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9"
)
CONFIGS = ROOT / "configs"


@pytest.mark.parametrize(
    ("filename", "task"),
    [
        ("spaq_mos.json", "MOS"),
        ("spaq_brightness.json", "Brightness"),
        ("spaq_colorfulness.json", "Colorfulness"),
        ("spaq_contrast.json", "Contrast"),
    ],
)
def test_all_four_single_attribute_configs(filename: str, task: str) -> None:
    settings = load_settings(CONFIGS / filename)
    assert settings.task_name == task
    assert settings.classification_prompt == TASK_PROMPTS[task]
    assert settings.student_language_mode == "optical_moe"
    assert settings.native_pre_attention_enabled
    assert settings.native_pre_attention_trainable
    assert not settings.native_pre_attention_initialize_from_teacher
    assert settings.attention_learning_rate == pytest.approx(1e-4)
    assert settings.transformer_residual_enabled
    assert settings.router_implementation == "electronic_amplitude_topk"
    assert settings.amplitude_phase_relay == "ideal_4f_identity"
    assert settings.detector_layernorm_scope == "per_token"
    assert settings.cpu_threads == 4
    assert settings.cpu_interop_threads == 1
    assert settings.teacher_cache_lru_shards == 128
    assert (
        settings.expert_interlayer_distance_m,
        settings.last_expert_to_global_distance_m,
        settings.global_to_detector_distance_m,
    ) == (0.1, 0.1, 0.1)


@pytest.mark.parametrize("task", ["MOS", "Brightness", "Colorfulness", "Contrast"])
def test_dataset_supports_every_attribute_and_rgb(tmp_path: Path, task: str) -> None:
    root = tmp_path / "SPAQ"
    images = root / "images"
    images.mkdir(parents=True)
    rows = ["Image name,MOS,Brightness,Colorfulness,Contrast"]
    for index in range(10):
        name = f"i{index}.jpg"
        Image.new("RGB", (8, 8), (index, 2, 3)).save(images / name)
        rows.append(f"{name},{50 + index},{40 + index},{30 + index},{20 + index}")
    (root / "scores.csv").write_text("\n".join(rows), encoding="utf-8")
    config = tmp_path / f"{task}.json"
    config.write_text(
        json.dumps(
            {
                "config_version": 3,
                "dataset": "spaq_single_attribute",
                "task_name": task,
                "data_root": str(root),
                "download": False,
                "output_dir": str(tmp_path / "run"),
                "classification_prompt": TASK_PROMPTS[task],
            }
        ),
        encoding="utf-8",
    )
    bundle = load_spaq(load_settings(config))
    image, target = bundle.train[0]
    assert image.mode == "RGB"
    assert 0 <= target <= 1
    assert bundle.metadata["task"] == task


def _encoder(hidden_size: int = 8, max_tokens: int = 120) -> HomogeneousMoEOpticalCore:
    module = HomogeneousMoEOpticalCore.__new__(HomogeneousMoEOpticalCore)
    nn.Module.__init__(module)
    module.hidden_size = hidden_size
    module.max_tokens = max_tokens
    module.geometry = MoEGeometry()
    module.input_adapter = nn.Linear(hidden_size, 120)
    module.input_norm = nn.LayerNorm(120)
    module.nonnegative = nn.Softplus()
    module.amplitude_slm_weight_domain = "amplitude"
    module.amplitude_slm_input_normalization = "none"
    module.amplitude_phase_relay = "ideal_4f_identity"
    module.register_buffer(
        "expert_canvas_indices",
        aperture_linear_indices(module.geometry.canvas_size, module.geometry.expert_apertures),
        persistent=False,
    )
    module.last_input_fields = None
    module.last_routing = {}
    module.last_amplitude_slm_canvas = None
    module.last_stage_fields = []
    return module


def test_token_row_mapping_is_nonnegative_and_zero_padded() -> None:
    encoder = _encoder()
    field = encoder.encode_groups([torch.randn(60, 8)])
    assert field.shape == (1, 120, 120)
    assert torch.all(field >= 0)
    assert torch.count_nonzero(field[:, 60:]) == 0
    with pytest.raises(RuntimeError, match="visual token count 121"):
        encoder.encode_groups([torch.randn(121, 8)])


class _FixedElectronicRouter(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_weights", weights)
        self.router = nn.Identity()

    def forward(self, fields: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = self.fixed_weights.expand(len(fields), -1)
        selected = weights > 0
        return {
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": selected.nonzero()[:, 1].reshape(len(fields), -1),
            "balance_loss": fields.new_zeros(()),
            "importance_loss": fields.new_zeros(()),
            "phase_prompt_used": False,
        }


def test_electronic_router_directly_loads_weighted_amplitude_copies() -> None:
    encoder = _encoder()
    weights = torch.tensor([[0.0, 0.2, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0, 0.0]])
    encoder.router = _FixedElectronicRouter(weights)
    source = torch.rand(1, 120, 120, requires_grad=True)
    canvas, routing = encoder.begin(source)
    assert canvas.dtype == torch.complex64
    assert routing["phase_prompt_used"] is False
    assert "prompt_phase" not in routing and "transmission" not in routing
    for index, aperture in enumerate(encoder.geometry.expert_apertures):
        crop = canvas.real[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
        assert torch.allclose(crop, source * weights[:, index, None, None])
    canvas.real.sum().backward()
    assert source.grad is not None and torch.count_nonzero(source.grad)


def test_real_electronic_router_is_sparse_balanced_and_differentiable() -> None:
    router = ElectronicAmplitudeRouter(MoEGeometry(), 3, 4, 1.0)
    fields = torch.rand(2, 120, 120, requires_grad=True)
    output = router(fields)
    assert output["weights"].shape == (2, 9)
    assert torch.equal((output["weights"] > 0).sum(1), torch.tensor([3, 3]))
    assert torch.allclose(output["weights"].sum(1), torch.ones(2))
    assert output["phase_prompt_used"] is False
    (output["weights"].square().sum() + output["balance_loss"]).backward()
    assert router.router.gate.weight.grad is not None


def test_power_domain_uses_sqrt_amplitude_scale() -> None:
    encoder = _encoder()
    encoder.amplitude_slm_weight_domain = "power"
    routing = {"weights": torch.tensor([[0.0, 0.25, 0.75] + [0.0] * 6])}
    assert torch.allclose(
        encoder._amplitude_scales(routing)[:, :3],
        torch.tensor([[0.0, 0.5, 0.75**0.5]]),
    )


def test_language_overflow_is_explicit() -> None:
    language = LanguageDeepStackHomogeneousMoE.__new__(LanguageDeepStackHomogeneousMoE)
    nn.Module.__init__(language)
    language.core = SimpleNamespace(max_tokens=120)
    with pytest.raises(RuntimeError, match="language sequence length 121"):
        language.set_attention_mask(torch.ones(1, 121))


def test_cached_multimodal_batch_padding_and_pixel_concatenation() -> None:
    rows = [
        {
            "input_ids": torch.tensor([1, 2]),
            "sequence_length": 2,
            "pixel_values": torch.ones(3, 4),
            "image_grid_thw": torch.tensor([1, 1, 3]),
        },
        {
            "input_ids": torch.tensor([3, 4, 5]),
            "sequence_length": 3,
            "pixel_values": torch.ones(2, 4),
            "image_grid_thw": torch.tensor([1, 1, 2]),
        },
    ]
    batch = collate_processor_samples(
        rows, {"padding_side": "left", "pad_token_id": 0}
    )
    assert batch["input_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batch["pixel_values"].shape == (5, 4)


def test_cached_collate_preserves_storage_dtype() -> None:
    rows = [
        {"input_ids": torch.tensor([1]), "sequence_length": 1,
         "pixel_values": torch.ones(2, 3, dtype=torch.float16),
         "image_grid_thw": torch.tensor([1, 1, 2])},
        {"input_ids": torch.tensor([2]), "sequence_length": 1,
         "pixel_values": torch.ones(1, 3, dtype=torch.float16),
         "image_grid_thw": torch.tensor([1, 1, 1])},
    ]
    batch = collate_processor_samples(rows, {"padding_side": "left", "pad_token_id": 0})
    assert batch["pixel_values"].dtype == torch.float16


def test_cache_batch_lookup_matches_single_lookup(tmp_path: Path) -> None:
    processor_shards = []
    teacher_shards = []
    for shard_number, start in enumerate((0, 2)):
        processor_path = tmp_path / f"processor_{shard_number}.pt"
        teacher_path = tmp_path / f"teacher_{shard_number}.pt"
        sample_indices = torch.arange(start, start + 2)
        torch.save({
            "sample_indices": sample_indices,
            "input_ids": [torch.tensor([index + 1]) for index in sample_indices],
            "pixel_values": [torch.full((1, 2), float(index), dtype=torch.float16) for index in sample_indices],
            "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
            "sequence_lengths": torch.ones(2, dtype=torch.long),
        }, processor_path)
        torch.save({
            "sample_indices": sample_indices,
            "targets": sample_indices.float(),
            "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
            "visual_token_counts": torch.ones(2, dtype=torch.long),
            "sequence_lengths": torch.ones(2, dtype=torch.long),
            "teacher_answer_hidden": sample_indices[:, None].half(),
            "teacher_vision_taps": [
                [torch.full((1, 2), float(index), dtype=torch.float16)]
                for index in sample_indices
            ],
        }, teacher_path)
        processor_shards.append({"path": str(processor_path), "count": 2})
        teacher_shards.append({"path": str(teacher_path), "count": 2})
    processor_manifest = tmp_path / "processor.pt"
    teacher_manifest = tmp_path / "teacher.pt"
    torch.save({"metadata": {"sample_count": 4}, "shards": processor_shards}, processor_manifest)
    torch.save({"metadata": {"sample_count": 4}, "shards": teacher_shards}, teacher_manifest)
    processor = ProcessorCacheStore(processor_manifest, max_cached_shards=4)
    teacher = TeacherCacheStore(teacher_manifest, max_cached_shards=4)
    order = [3, 0, 2, 1]
    assert [int(row["sample_index"]) for row in processor.get_many(order)] == order
    assert [int(row["sample_index"]) for row in teacher.get_many(order)] == order
    assert processor.stats()["shard_loads"] == 2
    assert teacher.stats()["shard_loads"] == 2
    # Both shards are now resident; a second batch must not deserialize again.
    processor.get_many(order); teacher.get_many(order)
    assert processor.stats()["shard_loads"] == 2
    assert teacher.stats()["shard_loads"] == 2


def test_multimodal_forward_uses_final_hook_without_all_hidden_states() -> None:
    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._spaq_electronic_router_optical_last_hidden = None
        def forward(self, **kwargs):
            assert kwargs["output_hidden_states"] is False
            self._spaq_electronic_router_optical_last_hidden = torch.ones(2, 3, 4)
            return SimpleNamespace(hidden_states=None)
    hidden = multimodal_forward_features(FakeModel(), {})
    assert hidden.shape == (2, 3, 4)


def test_answer_position_uses_last_valid_token() -> None:
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1]])
    answer, positions = pool_answer_hidden_state(hidden, mask)
    assert positions.tolist() == [2, 3]
    assert torch.equal(answer[0], hidden[0, 2])


class _KwargLinear(nn.Linear):
    def forward(self, input: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None, **_):
        value = input if input is not None else hidden_states
        assert value is not None
        return super().forward(value)


class _VisionBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _KwargLinear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, hidden_states, **_):
        return hidden_states


class _LanguageBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.self_attn = _KwargLinear(dim, dim, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(dim)

    def forward(self, hidden_states, **_):
        return hidden_states


class _FakeSurrogate(nn.Module):
    def __init__(self, stages: int = 5):
        super().__init__()
        self.core = SimpleNamespace(expert_layers=[None] * stages)
        self.weight = nn.Parameter(torch.ones(()))

    def set_attention_mask(self, mask):
        self.mask = mask

    def set_deepstack_injection_count(self, count):
        self.deepstack_injection_count = count


def _fake_replacement_settings(initialize_from_teacher: bool = False):
    return SimpleNamespace(
        student_language_mode="optical_moe",
        native_pre_attention_enabled=True,
        native_pre_attention_initialize_from_teacher=initialize_from_teacher,
        native_pre_attention_trainable=True,
        transformer_residual_enabled=True,
        vision_attention_source_layer=0,
        language_attention_source_layer=0,
    )


def _fake_qwen():
    visual = SimpleNamespace(
        blocks=nn.ModuleList([_VisionBlock() for _ in range(24)]),
        deepstack_visual_indexes=[5, 11, 17],
    )
    language = SimpleNamespace(
        layers=nn.ModuleList([_LanguageBlock() for _ in range(28)]),
        norm=nn.Identity(),
    )
    model = nn.Module()
    model.model = SimpleNamespace(visual=visual, language_model=language)
    return model


def test_attention_default_is_independent_trainable_and_residual_is_fixed_identity() -> None:
    torch.manual_seed(3)
    model = _fake_qwen()
    teacher_weight = model.model.visual.blocks[0].attn.weight.detach().clone()
    replacement = DeepStackMultimodalReplacement(
        model, _FakeSurrogate(), _FakeSurrogate(), _fake_replacement_settings(False)
    )
    assert not torch.equal(replacement.vision_pre_attention.attn.weight, teacher_weight)
    assert all(parameter.requires_grad for parameter in replacement.vision_pre_attention.parameters())
    specification = replacement.alignment_specification()
    assert specification["attention_initialization"] == "independent_random"
    assert specification["residual_identity_scale"] == 1.0
    assert not specification["residual_identity_scale_trainable"]
    replacement.close()


def test_attention_can_explicitly_inherit_teacher_weights() -> None:
    model = _fake_qwen()
    source = model.model.visual.blocks[0].attn.weight.detach().clone()
    prelude = VisionNativeAttentionPrelude(model.model.visual.blocks[0], True)
    assert torch.equal(prelude.attn.weight, source)


def test_replacement_maps_native_deepstack_taps() -> None:
    model = _fake_qwen()
    replacement = DeepStackMultimodalReplacement(
        model, _FakeSurrogate(), _FakeSurrogate(), _fake_replacement_settings()
    )
    replacement.use_student()
    assert [replacement.vision_blocks[i].slot for i in (5, 11, 17, 23)] == [0, 1, 2, 3]
    assert [replacement.language_layers[i].stage for i in range(5)] == list(range(5))
    replacement.close()


def test_small_text_regression_head_backward() -> None:
    settings = load_settings(CONFIGS / "spaq_mos.json")
    head = build_head(settings, 2048)
    prediction = head(torch.randn(4, 2048))
    assert prediction.shape == (4,)
    torch.nn.functional.smooth_l1_loss(
        prediction, torch.rand(4), beta=0.1
    ).backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_final_detector_per_token_normalization_preserves_gradient() -> None:
    settings = SimpleNamespace(
        canvas_size=16,
        detector_pool_kernel=2,
        detector_layernorm_scope="per_token",
        detector_layernorm_eps=1e-5,
        detector_layernorm_affine=False,
        detector_nonlinearity="relu",
    )
    readout = FullPlaneReadout(settings)
    phase = PhaseLayer(16, parameterization="unconstrained", init="small_normal")
    propagation = AngularSpectrumPropagator(
        wavelength_m=532e-9,
        pixel_size_m=16e-6,
        grid_size=16,
        distance_m=0.1,
    )
    torch.manual_seed(9)
    amplitude = torch.rand(2, 16, 16)
    values, _ = readout(propagation(phase(amplitude.to(torch.complex64))))
    assert values.shape == (2, 8, 8)
    assert torch.count_nonzero(values) > 0
    weighted_loss = (values * torch.linspace(0.1, 1.0, 8)[None, None, :]).mean()
    weighted_loss.backward()
    assert phase.raw_phase.grad is not None
    assert torch.isfinite(phase.raw_phase.grad).all()
    assert torch.count_nonzero(phase.raw_phase.grad) > 0


def test_vectorized_per_expert_detection_matches_reference() -> None:
    geometry = MoEGeometry()
    layer = SquareDetectionLayerNormReload(
        geometry.canvas_size,
        geometry.expert_apertures,
        1e-5,
        "relu",
        per_expert_enabled=True,
        elementwise_affine=True,
    )
    torch.manual_seed(11)
    field = torch.complex(
        torch.randn(2, 480, 480),
        torch.randn(2, 480, 480),
    )
    selected = torch.tensor(
        [
            [1, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    weights = torch.rand(2, 9)
    actual = layer(
        field, selected_experts=selected, routing_weights=weights
    ).real
    intensity = field.abs().square().float()
    expected = torch.zeros_like(intensity)
    for index, aperture in enumerate(geometry.expert_apertures):
        crop = intensity[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
        value = torch.nn.functional.layer_norm(crop, crop.shape[-2:], eps=1e-5)
        value = torch.relu(
            value * layer.affine_weight[index] + layer.affine_bias[index]
        )
        value = (
            value
            * weights[:, index, None, None]
            * selected[:, index, None, None]
        )
        expected[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = value
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
