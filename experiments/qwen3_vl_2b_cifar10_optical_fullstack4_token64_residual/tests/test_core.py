from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.datasets import (
    CIFAR10_CLASSES,
    stratified_split_indices,
)
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.optics import (
    LanguageOpticalStackSurrogate,
    VisionOpticalStackSurrogate,
)
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.optics import stacks
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.settings import load_settings
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.teacher_cache import (
    CACHED_TENSORS,
    expected_metadata,
)
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.training import (
    save_student_inference,
)


def make_stack(cls: type[nn.Module], hidden_size: int, residual_enabled: bool = True) -> nn.Module:
    module=cls(
        hidden_size=hidden_size,
        optical_dim=64,
        conversions=4,
        field_size=64,
        padding_size=128,
        wavelength_nm=532.0,
        pixel_pitch_um=8.0,
        distance_cm=5.0,
        amplitude_mask_enabled=False,
        phase_init="zeros",
        phase_init_std=0.02,
        residual_enabled=residual_enabled,
        identity_scale_init=1.0,
        modulated_scale_init=0.1,
        identity_scale_trainable=False,
        modulated_scale_trainable=True,
    )
    module.conversions=nn.ModuleList([nn.Identity() for _ in range(4)])
    return module


def test_config_parsing() -> None:
    path=Path(__file__).parents[1]/"configs"/"cifar10.json"
    settings=load_settings(path)
    assert settings.dataset=="cifar10"
    assert settings.download
    assert settings.classification_prompt.startswith("Classify:")
    assert settings.processor_min_pixels==settings.processor_max_pixels==16384
    assert settings.optical_dim==settings.optical_field_size==64
    assert settings.optical_padding_size==128 and settings.pixel_pitch_um==8
    assert settings.feature_batch_size==settings.inference_batch_size==settings.student_batch_size==1
    assert settings.optical_residual_enabled
    assert settings.optical_identity_scale_init==1.0
    assert settings.optical_modulated_scale_init==0.1
    assert not settings.optical_identity_scale_trainable
    assert settings.optical_modulated_scale_trainable


def test_cifar10_classes_and_stratified_split() -> None:
    assert CIFAR10_CLASSES == [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    dataset=SimpleNamespace(labels=[label for label in range(10) for _ in range(10)])
    train,validation=stratified_split_indices(dataset,0.1,42)
    assert len(train)==90 and len(validation)==10
    assert {dataset.labels[index] for index in validation}==set(range(10))


def test_vision_token_count_at_most_field_size() -> None:
    module=make_stack(VisionOpticalStackSurrogate,1024)
    hidden=torch.randn(60,1024)
    output=module(hidden,cu_seqlens=torch.tensor([0,60]))
    assert output.shape==(60,1024)
    assert module.last_token_counts==[60]


def test_vision_token_count_over_field_size_raises() -> None:
    module=make_stack(VisionOpticalStackSurrogate,1024)
    with pytest.raises(RuntimeError) as error:
        module(torch.randn(65,1024),cu_seqlens=torch.tensor([0,65]))
    message=str(error.value)
    assert "visual token count" in message
    assert "optical_field_size" in message
    assert "processor_max_pixels" in message


def test_language_sequence_at_most_field_size() -> None:
    module=make_stack(LanguageOpticalStackSurrogate,2048)
    hidden=torch.randn(1,50,2048)
    output=module(hidden,attention_mask=torch.ones(1,50))
    assert output.shape==(1,50,2048)


def test_language_sequence_over_field_size_raises() -> None:
    module=make_stack(LanguageOpticalStackSurrogate,2048)
    with pytest.raises(RuntimeError) as error:
        module(torch.randn(1,65,2048),attention_mask=torch.ones(1,65))
    message=str(error.value)
    assert "language sequence length" in message
    assert "optical_field_size" in message
    assert "prompt" in message
    assert "processor_max_pixels" in message


def test_zero_padding_and_valid_row_readout_without_interpolation() -> None:
    module=make_stack(VisionOpticalStackSurrogate,64,residual_enabled=False)
    with torch.no_grad():
        module.input_adapter.weight.copy_(torch.eye(64)); module.input_adapter.bias.zero_()
        module.output_adapter.weight.copy_(torch.eye(64)); module.output_adapter.bias.zero_()
    hidden=torch.randn(7,64)
    output=module(hidden,cu_seqlens=torch.tensor([0,7]))
    assert output.shape==(7,64)
    assert module.last_input_fields is not None
    assert module.last_input_fields.shape==(1,64,64)
    assert torch.count_nonzero(module.last_input_fields[0,7:])==0
    source=inspect.getsource(stacks)
    assert "F.interpolate" not in source
    assert "bilinear" not in source


def test_residual_enabled_and_disabled() -> None:
    hidden=torch.randn(3,64)
    enabled=make_stack(VisionOpticalStackSurrogate,64,residual_enabled=True)
    disabled=make_stack(VisionOpticalStackSurrogate,64,residual_enabled=False)
    for module in (enabled,disabled):
        with torch.no_grad():
            module.output_adapter.weight.zero_(); module.output_adapter.bias.fill_(1.0)
    enabled_output=enabled(hidden,cu_seqlens=torch.tensor([0,3]))
    disabled_output=disabled(hidden,cu_seqlens=torch.tensor([0,3]))
    assert torch.allclose(enabled_output,hidden+0.1*torch.ones_like(hidden),atol=1e-6)
    assert torch.allclose(disabled_output,torch.ones_like(hidden),atol=1e-6)


def test_scale_trainability_logging_and_state_dict() -> None:
    module=make_stack(VisionOpticalStackSurrogate,64)
    named_parameters=dict(module.named_parameters())
    assert "identity_scale" not in named_parameters
    assert "modulated_scale" in named_parameters
    assert module.scale_values()=={"identity_scale":1.0,"modulated_scale":pytest.approx(0.1)}
    state=module.state_dict()
    assert "identity_scale" in state and "modulated_scale" in state


def test_teacher_cache_schema_is_pixel_budget_specific_and_outputs_only() -> None:
    assert "teacher_vision_stack_output" in CACHED_TENSORS
    assert "teacher_answer_hidden" in CACHED_TENSORS
    assert not any("input" in name or "block" in name for name in CACHED_TENSORS)
    settings=SimpleNamespace(
        model_id="id",data_root=Path("data"),classification_prompt="prompt",
        processor_min_pixels=16384,processor_max_pixels=16384,dtype="bfloat16",
        attn_implementation="sdpa",cache_dtype="float16",vision_depth=24,
        vision_hidden_size=1024,text_depth=28,text_hidden_size=2048,
        train_limit=None,test_limit=None,train_limit_per_class=None,
        test_limit_per_class=None,seed=42,
    )
    model=SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
    metadata=expected_metadata("train",3,settings,model,CIFAR10_CLASSES)
    assert metadata["dataset"]=="cifar10"
    assert metadata["class_names"]==CIFAR10_CLASSES
    assert metadata["processor_min_pixels"]==metadata["processor_max_pixels"]==16384
    assert metadata["vision_hidden_size"]==1024 and metadata["text_hidden_size"]==2048
    assert metadata["replacement_mode"]=="vision_and_language_fullstack4_token64_residual"
    assert metadata["dataset_seed"]==42


def test_student_inference_includes_scales_and_writes_predictions(tmp_path: Path) -> None:
    vision=make_stack(VisionOpticalStackSurrogate,64)
    language=make_stack(LanguageOpticalStackSurrogate,64)
    replacement=SimpleNamespace(vision_surrogate=vision,language_surrogate=language)
    names=CIFAR10_CLASSES
    logits=torch.eye(10)[:2]; labels=torch.tensor([0,1])
    metrics={
        "top1_accuracy":1.0,"top5_accuracy":1.0,"macro_f1":1.0,
        "balanced_accuracy":1.0,"per_class_accuracy":{},"per_class_precision":{},
        "per_class_recall":{},"per_class_f1":{},
        "confusion_matrix":[[int(row==column and row<2) for column in range(10)] for row in range(10)],"samples":2,
    }
    save_student_inference(
        {"metrics":metrics,"indices":[0,1],"labels":labels,"logits":logits},
        SimpleNamespace(output_dir=tmp_path),names,replacement,
    )
    report=(tmp_path/"metrics"/"student_inference.json").read_text(encoding="utf-8")
    assert '"alpha_v"' in report and '"beta_l"' in report
    assert (tmp_path/"metrics"/"student_predictions.csv").is_file()
    header=(tmp_path/"metrics"/"student_predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "logit_airplane" in header and "logit_truck" in header


def test_training_records_scales_each_epoch() -> None:
    import experiments.qwen3_vl_2b_cifar10_optical_fullstack4_token64_residual.training as training
    source=inspect.getsource(training.train_student)
    assert "_residual_scale_values(replacement)" in source
    assert "alpha_v=" in source and "beta_l=" in source
    assert "student_training_history.csv" in inspect.getsource(training._write_student_epoch_outputs)
