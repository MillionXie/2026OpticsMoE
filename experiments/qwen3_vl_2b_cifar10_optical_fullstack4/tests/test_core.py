from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.qwen3_vl_2b_cifar10_optical_fullstack4.optics import (
    LanguageOpticalStackSurrogate, OpticalConversion, VisionOpticalStackSurrogate,
)
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4.settings import load_settings
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4.teacher_cache import CACHED_TENSORS, TeacherCacheStore, expected_metadata
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4.training import save_student_inference
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4.features import multimodal_forward_features
from experiments.qwen3_vl_2b_cifar10_optical_fullstack4.sampling import EpochClassMixedSampler


def stack(cls, hidden: int = 8):
    return cls(hidden_size=hidden,optical_dim=6,conversions=4,field_size=8,padding_size=10,
               wavelength_nm=532,pixel_pitch_um=17,distance_cm=5,amplitude_mask_enabled=True)


def test_config_parsing() -> None:
    path=Path(__file__).parents[1]/"configs"/"cifar10.json"; settings=load_settings(path)
    assert settings.model_id=="Qwen/Qwen3-VL-2B-Instruct"
    assert settings.optical_conversions_per_stack==4
    assert settings.replace_vision_stack and settings.replace_language_stack


def test_vision_stack_preserves_packed_boundaries() -> None:
    module=stack(VisionOpticalStackSurrogate); hidden=torch.randn(9,8); boundaries=torch.tensor([0,4,9])
    output=module(hidden,cu_seqlens=boundaries)
    assert output.shape==hidden.shape; assert module.last_token_counts==[4,5]
    assert all(field.shape==(2,8,8) for field in module.last_fields)


def test_language_stack_preserves_shape_and_ignores_padding() -> None:
    module=stack(LanguageOpticalStackSurrogate); hidden=torch.randn(2,7,8)
    mask=torch.tensor([[0,0,1,1,1,1,1],[1,1,1,0,0,0,0]])
    module.set_attention_mask(mask); output=module(hidden)
    assert output.shape==hidden.shape; assert module.last_token_counts==[5,3]
    assert torch.count_nonzero(output[0,:2])==0 and torch.count_nonzero(output[1,3:])==0


def test_conversion_is_nonnegative_and_does_not_sqrt_detected_intensity() -> None:
    module=OpticalConversion(8,10,532,17,5,True); output=module(torch.rand(2,8,8))
    assert output.shape==(2,8,8); assert torch.all(output>=0)
    source=inspect.getsource(OpticalConversion.forward)
    assert "sqrt" not in source


def test_teacher_cache_schema_stores_outputs_only() -> None:
    assert "teacher_vision_stack_output" in CACHED_TENSORS
    assert "teacher_answer_hidden" in CACHED_TENSORS
    assert not any("input" in name or "block" in name for name in CACHED_TENSORS)
    model=SimpleNamespace(config=SimpleNamespace(_commit_hash="abc"))
    settings=SimpleNamespace(model_id="id",data_root=Path("data"),classification_prompt="p",processor_min_pixels=1,
        processor_max_pixels=1,dtype="bfloat16",attn_implementation="sdpa",cache_dtype="float16",
        vision_depth=24,vision_hidden_size=8,text_depth=28,text_hidden_size=12,
        train_limit=None,test_limit=None,train_limit_per_class=None,test_limit_per_class=None,seed=42)
    metadata=expected_metadata("train",3,settings,model,[str(i) for i in range(10)])
    assert metadata["cached_tensors"]==CACHED_TENSORS


def test_teacher_cache_store_exposes_per_sample_names(tmp_path: Path) -> None:
    shard=tmp_path/"shard.pt"
    torch.save({
        "sample_indices":torch.tensor([0]), "labels":torch.tensor([3]),
        "image_grid_thw":torch.tensor([[1,8,8]]), "visual_token_counts":torch.tensor([64]),
        "teacher_answer_hidden":torch.randn(1,12),
        "teacher_vision_stack_output":[torch.randn(64,8)],
    },shard)
    manifest=tmp_path/"train.pt"
    torch.save({"metadata":{"sample_count":1},"shards":[{"path":str(shard),"count":1}]},manifest)
    sample=TeacherCacheStore(manifest).get(0)
    assert set(sample)=={"sample_index","label","image_grid_thw","visual_token_count","teacher_answer_hidden","teacher_vision_stack_output"}
    assert int(sample["label"])==3 and int(sample["visual_token_count"])==64


def test_student_inference_writes_predictions(tmp_path: Path) -> None:
    names=[f"c{i}" for i in range(10)]; logits=torch.eye(10)[:2]; labels=torch.tensor([0,1])
    settings=SimpleNamespace(output_dir=tmp_path)
    metrics={"top1_accuracy":1.0,"top5_accuracy":1.0,"macro_f1":0.2,"balanced_accuracy":0.2,
             "per_class_accuracy":{},"per_class_precision":{},"per_class_recall":{},"per_class_f1":{},
             "confusion_matrix":[[int(i==j and i<2) for j in range(10)] for i in range(10)],"samples":2}
    save_student_inference({"metrics":metrics,"indices":[0,1],"labels":labels,"logits":logits},settings,names)
    path=tmp_path/"metrics"/"student_predictions.csv"; assert path.is_file()
    assert "logit_c0" in path.read_text()


def test_training_source_writes_history_inside_epoch_loop() -> None:
    import experiments.qwen3_vl_2b_cifar10_optical_fullstack4.training as training
    source=inspect.getsource(training.train_student)
    assert "_write_student_epoch_outputs" in source
    assert "student_training_history.csv" in inspect.getsource(training._write_student_epoch_outputs)


def test_epoch_sampler_mixes_cifar_classes() -> None:
    labels=[class_index for class_index in range(10) for _ in range(4)];sampler=EpochClassMixedSampler(range(40),labels,10,20,42,None,8);sampler.set_epoch(1);indices=list(sampler)
    assert len(indices)==40
    for start in range(0,40,20):assert {labels[index] for index in indices[start:start+20]}==set(range(10))


def test_hidden_state_fallback_preserves_gradients() -> None:
    class Output:
        hidden_states = None
    class Model(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            del kwargs
            hidden=input_ids.float().unsqueeze(-1).repeat(1,1,3).requires_grad_()
            self._optical_fullstack_last_hidden=hidden
            return Output()
    model=Model(); hidden=multimodal_forward_features(model,{"input_ids":torch.ones(2,4,dtype=torch.long)})
    assert hidden.shape==(2,4,3) and hidden.requires_grad
