from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.data_prepare import (
    normalize_scene_label, prepare_scene_split,
)
from experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.features import multimodal_forward_features
from experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.metrics import metrics_from_logits
from experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.optics import (
    LanguageOpticalStackSurrogate, OpticalConversion, VisionOpticalStackSurrogate,
)
from experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.sampling import EpochClassMixedSampler
from experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.settings import load_settings
from experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.teacher_cache import CACHED_TENSORS, expected_metadata
from experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.training import save_student_inference


CLASS_NAMES = ["highway", "city_street", "residential", "other"]


def stack(cls, hidden: int = 8):
    return cls(hidden_size=hidden,optical_dim=6,conversions=4,field_size=8,padding_size=10,
               wavelength_nm=532,pixel_pitch_um=8,distance_cm=5,amplitude_mask_enabled=True,
               phase_init="zeros",phase_init_std=0.02)


def test_config_parsing() -> None:
    path=Path(__file__).parents[1]/"configs"/"bdd100k_scene4.json";settings=load_settings(path)
    assert settings.model_id=="Qwen/Qwen3-VL-2B-Instruct"
    assert settings.dataset=="bdd100k_scene4"
    assert settings.train_samples_per_class_per_epoch==1000 and settings.oversample_minority_classes
    assert settings.optical_field_size==64 and settings.optical_padding_size==128
    assert settings.pixel_pitch_um==8 and settings.phase_init=="zeros"


def test_scene_label_normalization() -> None:
    assert normalize_scene_label("highway")=="highway"
    assert normalize_scene_label("city street")=="city_street"
    assert normalize_scene_label("residential")=="residential"
    for value in ("parking lot","tunnel","gas stations"):
        assert normalize_scene_label(value)=="other"
    assert normalize_scene_label("undefined") is None


def test_scene_split_and_manifest_statistics(tmp_path:Path) -> None:
    images=tmp_path/"images";images.mkdir();labels=[]
    source_labels=["highway","city street","residential","parking lot","tunnel","gas stations","undefined"]
    for index,label in enumerate(source_labels):
        name=f"{index}.jpg";Image.new("RGB",(4,4),(index,index,index)).save(images/name)
        labels.append({"name":name,"attributes":{"scene":label}})
    label_path=tmp_path/"labels.json";label_path.write_text(json.dumps(labels),encoding="utf-8")
    stats=prepare_scene_split(images,label_path,tmp_path/"prepared")
    assert stats["counts"]=={"highway":1,"city_street":1,"residential":1,"other":3}
    assert stats["ignored_scene_labels"]=={"undefined":1}
    assert stats["other_source_counts"]=={"gas stations":1,"parking lot":1,"tunnel":1}


def test_balanced_sampler_oversamples_small_other_class() -> None:
    labels=[0]*5+[1]*5+[2]*5+[3]
    sampler=EpochClassMixedSampler(range(len(labels)),labels,4,4,42,3,None,True)
    sampled=list(iter(sampler));counts=Counter(labels[index] for index in sampled)
    assert len(sampled)==12 and counts==Counter({0:3,1:3,2:3,3:3})


def test_vision_stack_preserves_packed_boundaries() -> None:
    module=stack(VisionOpticalStackSurrogate);hidden=torch.randn(9,8);boundaries=torch.tensor([0,4,9])
    output=module(hidden,cu_seqlens=boundaries)
    assert output.shape==hidden.shape and module.last_token_counts==[4,5]
    assert all(field.shape==(2,8,8) for field in module.last_fields)


def test_language_stack_preserves_shape_and_padding() -> None:
    module=stack(LanguageOpticalStackSurrogate);hidden=torch.randn(2,7,8)
    mask=torch.tensor([[0,0,1,1,1,1,1],[1,1,1,0,0,0,0]])
    module.set_attention_mask(mask);output=module(hidden)
    assert output.shape==hidden.shape and module.last_token_counts==[5,3]
    assert torch.count_nonzero(output[0,:2])==0 and torch.count_nonzero(output[1,3:])==0


def test_configured_64_field_uses_128_grid() -> None:
    module=VisionOpticalStackSurrogate(hidden_size=8,optical_dim=6,conversions=4,field_size=64,padding_size=128,
        wavelength_nm=532,pixel_pitch_um=8,distance_cm=5,amplitude_mask_enabled=False,phase_init="zeros")
    output=module(torch.randn(9,8),cu_seqlens=torch.tensor([0,4,9]))
    assert output.shape==(9,8) and all(field.shape==(2,64,64) for field in module.last_fields)


def test_phase_initialization_and_conversion() -> None:
    zeros=OpticalConversion(8,10,532,8,5,False,"zeros",0.02)
    assert torch.count_nonzero(zeros.phase_mask)==0
    uniform=OpticalConversion(8,10,532,8,5,False,"uniform_0_2pi",0.02)
    assert torch.all(uniform.phase_mask>=0) and torch.all(uniform.phase_mask<=2*torch.pi)
    output=zeros(torch.rand(2,8,8));assert output.shape==(2,8,8) and torch.all(output>=0)
    assert "sqrt" not in inspect.getsource(OpticalConversion.forward)


def test_teacher_cache_schema_is_scene4_outputs_only() -> None:
    assert "teacher_vision_stack_output" in CACHED_TENSORS and "teacher_answer_hidden" in CACHED_TENSORS
    assert not any("input" in name or "block" in name for name in CACHED_TENSORS)
    model=SimpleNamespace(config=SimpleNamespace(_commit_hash="abc"))
    settings=SimpleNamespace(model_id="id",data_root=Path("data"),classification_prompt="p",processor_min_pixels=1,
        processor_max_pixels=1,dtype="bfloat16",attn_implementation="sdpa",cache_dtype="float16",
        vision_depth=24,vision_hidden_size=8,text_depth=28,text_hidden_size=12,
        train_limit=None,test_limit=None,train_limit_per_class=None,test_limit_per_class=None,seed=42)
    metadata=expected_metadata("train",4,settings,model,CLASS_NAMES)
    assert metadata["dataset"]=="bdd100k_scene4" and metadata["cached_tensors"]==CACHED_TENSORS


def test_student_inference_writes_four_class_predictions(tmp_path:Path) -> None:
    logits=torch.eye(4)[:2];labels=torch.tensor([0,1]);settings=SimpleNamespace(output_dir=tmp_path)
    metrics={"top1_accuracy":1.0,"top5_accuracy":1.0,"macro_f1":1.0,"balanced_accuracy":1.0,
             "per_class_accuracy":{},"per_class_precision":{},"per_class_recall":{},"per_class_f1":{},
             "confusion_matrix":[[int(i==j and i<2) for j in range(4)] for i in range(4)],"samples":2}
    save_student_inference({"metrics":metrics,"indices":[0,1],"labels":labels,"logits":logits},settings,CLASS_NAMES)
    text=(tmp_path/"metrics"/"student_predictions.csv").read_text()
    assert "logit_highway" in text and "logit_other" in text


def test_four_class_metrics_include_nontrivial_top2() -> None:
    logits=torch.tensor([[4.0,3.0,1.0,0.0],[0.0,2.0,3.0,1.0]])
    metrics=metrics_from_logits(logits,torch.tensor([1,2]),CLASS_NAMES)
    assert metrics["top1_accuracy"]==0.5 and metrics["top2_accuracy"]==1.0
    assert metrics["top5_accuracy"]==1.0


def test_hidden_state_fallback_preserves_gradients() -> None:
    class Output:
        hidden_states=None
    class Model(torch.nn.Module):
        def forward(self,input_ids,**kwargs):
            del kwargs;hidden=input_ids.float().unsqueeze(-1).repeat(1,1,3).requires_grad_();self._optical_fullstack_last_hidden=hidden;return Output()
    hidden=multimodal_forward_features(Model(),{"input_ids":torch.ones(2,4,dtype=torch.long)})
    assert hidden.shape==(2,4,3) and hidden.requires_grad


def test_training_writes_history_during_epochs() -> None:
    import experiments.qwen3_vl_2b_bdd100k_scene4_optical_fullstack4.training as training
    assert "_write_student_epoch_outputs" in inspect.getsource(training.train_student)
    assert "student_training_history.csv" in inspect.getsource(training._write_student_epoch_outputs)
