from __future__ import annotations

import csv
import inspect
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.datasets import (
    distortion_level_label,
    infer_score_direction,
    load_kadid10k,
    reference_split_indices,
    references_of,
    sample_metadata_of,
    score_tertile_labels,
    stratified_split_indices,
)
from experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.data_prepare import ensure_kadid10k_dataset
from experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.optics import (
    LanguageOpticalStackSurrogate,
    VisionOpticalStackSurrogate,
)
from experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.optics import stacks
from experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.settings import (
    Settings,
    load_settings,
)
from experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.teacher_cache import (
    CACHED_TENSORS,
    expected_metadata,
)
from experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.training import (
    save_student_inference,
)


def make_stack(cls:type[nn.Module],hidden_size:int,residual_enabled:bool=True)->nn.Module:
    module=cls(hidden_size=hidden_size,optical_dim=64,conversions=4,field_size=64,padding_size=128,
        wavelength_nm=532.0,pixel_pitch_um=8.0,distance_cm=5.0,amplitude_mask_enabled=False,
        phase_init="zeros",phase_init_std=0.02,residual_enabled=residual_enabled,
        identity_scale_init=1.0,modulated_scale_init=0.1,
        identity_scale_trainable=False,modulated_scale_trainable=True)
    module.conversions=nn.ModuleList([nn.Identity() for _ in range(4)])
    return module


def make_kadid_root(tmp_path:Path,score_column:str="dmos")->Path:
    root=tmp_path/"kadid10k";images=root/"images";images.mkdir(parents=True)
    rows=[];number=0
    for ref_index in range(6):
        for local in range(3):
            number+=1;name=f"ref{ref_index:02d}_dist{local}.png"
            Image.new("RGB",(8,8),(number,number,number)).save(images/name)
            rows.append({"dist_img":name,"ref_img":f"ref{ref_index:02d}.png",
                         score_column:str(number),"distortion_level":str(local+1),
                         "distortion_type":"blur"})
    with (root/"dmos.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return root


def dataset_settings(root:Path)->Settings:
    settings=Settings(data_root=root,output_dir=root/"run",train_limit_per_class=None,
        test_limit_per_class=None,train_samples_per_class_per_epoch=None,num_workers=0)
    settings.validate();return settings


def test_config_parsing()->None:
    path=Path(__file__).parents[1]/"configs"/"kadid10k_quality3.json";settings=load_settings(path)
    assert settings.dataset=="kadid10k_quality3" and settings.metadata_csv=="dmos.csv"
    assert settings.image_dir=="images" and settings.quality_label_mode=="score_tertile"
    assert settings.processor_min_pixels==settings.processor_max_pixels==16384
    assert settings.optical_dim==settings.optical_field_size==64
    assert settings.optical_residual_enabled and settings.optical_modulated_scale_trainable


def test_kadid_metadata_parsing(tmp_path:Path)->None:
    settings=dataset_settings(make_kadid_root(tmp_path));bundle=load_kadid10k(settings)
    metadata=sample_metadata_of(bundle.train,0)
    assert Path(metadata["image_path"]).is_file()
    assert metadata["image_name"].endswith(".png")
    assert metadata["reference_image"].startswith("ref")
    assert isinstance(metadata["quality_score"],float)
    assert metadata["distortion_level"] in {1,2,3}
    assert metadata["distortion_type"]=="blur"
    assert bundle.class_names==["high_quality","medium_quality","low_quality"]
    assert bundle.metadata["quality_score_column"]=="dmos"
    assert bundle.metadata["quality_score_higher_is_better"] is False


def test_prepare_data_downloads_and_locates_archive(tmp_path:Path)->None:
    source=tmp_path/"source";images=source/"KADID-10k"/"images";images.mkdir(parents=True)
    Image.new("RGB",(8,8)).save(images/"I01_01_01.png")
    (source/"KADID-10k"/"dmos.csv").write_text("dist_img,ref_img,dmos\nI01_01_01.png,I01.png,1.0\n",encoding="utf-8")
    archive=tmp_path/"source.zip"
    with zipfile.ZipFile(archive,"w") as bundle:
        for path in (source/"KADID-10k").rglob("*"):
            if path.is_file():bundle.write(path,path.relative_to(source))
    prepared=ensure_kadid10k_dataset(tmp_path/"target","dmos.csv","images",archive.as_uri())
    assert Path(prepared["metadata_csv"]).name=="dmos.csv"
    assert Path(prepared["image_dir"]).name=="images"
    assert prepared["prepared"] is True


def test_score_tertile_labels_both_directions()->None:
    scores=list(range(1,10))
    higher,_=score_tertile_labels(scores,True);lower,_=score_tertile_labels(scores,False)
    assert higher==[2,2,2,1,1,1,0,0,0]
    assert lower==[0,0,0,1,1,1,2,2,2]


def test_dmos_mos_direction_inference()->None:
    assert infer_score_direction("dmos",None) is False
    assert infer_score_direction("dmos_mean",None) is False
    assert infer_score_direction("mos",None) is True
    with pytest.raises(RuntimeError,match="quality_score_higher_is_better"):
        infer_score_direction("score",None)
    assert infer_score_direction("score",True) is True


def test_distortion_level_three_class_labels()->None:
    assert [distortion_level_label(level) for level in range(1,6)]==[0,0,1,2,2]
    with pytest.raises(RuntimeError,match="levels 1-5"):distortion_level_label(6)


def test_reference_disjoint_split_and_validation(tmp_path:Path)->None:
    settings=dataset_settings(make_kadid_root(tmp_path));bundle=load_kadid10k(settings)
    train_indices,validation_indices=stratified_split_indices(bundle.train,settings.validation_fraction,settings.seed)
    train_refs={references_of(bundle.train)[index] for index in train_indices}
    validation_refs={references_of(bundle.train)[index] for index in validation_indices}
    test_refs=set(references_of(bundle.test))
    assert train_refs.isdisjoint(validation_refs)
    assert train_refs.isdisjoint(test_refs)
    assert validation_refs.isdisjoint(test_refs)
    assert bundle.metadata["reference_disjoint_train_validation_test"]


def test_reference_split_is_deterministic()->None:
    dataset=SimpleNamespace(references=[f"ref{i//2}" for i in range(20)],labels=[i%3 for i in range(20)])
    first=reference_split_indices(dataset,0.2,42);second=reference_split_indices(dataset,0.2,42)
    assert first==second
    assert {dataset.references[i] for i in first[0]}.isdisjoint(dataset.references[i] for i in first[1])


def test_vision_token64_success_and_overflow()->None:
    module=make_stack(VisionOpticalStackSurrogate,1024)
    assert module(torch.randn(60,1024),cu_seqlens=torch.tensor([0,60])).shape==(60,1024)
    with pytest.raises(RuntimeError) as error:module(torch.randn(65,1024),cu_seqlens=torch.tensor([0,65]))
    message=str(error.value);assert "visual token count" in message and "optical_field_size" in message and "processor_max_pixels" in message


def test_language_token64_success_and_overflow()->None:
    module=make_stack(LanguageOpticalStackSurrogate,2048)
    assert module(torch.randn(1,50,2048),attention_mask=torch.ones(1,50)).shape==(1,50,2048)
    with pytest.raises(RuntimeError) as error:module(torch.randn(1,65,2048),attention_mask=torch.ones(1,65))
    message=str(error.value);assert "language sequence length" in message and "prompt" in message and "processor_max_pixels" in message


def test_zero_padding_without_interpolation()->None:
    module=make_stack(VisionOpticalStackSurrogate,64,residual_enabled=False)
    output=module(torch.randn(7,64),cu_seqlens=torch.tensor([0,7]))
    assert output.shape==(7,64) and module.last_input_fields.shape==(1,64,64)
    assert torch.count_nonzero(module.last_input_fields[0,7:])==0
    source=inspect.getsource(stacks);assert "F.interpolate" not in source and "bilinear" not in source


def test_residual_scales_and_trainability()->None:
    hidden=torch.randn(3,64);enabled=make_stack(VisionOpticalStackSurrogate,64,True);disabled=make_stack(VisionOpticalStackSurrogate,64,False)
    for module in (enabled,disabled):
        with torch.no_grad():module.output_adapter.weight.zero_();module.output_adapter.bias.fill_(1.0)
    assert torch.allclose(enabled(hidden,cu_seqlens=torch.tensor([0,3])),hidden+0.1,atol=1e-6)
    assert torch.allclose(disabled(hidden,cu_seqlens=torch.tensor([0,3])),torch.ones_like(hidden),atol=1e-6)
    parameters=dict(enabled.named_parameters());state=enabled.state_dict()
    assert "identity_scale" not in parameters and "modulated_scale" in parameters
    assert "identity_scale" in state and "modulated_scale" in state


def test_teacher_cache_metadata_is_kadid_specific(tmp_path:Path)->None:
    settings=dataset_settings(make_kadid_root(tmp_path));bundle=load_kadid10k(settings)
    settings.vision_depth=24;settings.vision_hidden_size=1024;settings.text_depth=28;settings.text_hidden_size=2048
    model=SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
    metadata=expected_metadata("train",len(bundle.train),settings,model,bundle.class_names)
    for key in ("metadata_csv","image_dir","quality_label_mode","quality_score_column",
                "quality_score_higher_is_better","classification_prompt","vision_hidden_size","text_hidden_size"):
        assert key in metadata
    assert metadata["dataset"]=="kadid10k_quality3"
    assert "reference_ids" in CACHED_TENSORS
    assert not any("input" in name or "block" in name for name in CACHED_TENSORS)


def test_student_prediction_contains_kadid_metadata(tmp_path:Path)->None:
    names=["high_quality","medium_quality","low_quality"];logits=torch.eye(3)[:2];labels=torch.tensor([0,1])
    metrics={"top1_accuracy":1.0,"top5_accuracy":1.0,"macro_f1":1.0,"balanced_accuracy":1.0,
        "per_class_accuracy":{},"per_class_precision":{},"per_class_recall":{},"per_class_f1":{},
        "confusion_matrix":[[1,0,0],[0,1,0],[0,0,0]],"samples":2}
    metadata=[{"image_path":f"/images/{i}.png","image_name":f"{i}.png","reference_image":f"ref{i}",
               "quality_score":float(i),"distortion_level":i+1,"distortion_type":"blur"} for i in range(2)]
    replacement=SimpleNamespace(vision_surrogate=make_stack(VisionOpticalStackSurrogate,64),language_surrogate=make_stack(LanguageOpticalStackSurrogate,64))
    save_student_inference({"metrics":metrics,"indices":[0,1],"labels":labels,"logits":logits,"sample_metadata":metadata},SimpleNamespace(output_dir=tmp_path),names,replacement)
    header=(tmp_path/"metrics"/"student_predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    for name in ("image_path","image_name","reference_image","quality_score","distortion_level","distortion_type","true_name","pred_name"):
        assert name in header


def test_training_records_residual_scales()->None:
    import experiments.qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual.training as training
    source=inspect.getsource(training.train_student)
    assert "_residual_scale_values(replacement)" in source
    assert "alpha_v=" in source and "beta_l=" in source
