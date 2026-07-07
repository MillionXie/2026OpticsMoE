from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from .datasets import DatasetBundle, load_kadid10k, make_indexed_loader, stratified_split_indices
from .download import download_checkpoint
from .io_utils import resolve_device, resolve_dtype, runtime_metadata, set_seed, write_csv, write_json
from .modeling import MLPHead, LoadedBackbone, load_backbone, parameter_report
from .optics import FullStackReplacement, LanguageOpticalStackSurrogate, VisionOpticalStackSurrogate
from .settings import Settings, load_settings, resolve_path
from .teacher_cache import TeacherCacheStore, build_teacher_cache, expected_metadata, load_teacher_logits
from .training import (evaluate_student, generate_teacher_logits, load_head, load_student_parts,
                       save_student_inference, teacher_inference, train_student, train_teacher_head)


PHASES = ("download", "prepare_data", "teacher_precompute", "teacher_train", "teacher_logits",
          "teacher_inference", "student_train", "student_inference", "compare", "all")


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="Qwen3-VL-2B KADID-10k Quality-3 token64 optical4 residual distillation")
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--phase",choices=PHASES,default="all")
    parser.add_argument("--device"); parser.add_argument("--cache-dir",type=Path); parser.add_argument("--model-id")
    parser.add_argument("--local-files-only",action="store_true"); parser.add_argument("--output-dir",type=Path)
    return parser


def main(argv:list[str]|None=None)->int:
    args=build_parser().parse_args(argv); settings=load_settings(args.config); _overrides(settings,args); set_seed(settings.seed)
    _make_dirs(settings.output_dir); write_json(settings.output_dir/"config_resolved.json",settings.to_dict())
    if args.phase=="download":
        path=download_checkpoint(settings.model_id,settings.cache_dir); write_json(settings.output_dir/"download.json",{"model_id":settings.model_id,"snapshot":str(path)}); print(path); return 0
    data=load_kadid10k(settings); write_json(settings.output_dir/"dataset.json",data.metadata); write_json(settings.output_dir/"config_resolved.json",settings.to_dict())
    if args.phase=="prepare_data": print(f"KADID-10k Quality-3 ready: train_pool={len(data.train)} test={len(data.test)}"); return 0
    device=resolve_device(settings.device); write_json(settings.output_dir/"environment.json",runtime_metadata(device))

    # Cache/MLP-only phases intentionally do not construct or run Qwen.
    if args.phase in {"teacher_train","teacher_logits","teacher_inference","compare"}:
        stores=_load_stores_without_model(settings,data)
        _resolve_architecture_from_cache(settings,stores["train"]); write_json(settings.output_dir/"config_resolved.json",settings.to_dict())
        if args.phase=="teacher_train": train_teacher_head(stores["train"],stores["test"],settings,data.class_names,device); return 0
        head=load_head(settings.output_dir/"checkpoints"/"teacher_mlp.pt",settings.dropout,device)
        if args.phase=="teacher_logits": generate_teacher_logits(head,stores,settings,device); return 0
        if args.phase=="teacher_inference": teacher_inference(head,stores["test"],settings,data.class_names,device); return 0
        _compare(settings,data.class_names); return 0

    loaded=_load_model(settings,device); settings.resolve_architecture(loaded.model)
    replacement=_build_replacement(loaded,settings,device); write_json(settings.output_dir/"config_resolved.json",settings.to_dict())
    _write_model_report(loaded.model,replacement,settings)
    try:
        if args.phase in {"teacher_precompute","all"}:
            _precompute(loaded,replacement,data,settings,device)
            if args.phase=="teacher_precompute": return 0
        stores = (
            _load_stores(settings, loaded.model, data.class_names, data)
            if args.phase in {"student_train", "all"}
            else None
        )
        if args.phase=="all":
            assert stores is not None
            teacher_head=train_teacher_head(stores["train"],stores["test"],settings,data.class_names,device)
            generate_teacher_logits(teacher_head,stores,settings,device)
            teacher_inference(teacher_head,stores["test"],settings,data.class_names,device)
        elif args.phase=="student_train":
            teacher_head=load_head(settings.output_dir/"checkpoints"/"teacher_mlp.pt",settings.dropout,device)
        if args.phase in {"student_train","all"}:
            assert stores is not None
            student_head=MLPHead(settings.text_hidden_size,settings.hidden_dim,len(data.class_names),settings.dropout).to(device)
            student_head.load_state_dict(teacher_head.state_dict())
            train_indices,val_indices=stratified_split_indices(data.train,settings.validation_fraction,settings.seed)
            train_student(loaded.model,loaded.processor,replacement,student_head,data.train,Subset(data.train,val_indices),stores["train"],settings,data.class_names,device)
            if args.phase=="student_train": return 0
        if args.phase in {"student_inference","all"}:
            student_head=MLPHead(settings.text_hidden_size,settings.hidden_dim,len(data.class_names),settings.dropout).to(device)
            load_student_parts(settings,replacement,student_head,device,"best")
            loader=make_indexed_loader(data.test,settings.inference_batch_size,settings.num_workers,False,settings.seed)
            result=evaluate_student(loaded.model,loaded.processor,replacement,student_head,loader,data.class_names,settings,device)
            save_student_inference(result,settings,data.class_names,replacement)
            if args.phase=="student_inference": return 0
        if args.phase=="all": _compare(settings,data.class_names)
        return 0
    finally:
        replacement.close()


def _overrides(settings:Settings,args:argparse.Namespace)->None:
    if args.device: settings.device=args.device
    if args.cache_dir: settings.cache_dir=resolve_path(args.cache_dir,Path.cwd(),"cache_dir")
    if args.output_dir: settings.output_dir=resolve_path(args.output_dir,Path.cwd(),"output_dir")
    if args.model_id: settings.model_id=args.model_id if args.model_id=="Qwen/Qwen3-VL-2B-Instruct" else str(resolve_path(args.model_id,Path.cwd(),"model_id"))
    if args.local_files_only: settings.local_files_only=True
    settings.validate()


def _load_model(settings:Settings,device:torch.device)->LoadedBackbone:
    _log(f"loading {settings.model_id}")
    return load_backbone(settings.model_id,settings.cache_dir,settings.local_files_only,resolve_dtype(settings.dtype),device,settings.attn_implementation,settings.processor_min_pixels,settings.processor_max_pixels)


def _stack_kwargs(settings:Settings)->dict[str,Any]:
    return {
        "optical_dim":settings.optical_dim,
        "conversions":settings.optical_conversions_per_stack,
        "field_size":settings.optical_field_size,
        "padding_size":settings.optical_padding_size,
        "wavelength_nm":settings.wavelength_nm,
        "pixel_pitch_um":settings.pixel_pitch_um,
        "distance_cm":settings.mask_distance_cm,
        "amplitude_mask_enabled":settings.amplitude_mask_enabled,
        "phase_init":settings.phase_init,
        "phase_init_std":settings.phase_init_std,
        "residual_enabled":settings.optical_residual_enabled,
        "identity_scale_init":settings.optical_identity_scale_init,
        "modulated_scale_init":settings.optical_modulated_scale_init,
        "identity_scale_trainable":settings.optical_identity_scale_trainable,
        "modulated_scale_trainable":settings.optical_modulated_scale_trainable,
    }


def _build_replacement(loaded:LoadedBackbone,settings:Settings,device:torch.device)->FullStackReplacement:
    vision=VisionOpticalStackSurrogate(hidden_size=settings.vision_hidden_size,**_stack_kwargs(settings)).to(device)
    language=LanguageOpticalStackSurrogate(hidden_size=settings.text_hidden_size,**_stack_kwargs(settings)).to(device)
    return FullStackReplacement(loaded.model,vision,language)


def _precompute(loaded:LoadedBackbone,replacement:FullStackReplacement,data:DatasetBundle,settings:Settings,device:torch.device)->None:
    for split,dataset in (("train",data.train),("test",data.test)):
        loader=make_indexed_loader(dataset,settings.feature_batch_size,settings.num_workers,False,settings.seed)
        build_teacher_cache(split,loaded.model,loaded.processor,replacement,loader,len(dataset),data.class_names,settings,device)


def _load_stores(settings:Settings,model:torch.nn.Module,names:list[str],data:DatasetBundle)->dict[str,TeacherCacheStore]:
    return {split:TeacherCacheStore(settings.output_dir/"teacher_cache"/f"{split}.pt",expected_metadata(split,len(dataset),settings,model,names),settings.teacher_cache_lru_shards) for split,dataset in (("train",data.train),("test",data.test))}


def _load_stores_without_model(settings:Settings,data:DatasetBundle)->dict[str,TeacherCacheStore]:
    stores={split:TeacherCacheStore(settings.output_dir/"teacher_cache"/f"{split}.pt",max_cached_shards=settings.teacher_cache_lru_shards) for split in ("train","test")}
    for split,dataset in (("train",data.train),("test",data.test)):
        metadata=stores[split].metadata
        expected={
            "cache_schema_version":3,"split":split,"sample_count":len(dataset),
            "dataset":"kadid10k_quality3",
            "data_root":str(settings.data_root),"class_names":list(data.class_names),
            "metadata_csv":settings.metadata_csv,"image_dir":settings.image_dir,
            "quality_label_mode":settings.quality_label_mode,
            "quality_score_column":settings.quality_score_column,
            "quality_score_higher_is_better":settings.quality_score_higher_is_better,
            "prompt":settings.classification_prompt,
            "classification_prompt":settings.classification_prompt,
            "processor_min_pixels":settings.processor_min_pixels,
            "processor_max_pixels":settings.processor_max_pixels,
            "dtype":settings.dtype,"attention_implementation":settings.attn_implementation,
            "cache_dtype":settings.cache_dtype,
            "replacement_mode":"vision_and_language_fullstack4_token64_residual",
        }
        changed=[key for key,value in expected.items() if metadata.get(key)!=value]
        if changed:
            raise RuntimeError(
                f"Teacher cache metadata mismatch for {split}: {changed}. "
                "Delete the manifest and shard directory, then rerun teacher_precompute."
            )
    return stores


def _resolve_architecture_from_cache(settings:Settings,store:TeacherCacheStore)->None:
    for name in ("vision_depth","vision_hidden_size","text_depth","text_hidden_size"): setattr(settings,name,int(store.metadata[name]))


def _write_model_report(model:torch.nn.Module,replacement:FullStackReplacement,settings:Settings)->None:
    report=parameter_report(model); report.update({"model_id":settings.model_id,"replacement_mode":"vision_and_language_fullstack4_token64_residual","vision_depth":settings.vision_depth,"vision_hidden_size":settings.vision_hidden_size,"text_depth":settings.text_depth,"text_hidden_size":settings.text_hidden_size,"vision_optical_output_hidden_size":settings.vision_hidden_size,"vision_merger_output_hidden_size":settings.text_hidden_size,"language_optical_output_hidden_size":settings.text_hidden_size,"vision_optical_conversions":4,"language_optical_conversions":4,"optical_dim":settings.optical_dim,"optical_field_size":settings.optical_field_size,"optical_padding_size":settings.optical_padding_size,"pixel_pitch_um":settings.pixel_pitch_um,"phase_init":settings.phase_init,"phase_init_std":settings.phase_init_std,"amplitude_mask_enabled":settings.amplitude_mask_enabled,"token_field_mapping":"direct token rows plus strict zero padding","optical_residual_enabled":settings.optical_residual_enabled,"optical_identity_scale_init":settings.optical_identity_scale_init,"optical_modulated_scale_init":settings.optical_modulated_scale_init,"optical_identity_scale_trainable":settings.optical_identity_scale_trainable,"optical_modulated_scale_trainable":settings.optical_modulated_scale_trainable,**_scale_values(replacement),"detected_intensity_reencoded_with_sqrt":False,"teacher_cache":"stack outputs only; independent pixel-budget metadata","vision_surrogate_parameters":sum(p.numel() for p in replacement.vision_surrogate.parameters()),"language_surrogate_parameters":sum(p.numel() for p in replacement.language_surrogate.parameters())})
    write_json(settings.output_dir/"model.json",report)


def _scale_values(replacement:FullStackReplacement)->dict[str,float]:
    vision=replacement.vision_surrogate.scale_values(); language=replacement.language_surrogate.scale_values()
    return {"beta_v":vision["identity_scale"],"alpha_v":vision["modulated_scale"],"beta_l":language["identity_scale"],"alpha_l":language["modulated_scale"]}


def _compare(settings:Settings,names:list[str])->None:
    teacher=json.loads((settings.output_dir/"metrics"/"teacher_inference.json").read_text()); student=json.loads((settings.output_dir/"metrics"/"student_inference.json").read_text())
    comparison={"dataset":"kadid10k_quality3","classes":names,"replacement":"both_full_stacks_optical4_token64_residual","teacher":teacher,"student":student,"accuracy_drop":{"top1":teacher["top1_accuracy"]-student["top1_accuracy"],"top5":teacher["top5_accuracy"]-student["top5_accuracy"]}}
    write_json(settings.output_dir/"metrics"/"comparison.json",comparison)
    teacher_logits=load_teacher_logits(settings.output_dir/"teacher_cache"/"test_teacher_logits.pt"); teacher_preds=teacher_logits.argmax(1).tolist()
    with (settings.output_dir/"metrics"/"student_predictions.csv").open(encoding="utf-8") as handle: student_rows=list(csv.DictReader(handle))
    rows=[]
    for row,tp in zip(student_rows,teacher_preds):
        truth=int(row["true_label"]); sp=int(row["pred_label"]); rows.append({"sample_index":row["sample_index"],"true_name":names[truth],"teacher_pred_name":names[tp],"student_pred_name":names[sp],"teacher_correct":tp==truth,"student_correct":sp==truth,"agreement":tp==sp})
    write_csv(settings.output_dir/"metrics"/"teacher_student_agreement.csv",rows,list(rows[0]))


def _make_dirs(root:Path)->None:
    for name in ("teacher_cache","metrics","checkpoints","figures/vision_phase_masks","figures/language_phase_masks","figures/vision_light_fields","figures/language_light_fields"): (root/name).mkdir(parents=True,exist_ok=True)


def _log(message:str)->None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}",flush=True)


if __name__=="__main__": raise SystemExit(main())
