from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn

from .datasets import reference_ids_of
from .features import move_inputs, multimodal_forward_features, pool_answer_hidden_state, preprocess_image_text
from .io_utils import write_json


CACHE_SCHEMA_VERSION = 3
CACHED_TENSORS = [
    "sample_indices", "labels", "reference_ids", "image_grid_thw", "visual_token_counts",
    "teacher_answer_hidden", "teacher_vision_stack_output",
]


def expected_metadata(split: str, samples: int, settings: Any, model: nn.Module, class_names: Sequence[str]) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION, "split": split, "sample_count": int(samples),
        "model_id": str(settings.model_id), "model_revision": getattr(model.config, "_commit_hash", None),
        "dataset": "kadid10k_quality3", "data_root": str(settings.data_root),
        "metadata_csv": settings.metadata_csv, "image_dir": settings.image_dir,
        "quality_label_mode": settings.quality_label_mode,
        "quality_score_column": settings.quality_score_column,
        "quality_score_higher_is_better": settings.quality_score_higher_is_better,
        "class_names": list(class_names),
        "train_limit": settings.train_limit, "test_limit": settings.test_limit,
        "train_limit_per_class": settings.train_limit_per_class,
        "test_limit_per_class": settings.test_limit_per_class, "dataset_seed": settings.seed,
        "prompt": settings.classification_prompt, "classification_prompt": settings.classification_prompt,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels, "dtype": settings.dtype,
        "attention_implementation": settings.attn_implementation, "cache_dtype": settings.cache_dtype,
        "vision_depth": settings.vision_depth, "vision_hidden_size": settings.vision_hidden_size,
        "text_depth": settings.text_depth, "text_hidden_size": settings.text_hidden_size,
        "replacement_mode": "vision_and_language_fullstack4_token64_residual",
        "cached_tensors": list(CACHED_TENSORS),
    }


@torch.inference_mode()
def build_teacher_cache(split: str, model: nn.Module, processor: Any, replacement: Any, loader: Any,
                        dataset_size: int, class_names: Sequence[str], settings: Any, device: torch.device) -> Path:
    root = settings.output_dir / "teacher_cache"; manifest_path = root / f"{split}.pt"
    expected = expected_metadata(split, dataset_size, settings, model, class_names)
    if manifest_path.is_file():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        changed = [key for key in expected if expected.get(key)!=manifest["metadata"].get(key)]
        if changed:
            raise RuntimeError(f"Teacher cache metadata mismatch for {split}: {changed}. Delete {manifest_path} and its shard directory, then rerun teacher_precompute.")
        missing = [record["path"] for record in manifest.get("shards", []) if not Path(record["path"]).is_file()]
        if missing:
            raise RuntimeError(f"Teacher cache manifest references missing shards: {missing[:3]}. Delete and regenerate the {split} cache.")
        print(f"[teacher_precompute] validated existing cache: {manifest_path}", flush=True); return manifest_path
    shard_dir = root / f"{split}_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    replacement.use_teacher(); cache_dtype = torch.float16 if settings.cache_dtype=="float16" else torch.float32
    dataset_reference_ids=reference_ids_of(loader.dataset.dataset)
    pending: list[dict[str,Any]]=[]; shards=[]; total_bytes=0
    for batch_index,(images,labels,indices) in enumerate(loader):
        inputs_cpu=preprocess_image_text(processor,images,settings.classification_prompt)
        inputs=move_inputs(inputs_cpu,device); replacement.teacher_vision_output=None; replacement.teacher_cu_seqlens=None
        hidden=multimodal_forward_features(model,inputs); answer,_=pool_answer_hidden_state(hidden,inputs["attention_mask"])
        vision=replacement.teacher_vision_output
        if vision is None: raise RuntimeError("Teacher hook did not capture full vision stack output")
        counts=replacement.teacher_token_counts(); groups=list(vision.split(counts,dim=0))
        if len(groups)!=len(images): raise RuntimeError("Vision token boundaries do not match image batch")
        oversized_visual=[count for count in counts if count>settings.optical_field_size]
        if oversized_visual:
            raise RuntimeError(
                f"visual token count {max(oversized_visual)} exceeds "
                f"optical_field_size={settings.optical_field_size}. Lower processor_max_pixels "
                "and regenerate the teacher cache."
            )
        sequence_lengths=inputs_cpu["attention_mask"].sum(dim=1).long().tolist()
        oversized_language=[count for count in sequence_lengths if count>settings.optical_field_size]
        if oversized_language:
            raise RuntimeError(
                f"language sequence length {max(oversized_language)} exceeds "
                f"optical_field_size={settings.optical_field_size}. Shorten the prompt or lower "
                "the visual token budget / processor_max_pixels, then regenerate the teacher cache."
            )
        for local in range(len(images)):
            pending.append({"sample_index":int(indices[local]),"label":int(labels[local]),
                            "reference_id":int(dataset_reference_ids[int(indices[local])]),
                            "image_grid_thw":inputs_cpu["image_grid_thw"][local].cpu(),
                            "visual_token_count":counts[local],
                            "teacher_answer_hidden":answer[local].to(cache_dtype).cpu(),
                            "teacher_vision_stack_output":groups[local].to(cache_dtype).cpu()})
            if len(pending)>=settings.teacher_cache_shard_size:
                record=_flush_shard(shard_dir,len(shards),pending); shards.append(record); total_bytes+=record["bytes"]; pending=[]
        print(f"[teacher_precompute] {split} batch={batch_index+1}/{len(loader)} cached={min((batch_index+1)*settings.feature_batch_size,dataset_size)}/{dataset_size}",flush=True)
    if pending:
        record=_flush_shard(shard_dir,len(shards),pending); shards.append(record); total_bytes+=record["bytes"]
    metadata={**expected,"shard_count":len(shards),"total_cache_bytes":total_bytes}
    manifest={"metadata":metadata,"shards":shards}
    root.mkdir(parents=True,exist_ok=True); torch.save(manifest,manifest_path)
    write_json(root/f"{split}_metadata.json",metadata)
    return manifest_path


def _flush_shard(directory: Path, number: int, rows: list[dict[str,Any]]) -> dict[str,Any]:
    path=directory/f"shard_{number:06d}.pt"
    payload={
        "sample_indices":torch.tensor([r["sample_index"] for r in rows],dtype=torch.long),
        "labels":torch.tensor([r["label"] for r in rows],dtype=torch.long),
        "reference_ids":torch.tensor([r["reference_id"] for r in rows],dtype=torch.long),
        "image_grid_thw":torch.stack([r["image_grid_thw"] for r in rows]),
        "visual_token_counts":torch.tensor([r["visual_token_count"] for r in rows],dtype=torch.long),
        "teacher_answer_hidden":torch.stack([r["teacher_answer_hidden"] for r in rows]),
        "teacher_vision_stack_output":[r["teacher_vision_stack_output"] for r in rows],
    }
    temporary=path.with_suffix(".tmp"); torch.save(payload,temporary); temporary.replace(path)
    return {"path":str(path),"count":len(rows),"first_index":int(payload["sample_indices"][0]),
            "last_index":int(payload["sample_indices"][-1]),"bytes":path.stat().st_size,"sha256":_sha256(path)}


class TeacherCacheStore:
    def __init__(self, manifest_path: Path, expected: dict[str,Any] | None=None, max_cached_shards: int=8) -> None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Teacher cache manifest missing: {manifest_path}. Run --phase teacher_precompute first.")
        manifest=torch.load(manifest_path,map_location="cpu",weights_only=True)
        if expected is not None:
            changed=[k for k in expected if expected.get(k)!=manifest["metadata"].get(k)]
            if changed: raise RuntimeError(f"Teacher cache metadata mismatch: {changed}")
        self.metadata=manifest["metadata"]; self.shards=manifest["shards"]
        missing=[record["path"] for record in self.shards if not Path(record["path"]).is_file()]
        if missing: raise RuntimeError(f"Teacher cache shards are missing: {missing[:3]}")
        if max_cached_shards<=0: raise ValueError("max_cached_shards must be positive")
        self.max_cached_shards=int(max_cached_shards); self._loaded_shards:OrderedDict[int,dict[str,Any]]=OrderedDict(); self._ranges=[]
        offset=0
        for number,record in enumerate(self.shards): self._ranges.append((offset,offset+record["count"],number)); offset+=record["count"]
    def __len__(self): return int(self.metadata["sample_count"])
    def get(self,index:int)->dict[str,Any]:
        for start,end,number in self._ranges:
            if start<=index<end:
                if number in self._loaded_shards:
                    p=self._loaded_shards.pop(number); self._loaded_shards[number]=p
                else:
                    p=torch.load(self.shards[number]["path"],map_location="cpu",weights_only=True); self._loaded_shards[number]=p
                    while len(self._loaded_shards)>self.max_cached_shards:self._loaded_shards.popitem(last=False)
                pos=index-start
                if int(p["sample_indices"][pos])!=index: raise RuntimeError("Cache sample index ordering mismatch")
                # Shards use plural names for batched tensors. Expose an explicit
                # per-sample contract so callers cannot accidentally request a
                # non-existent singular/plural variant.
                return {
                    "sample_index": p["sample_indices"][pos],
                    "label": p["labels"][pos],
                    "reference_id": p["reference_ids"][pos],
                    "image_grid_thw": p["image_grid_thw"][pos],
                    "visual_token_count": p["visual_token_counts"][pos],
                    "teacher_answer_hidden": p["teacher_answer_hidden"][pos],
                    "teacher_vision_stack_output": p["teacher_vision_stack_output"][pos],
                }
        raise IndexError(index)
    def iter_shards(self) -> Iterator[dict[str,Any]]:
        for record in self.shards: yield torch.load(record["path"],map_location="cpu",weights_only=True)


def load_cached_tensor(store: TeacherCacheStore, key: str) -> torch.Tensor:
    return torch.cat([shard[key] for shard in store.iter_shards()])


def write_teacher_logits(output_dir: Path, split: str, logits: torch.Tensor, labels: torch.Tensor) -> Path:
    path=output_dir/"teacher_cache"/f"{split}_teacher_logits.pt"
    torch.save({"sample_indices":torch.arange(len(logits)),"labels":labels.long(),"teacher_logits":logits.half()},path)
    return path


def load_teacher_logits(path: Path) -> torch.Tensor:
    if not path.is_file(): raise FileNotFoundError(f"Teacher logits missing: {path}. Run --phase teacher_logits first.")
    return torch.load(path,map_location="cpu",weights_only=True)["teacher_logits"].float()


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
