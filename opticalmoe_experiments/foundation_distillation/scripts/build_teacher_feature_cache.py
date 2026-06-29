import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENTS_ROOT.parent
for path in (EXPERIMENTS_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.data.foundation_distillation import (
    create_distillation_datasets,
    dataset_config_hash,
    teacher_input_from_student_gray,
)
from common.data.loader_utils import apply_smoke_loader_overrides, dataloader_kwargs
from common.utils.config import load_yaml, save_json
from common.utils.seed import choose_device, set_seed
from foundation_distillation.runtime import resolve_cache_dir
from foundation_distillation.teacher import load_clip_image_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Build frozen CLIP image feature cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def encode_split(dataset, teacher, device, dataset_cfg):
    loader_cfg = dict(dataset_cfg)
    loader_cfg["batch_size"] = int(dataset_cfg.get("teacher_cache_batch_size", dataset_cfg.get("batch_size", 64)))
    loader = DataLoader(dataset, **dataloader_kwargs(loader_cfg, shuffle=False))
    features, labels, indices = [], [], []
    for images, batch_labels, batch_indices in loader:
        teacher_images = teacher_input_from_student_gray(images).to(device, non_blocking=True)
        encoded = teacher(teacher_images)
        features.append(F.normalize(encoded.float(), dim=-1).cpu())
        labels.append(torch.as_tensor(batch_labels, dtype=torch.long).cpu())
        indices.append(torch.as_tensor(batch_indices, dtype=torch.long).cpu())
    return {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "indices": torch.cat(indices, dim=0),
    }


def build_cache(config, device, overwrite=False):
    seed = int(config.get("seed", 7))
    set_seed(seed)
    dataset_cfg = config["dataset"]
    teacher_cfg = config["teacher"]
    if teacher_cfg.get("type", "clip_image_encoder") != "clip_image_encoder":
        raise NotImplementedError("Only teacher.type=clip_image_encoder is implemented in v1; DINOv2 is reserved for later.")
    if not bool(teacher_cfg.get("freeze", True)):
        raise ValueError("The CLIP image encoder must remain frozen.")
    if teacher_cfg.get("input_mode") != "grayscale_replicated_rgb":
        raise ValueError("First-version distillation requires teacher.input_mode=grayscale_replicated_rgb.")
    datasets = create_distillation_datasets(dataset_cfg, seed=seed)
    cache_dir = resolve_cache_dir(config["teacher_cache"]["cache_dir"], EXPERIMENTS_ROOT)
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = [cache_dir / f"{split}_features.pt" for split in ("train", "val", "test")]
    if not overwrite and any(path.exists() for path in existing):
        raise FileExistsError(f"Teacher cache already exists under {cache_dir}; pass --overwrite to replace it.")
    teacher, backend = load_clip_image_encoder(teacher_cfg.get("model_name", "ViT-B/32"), device)
    payloads = {}
    for split, dataset in (
        ("train", datasets.train_dataset),
        ("val", datasets.val_dataset),
        ("test", datasets.test_dataset),
    ):
        print(f"encoding {split}: {len(dataset)} grayscale samples")
        payload = encode_split(dataset, teacher, device, dataset_cfg)
        payload["split"] = split
        torch.save(payload, cache_dir / f"{split}_features.pt")
        payloads[split] = payload
    teacher_dim = int(payloads["train"]["features"].shape[1])
    metadata = {
        "teacher_type": teacher_cfg.get("type", "clip_image_encoder"),
        "teacher_model_name": teacher_cfg.get("model_name", "ViT-B/32"),
        "teacher_backend": backend,
        "dataset_name": dataset_cfg.get("name"),
        "input_mode": "grayscale_replicated_rgb",
        "teacher_feature_dim": teacher_dim,
        "num_train": len(datasets.train_dataset),
        "num_val": len(datasets.val_dataset),
        "num_test": len(datasets.test_dataset),
        "config_hash_or_summary": dataset_config_hash(dataset_cfg, teacher_cfg, seed),
        "class_names": datasets.class_names,
        "teacher_text_encoder_used": False,
        "features_are_l2_normalized": True,
    }
    save_json(metadata, cache_dir / "metadata.json")
    print(f"teacher cache saved: {cache_dir} feature_dim={teacher_dim}")
    return metadata


def main():
    args = parse_args()
    config = load_yaml(args.config)
    if args.smoke_test:
        config["dataset"]["smoke_test"] = True
        apply_smoke_loader_overrides(config["dataset"])
    device = choose_device(args.device or config.get("device", "auto"))
    print(f"device: {device}")
    build_cache(config, device, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
