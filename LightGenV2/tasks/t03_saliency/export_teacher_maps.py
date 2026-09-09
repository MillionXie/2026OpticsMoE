"""Export only TRAIN predictions from the completed aligned frozen-Qwen baseline."""
import argparse
import json
import subprocess
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .aligned_baseline import AlignedReadout
from .modeling import load_vision_backbone, sha256_file
from .settings import load_settings
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon, SALICONSaliencyDataset, collate_salicon, _annotation_path
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.training import preprocess_vision
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import FrozenQwenVisionTeacher


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    s = load_settings(args.config)
    if sha256_file(args.checkpoint) != s.distillation_teacher_sha256:
        raise ValueError("Teacher checkpoint SHA mismatch")
    bundle = prepare_salicon(s, persist=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_vision_backbone(s, device)
    s.resolve_architecture(loaded.model)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload["architecture"] != "frozen_qwen24_adapter192_identical_progressive_decoder_v1":
        raise ValueError("Wrong teacher architecture")
    loaded.model.requires_grad_(False).eval()
    head = AlignedReadout(s.vision_hidden_size).to(device)
    head.load_state_dict(payload["head"], strict=True)
    model = FrozenQwenVisionTeacher(loaded, head).eval()
    records = bundle.train_records
    loader = DataLoader(SALICONSaliencyDataset(records, s, training=False),
                        batch_size=s.inference_batch_size, shuffle=False,
                        num_workers=s.num_workers, collate_fn=collate_salicon)
    maps = torch.empty(len(records), 1, s.image_size, s.image_size, dtype=torch.float16)
    ids, offset = [], 0
    try:
        for batch in loader:
            inputs = preprocess_vision(loaded.processor, batch["images"], device)
            logits = model(inputs["pixel_values"], inputs["image_grid_thw"])[0]
            count = len(batch["sample_ids"])
            maps[offset:offset+count].copy_(logits.cpu().half())
            ids.extend(batch["sample_ids"])
            offset += count
            print(f"[train teacher maps] {offset}/{len(records)}", flush=True)
    finally:
        model.close()
    assert ids == [r.sample_id for r in records]
    manifest = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
                "image_size": s.image_size, "augmentation": False, "split": "train_only",
                "samples": len(ids), "teacher_selected_on_public_test": True,
                "image_manifest_sha256": __import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest(),
                "train_annotations_sha256": sha256_file(_annotation_path(s.data_root, 'train'))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    torch.save({"manifest": manifest, "sample_ids": ids, "logits": maps}, temporary)
    temporary.replace(args.output)
    manifest["cache_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
