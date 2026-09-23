"""Prepare ABO morphology manifests and generate targets with InstructPix2Pix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image, ImageOps


MORPHOLOGIES = {
    "lamp": (
        "make this lamp moderately taller and slimmer and make its lampshade narrower, while keeping a clearly recognizable complete lamp",
        "make this lamp shorter and wider with a gently rounded shade and a wider circular base",
        "change this lamp to a conical lampshade and a slightly curved stem, preserving its complete lamp structure",
        "change this lamp to an oval lampshade, a thinner straight stem, and a flat round base",
    ),
    "table": (
        "make this table moderately lower and wider, with an oval top and four outward-tapered legs",
        "make this table slightly taller and narrower with a round top and one central pedestal support",
        "change this table to a round top with three slim angled legs, keeping the entire table recognizable",
        "make this table lower with a thicker rectangular top and two crossed support legs",
    ),
    "backpack": (
        "make this backpack slightly taller and slimmer with a roll-top opening, preserving its straps and pockets",
        "make this backpack more compact with a rounded flap top and one visible front buckle",
        "make this backpack slightly wider at the base with two visible side pockets and a structured outline",
        "make this backpack softer with a drawstring top and a small rounded front pouch, preserving the complete bag",
    ),
}


def _read_rows(root: Path, split: str, category: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row["category"] == category]


def _stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:4], "big")


def _center_white_background_product(image: Image.Image, size: int = 512) -> Image.Image:
    """Enlarge the non-white ABO render without synthesizing any output pixels."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    foreground = np.any(array < 245, axis=2)
    ys, xs = np.nonzero(foreground)
    if len(xs) == 0:
        return ImageOps.fit(rgb, (size, size), method=Image.Resampling.LANCZOS)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    pad_x = max(2, int((right - left) * 0.08))
    pad_y = max(2, int((bottom - top) * 0.08))
    crop = rgb.crop((max(0, left - pad_x), max(0, top - pad_y), min(rgb.width, right + pad_x), min(rgb.height, bottom + pad_y)))
    crop.thumbnail((410, 410), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas


def prepare(
    *, cleanrender: Path, backpack: Path, output: Path,
    train_sources: int, validation_sources: int, test_sources: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    limits = {"train": train_sources, "val": validation_sources, "test": test_sources}
    summary: dict[str, Any] = {"schema_version": 1, "splits": {}, "targets_per_source": 8}
    for split, limit in limits.items():
        manifest = []
        for category in ("lamp", "table", "backpack"):
            source_root = backpack if category == "backpack" else cleanrender
            available = _read_rows(source_root, split, category)
            # Deterministic but non-prefix selection avoids adjacent turntable views.
            available.sort(key=lambda row: _stable_seed(split, category, row["sample_id"]))
            chosen = available[:limit]
            if len(chosen) != limit:
                raise ValueError(f"{category}/{split}: need {limit}, have {len(chosen)}")
            for source in chosen:
                reference_rel = Path("references") / split / category / f"{source['sample_id']}.png"
                reference_path = output / reference_rel
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                original = Image.open(source_root / source["image_path"]).convert("RGB")
                image = (
                    _center_white_background_product(original)
                    if category in {"lamp", "table"}
                    else ImageOps.fit(original, (512, 512), method=Image.Resampling.LANCZOS)
                )
                image.save(reference_path)
                for local_index, instruction in enumerate(MORPHOLOGIES[category]):
                    morphology_index = tuple(MORPHOLOGIES).index(category) * 4 + local_index
                    prompt = (
                        instruction
                        + "; regenerate the complete single product coherently, fully visible with generous margins, centered on a clean premium neutral studio background, no extra objects or text"
                    )
                    for seed_offset in (0, 1):
                        seed = _stable_seed(source["sample_id"], str(local_index), str(seed_offset))
                        target_rel = Path("targets") / split / category / f"{source['sample_id']}__m{local_index}__s{seed_offset}.png"
                        manifest.append({
                            "sample_id": f"{source['sample_id']}:m{local_index}:s{seed_offset}",
                            "source_id": source["sample_id"],
                            "category": category,
                            "reference_path": reference_rel.as_posix(),
                            "target_path": target_rel.as_posix(),
                            "morphology_id": f"{category}-m{local_index}",
                            "morphology_index": morphology_index,
                            "prompt": prompt,
                            "seed": seed,
                            "license": source.get("license", "CC BY 4.0"),
                            "source_url": source.get("source_url", ""),
                            "teacher": "timbrooks/instruct-pix2pix",
                        })
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
        )
        summary["splits"][split] = len(manifest)
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


@torch.inference_mode()
def generate(
    *, data_dir: Path, model: Path, split: str, shard_index: int, num_shards: int,
    device: torch.device, steps: int, guidance_scale: float, image_guidance_scale: float,
    force: bool = False,
) -> dict[str, Any]:
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model, torch_dtype=dtype, safety_checker=None, local_files_only=True
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to(device)
    rows = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = [row for index, row in enumerate(rows) if index % num_shards == shard_index]
    written = skipped = 0
    for row in selected:
        target = data_dir / row["target_path"]
        if target.exists() and not force:
            skipped += 1
            continue
        source = Image.open(data_dir / row["reference_path"]).convert("RGB")
        result = pipe(
            prompt=row["prompt"], image=source, num_inference_steps=steps,
            guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale,
            generator=torch.Generator(device=device).manual_seed(int(row["seed"])),
        ).images[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target)
        written += 1
        if written % 10 == 0:
            print(json.dumps({"split": split, "shard": shard_index, "written": written}), flush=True)
    return {"split": split, "shard": shard_index, "selected": len(selected), "written": written, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--cleanrender", type=Path, required=True)
    prep.add_argument("--backpack", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--train-sources", type=int, default=12)
    prep.add_argument("--validation-sources", type=int, default=3)
    prep.add_argument("--test-sources", type=int, default=3)
    gen = commands.add_parser("generate")
    gen.add_argument("--data-dir", type=Path, required=True)
    gen.add_argument("--model", type=Path, required=True)
    gen.add_argument("--split", choices=("train", "val", "test"), required=True)
    gen.add_argument("--shard-index", type=int, required=True)
    gen.add_argument("--num-shards", type=int, default=1)
    gen.add_argument("--device", default="cuda")
    gen.add_argument("--steps", type=int, default=20)
    gen.add_argument("--guidance-scale", type=float, default=7.5)
    gen.add_argument("--image-guidance-scale", type=float, default=1.25)
    gen.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            cleanrender=args.cleanrender.resolve(), backpack=args.backpack.resolve(),
            output=args.output.resolve(), train_sources=args.train_sources,
            validation_sources=args.validation_sources, test_sources=args.test_sources,
        )
    else:
        result = generate(
            data_dir=args.data_dir.resolve(), model=args.model.resolve(), split=args.split,
            shard_index=args.shard_index, num_shards=args.num_shards,
            device=torch.device(args.device), steps=args.steps,
            guidance_scale=args.guidance_scale, image_guidance_scale=args.image_guidance_scale,
            force=args.force,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
