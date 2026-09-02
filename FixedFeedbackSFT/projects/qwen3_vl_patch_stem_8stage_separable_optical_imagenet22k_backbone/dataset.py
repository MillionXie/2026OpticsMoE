"""Manifest-driven, bounded-memory class-folder input for ImageNet-21K/22K.

The real Fall11 corpus contains more than fourteen million images.  A normal
``ImageFolder`` instance materializes one Python tuple per image and is not an
acceptable long-running representation here.  This module builds a one-time
on-disk TSV plus uint64 offsets, then memory maps both files in every worker.

Dataset identity is never inferred from a directory name.  The data owner must
provide an explicit declaration containing the release, variant, WNID order,
class count and sample count.  Known public recipe names only add consistency
checks; the declaration remains the source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import random
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler


INDEX_FORMAT = "optical-large-class-folder-index-v1"
SOURCE_DECLARATION_FORMAT = "authorized-class-folder-source-v1"
UINT64 = struct.Struct("<Q")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

# These checks prevent the most damaging taxonomy mix-ups.  MIIL sample counts
# are intentionally absent: they differ by exact release and must come from the
# owner's declaration rather than an approximate number in code.
KNOWN_VARIANTS: dict[str, dict[str, Any]] = {
    "imagenet-fall11-full": {
        "release_id": "fall11",
        "expected_num_classes": 21_841,
        "expected_num_samples": 14_197_122,
        "official_validation_split": False,
    },
    "miil-imagenet21k-p-fall11": {
        "release_id": "fall11",
        "expected_num_classes": 11_221,
        "expected_num_samples": None,
        "expected_samples_by_split": {
            "train": 11_797_632,
            "validation": 561_052,
            "val": 561_052,
            "test": 561_052,
        },
        "official_validation_split": True,
    },
    "miil-imagenet21k-p-winter21": {
        "release_id": "winter21",
        "expected_num_classes": 10_450,
        "expected_num_samples": None,
        "official_validation_split": True,
    },
}


class DatasetContractError(RuntimeError):
    """Raised when a data identity or index integrity contract is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class SourceDeclaration:
    path: Path
    raw: dict[str, Any]
    variant_id: str
    release_id: str
    split_id: str
    expected_num_classes: int
    expected_num_samples: int
    wnids: tuple[str, ...]
    declaration_sha256: str
    class_list_sha256: str


def load_source_declaration(
    path: str | Path,
    *,
    strict_known_variant: bool = True,
) -> SourceDeclaration:
    declaration_path = Path(path).expanduser().resolve()
    raw = _load_json(declaration_path)
    if not isinstance(raw, dict) or raw.get("format") != SOURCE_DECLARATION_FORMAT:
        raise DatasetContractError(
            f"Expected {SOURCE_DECLARATION_FORMAT!r}: {declaration_path}"
        )
    if raw.get("license_or_access_acknowledged") is not True:
        raise DatasetContractError("The source declaration must acknowledge authorized access")
    required_text = ("dataset_id", "variant_id", "release_id", "split_id")
    for key in required_text:
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise DatasetContractError(f"Source declaration requires non-empty {key}")
    try:
        expected_classes = int(raw["expected_num_classes"])
        expected_samples = int(raw["expected_num_samples"])
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetContractError(
            "Source declaration requires exact integer class and sample counts"
        ) from error
    if expected_classes <= 1 or expected_samples <= 0:
        raise DatasetContractError("Declared class/sample counts must be positive")

    inline = raw.get("class_wnids")
    class_file = raw.get("class_list_file")
    if (inline is None) == (class_file is None):
        raise DatasetContractError(
            "Declare exactly one of class_wnids or class_list_file"
        )
    if inline is not None:
        if not isinstance(inline, list):
            raise DatasetContractError("class_wnids must be a JSON list")
        wnids = tuple(str(value).strip() for value in inline)
        class_bytes = ("\n".join(wnids) + "\n").encode("utf-8")
        class_sha = hashlib.sha256(class_bytes).hexdigest()
    else:
        class_path = (declaration_path.parent / str(class_file)).resolve()
        if not class_path.is_file():
            raise FileNotFoundError(f"Declared WNID list is missing: {class_path}")
        class_bytes = class_path.read_bytes()
        try:
            wnids = tuple(
                line.strip()
                for line in class_bytes.decode("utf-8").splitlines()
                if line.strip()
            )
        except UnicodeDecodeError as error:
            raise DatasetContractError("WNID list must be UTF-8") from error
        class_sha = hashlib.sha256(class_bytes).hexdigest()
    if len(wnids) != expected_classes:
        raise DatasetContractError(
            f"Declared {expected_classes} classes but WNID order has {len(wnids)}"
        )
    if len(set(wnids)) != len(wnids) or any("/" in value or "\\" in value for value in wnids):
        raise DatasetContractError("WNIDs must be unique single path components")

    variant_id = str(raw["variant_id"])
    known = KNOWN_VARIANTS.get(variant_id)
    if known is None and strict_known_variant:
        raise DatasetContractError(
            f"Unknown large-data variant {variant_id!r}; add a reviewed identity before use"
        )
    if known is not None:
        checks = {
            "release_id": str(raw["release_id"]),
            "expected_num_classes": expected_classes,
        }
        for key, actual in checks.items():
            if actual != known[key]:
                raise DatasetContractError(
                    f"{variant_id} has {key}={actual!r}, expected {known[key]!r}"
                )
        known_samples = known.get("expected_num_samples")
        split_samples = known.get("expected_samples_by_split", {}).get(str(raw["split_id"]))
        if split_samples is not None:
            known_samples = int(split_samples)
        if known_samples is not None and expected_samples != int(known_samples):
            raise DatasetContractError(
                f"{variant_id} must contain exactly {int(known_samples):,} samples"
            )
    return SourceDeclaration(
        path=declaration_path,
        raw=dict(raw),
        variant_id=variant_id,
        release_id=str(raw["release_id"]),
        split_id=str(raw["split_id"]),
        expected_num_classes=expected_classes,
        expected_num_samples=expected_samples,
        wnids=wnids,
        declaration_sha256=sha256_file(declaration_path),
        class_list_sha256=class_sha,
    )


def _iter_class_images(directory: Path) -> list[Path]:
    # One class is sorted at a time.  This bounds memory by the largest class,
    # not by all 14.2M image paths.
    paths = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    paths.sort(key=lambda value: value.relative_to(directory).as_posix())
    return paths


def build_class_folder_index(
    *,
    source_root: str | Path,
    declaration_path: str | Path,
    output_dir: str | Path,
    strict_known_variant: bool = True,
) -> dict[str, Any]:
    """Build and verify a one-time disk index without an all-sample Python list."""

    root = Path(source_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Class-folder source root is missing: {root}")
    declaration = load_source_declaration(
        declaration_path,
        strict_known_variant=strict_known_variant,
    )
    if output.exists():
        existing = verify_index(output)
        if existing["source_declaration_sha256"] != declaration.declaration_sha256:
            raise DatasetContractError(
                "Index directory already belongs to a different source declaration"
            )
        return existing

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.build.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise DatasetContractError(f"Another index build owns {lock}") from error
    os.close(descriptor)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        class_to_idx = {wnid: index for index, wnid in enumerate(declaration.wnids)}
        class_bytes = canonical_json_bytes(class_to_idx) + b"\n"
        (staging / "class_to_idx.json").write_bytes(class_bytes)
        class_counts: list[int] = []
        sample_digest = hashlib.sha256()
        offset_digest = hashlib.sha256()
        offset = 0
        sample_count = 0
        offset_buffer = bytearray()
        with (staging / "samples.tsv").open("wb") as samples, (
            staging / "offsets.u64"
        ).open("wb") as offsets:
            for class_index, wnid in enumerate(declaration.wnids):
                directory = root / wnid
                if not directory.is_dir():
                    raise DatasetContractError(f"Declared class directory is missing: {directory}")
                images = _iter_class_images(directory)
                class_counts.append(len(images))
                for image_path in images:
                    relative = image_path.relative_to(root).as_posix()
                    if "\t" in relative or "\n" in relative:
                        raise DatasetContractError(f"Unsupported control character in path: {relative!r}")
                    packed = UINT64.pack(offset)
                    offset_buffer.extend(packed)
                    if len(offset_buffer) >= 8 * 65_536:
                        offsets.write(offset_buffer)
                        offset_digest.update(offset_buffer)
                        offset_buffer.clear()
                    line = f"{relative}\t{class_index}\n".encode("utf-8")
                    samples.write(line)
                    sample_digest.update(line)
                    offset += len(line)
                    sample_count += 1
            offset_buffer.extend(UINT64.pack(offset))
            offsets.write(offset_buffer)
            offset_digest.update(offset_buffer)
            samples.flush()
            offsets.flush()
            os.fsync(samples.fileno())
            os.fsync(offsets.fileno())
        if sample_count != declaration.expected_num_samples:
            raise DatasetContractError(
                f"Scanned {sample_count:,} samples; declaration requires "
                f"{declaration.expected_num_samples:,}"
            )
        _write_json(staging / "class_counts.json", class_counts)
        manifest = {
            "format": INDEX_FORMAT,
            "dataset_id": declaration.raw["dataset_id"],
            "variant_id": declaration.variant_id,
            "release_id": declaration.release_id,
            "split_id": declaration.split_id,
            "source_root": str(root),
            "source_declaration": str(declaration.path),
            "source_declaration_sha256": declaration.declaration_sha256,
            "class_list_sha256": declaration.class_list_sha256,
            "num_classes": declaration.expected_num_classes,
            "num_samples": sample_count,
            "class_to_idx_sha256": hashlib.sha256(class_bytes).hexdigest(),
            "class_counts_sha256": sha256_file(staging / "class_counts.json"),
            "samples_tsv_sha256": sample_digest.hexdigest(),
            "offsets_u64_sha256": offset_digest.hexdigest(),
            "offset_count": sample_count + 1,
            "ordering": "declaration WNID order; lexical path order within class",
            "runtime_sampler": "global affine permutation before DDP rank sharding",
        }
        _write_json(staging / "index_manifest.json", manifest)
        os.replace(staging, output)
        return verify_index(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def verify_index(index_dir: str | Path, *, verify_large_files: bool = True) -> dict[str, Any]:
    directory = Path(index_dir).expanduser().resolve()
    manifest_path = directory / "index_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Index manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("format") != INDEX_FORMAT:
        raise DatasetContractError(f"Unsupported index format: {manifest_path}")
    required = {
        "class_to_idx.json": "class_to_idx_sha256",
        "class_counts.json": "class_counts_sha256",
        "samples.tsv": "samples_tsv_sha256",
        "offsets.u64": "offsets_u64_sha256",
    }
    for filename, key in required.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Index component is missing: {path}")
        if verify_large_files or filename not in {"samples.tsv", "offsets.u64"}:
            if sha256_file(path) != manifest.get(key):
                raise DatasetContractError(f"Index checksum mismatch: {path}")
    sample_count = int(manifest["num_samples"])
    if (directory / "offsets.u64").stat().st_size != (sample_count + 1) * UINT64.size:
        raise DatasetContractError("Offset file length does not match num_samples")
    class_map = _load_json(directory / "class_to_idx.json")
    if len(class_map) != int(manifest["num_classes"]):
        raise DatasetContractError("class_to_idx length does not match num_classes")
    if sorted(int(value) for value in class_map.values()) != list(range(len(class_map))):
        raise DatasetContractError("class_to_idx must be a contiguous fixed mapping")
    manifest = dict(manifest)
    manifest["index_manifest_sha256"] = sha256_file(manifest_path)
    manifest["index_dir"] = str(directory)
    return manifest


def image_transform(*, train: bool, image_size: int = 224) -> Callable[[Any], torch.Tensor]:
    try:
        from torchvision import transforms
    except Exception as error:
        raise RuntimeError("torchvision is required for image preprocessing") from error
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.08, 1.0),
                    ratio=(0.75, 4.0 / 3.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    num_ops=2,
                    magnitude=9,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(
                int(round(image_size / 0.875)),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class ClassFolderMMapDataset(Dataset):
    """Map an indexed sample to an image without keeping paths in Python RAM."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        transform: Callable[[Any], torch.Tensor] | None,
        verify_large_files: bool = False,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.manifest = verify_index(
            self.index_dir,
            verify_large_files=verify_large_files,
        )
        self.source_root = Path(self.manifest["source_root"])
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"Indexed source root is unavailable: {self.source_root}")
        self.transform = transform
        self._samples_handle = None
        self._offsets_handle = None
        self._samples_map: mmap.mmap | None = None
        self._offsets_map: mmap.mmap | None = None

    def __len__(self) -> int:
        return int(self.manifest["num_samples"])

    def _ensure_open(self) -> None:
        if self._samples_map is not None:
            return
        self._samples_handle = (self.index_dir / "samples.tsv").open("rb")
        self._offsets_handle = (self.index_dir / "offsets.u64").open("rb")
        self._samples_map = mmap.mmap(self._samples_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._offsets_map = mmap.mmap(self._offsets_handle.fileno(), 0, access=mmap.ACCESS_READ)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        for key in ("_samples_handle", "_offsets_handle", "_samples_map", "_offsets_map"):
            state[key] = None
        return state

    def record(self, index: int) -> tuple[str, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        self._ensure_open()
        assert self._samples_map is not None and self._offsets_map is not None
        start = UINT64.unpack_from(self._offsets_map, index * UINT64.size)[0]
        stop = UINT64.unpack_from(self._offsets_map, (index + 1) * UINT64.size)[0]
        raw = self._samples_map[start:stop].rstrip(b"\n")
        relative_bytes, label_bytes = raw.rsplit(b"\t", 1)
        return relative_bytes.decode("utf-8"), int(label_bytes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        relative, label = self.record(int(index))
        try:
            from PIL import Image
        except Exception as error:
            raise RuntimeError("Pillow is required to decode indexed images") from error
        path = self.source_root / relative
        with Image.open(path) as image:
            image = image.convert("RGB")
            value = self.transform(image) if self.transform is not None else image.copy()
        return {"image": value, "label": label, "index": int(index), "path": str(path)}


def _seed64(seed: int, epoch: int, salt: str) -> int:
    payload = f"global-affine-v1:{seed}:{epoch}:{salt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def affine_permutation_parameters(size: int, seed: int, epoch: int) -> tuple[int, int]:
    if size <= 1:
        return 1, 0
    candidate = (_seed64(seed, epoch, "multiplier") % (size - 1)) + 1
    # Favor large global jumps so a class-major index cannot yield 4096-image
    # same-class chunks after sharding.  The loop remains O(log n) in practice.
    if candidate < max(2, size // 32):
        candidate += max(2, size // 3)
    candidate %= size
    candidate = max(candidate, 1)
    while math.gcd(candidate, size) != 1:
        candidate = (candidate + 1) % size
        if candidate == 0:
            candidate = 1
    offset = _seed64(seed, epoch, "offset") % size
    return candidate, offset


class GlobalAffineDistributedSampler(Sampler[int]):
    """Constant-memory global shuffle followed by deterministic DDP sharding.

    The affine map is a bijection over ``[0, N)``.  Global positions are
    permuted before ranks take their strided shares, unlike a block shuffle
    that can feed thousands of neighboring images from one WNID to a rank.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle: bool = True,
        pad_to_even: bool = True,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid DDP rank/world_size")
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.pad_to_even = bool(pad_to_even)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        size = len(self.dataset)
        if self.pad_to_even:
            return math.ceil(size / self.world_size)
        return max((size - self.rank + self.world_size - 1) // self.world_size, 0)

    def __iter__(self) -> Iterator[int]:
        size = len(self.dataset)
        if size == 0:
            return
        multiplier, offset = affine_permutation_parameters(size, self.seed, self.epoch)
        total = math.ceil(size / self.world_size) * self.world_size if self.pad_to_even else size
        for global_position in range(self.rank, total, self.world_size):
            base = global_position % size
            yield (multiplier * base + offset) % size if self.shuffle else base


class PlumbingImageNet1KDataset(Dataset):
    """Small, explicitly non-publishable view of the existing IN-1K cache."""

    def __init__(self, base: Dataset, *, base_sample_count: int, views: int, limit: int, seed: int) -> None:
        self.base = base
        self.base_sample_count = int(base_sample_count)
        self.views = int(views)
        self.limit = min(int(limit), self.base_sample_count)
        if self.limit <= 0:
            raise ValueError("plumbing_sample_limit must be positive")
        self.multiplier, self.offset = affine_permutation_parameters(
            self.base_sample_count,
            int(seed),
            0,
        )

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = (self.multiplier * int(index) + self.offset) % self.base_sample_count
        record = self.base[source * self.views]
        return {"image": record["image"], "label": int(record["label"]), "source_index": source}


def load_plumbing_imagenet1k(config_path: str | Path, *, limit: int, seed: int):
    from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.datasets import (
        load_imagenet,
    )
    from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.settings import (
        load_settings,
    )

    settings = load_settings(config_path)
    bundle = load_imagenet(settings)
    dataset = PlumbingImageNet1KDataset(
        bundle.train,
        base_sample_count=bundle.train.base_sample_count,
        views=bundle.train.views,
        limit=limit,
        seed=seed,
    )
    identity = {
        "mode": "plumbing_smoke_only",
        "publishable_result": False,
        "source_dataset": "ImageNet-1K cache (not ImageNet-22K)",
        "source_digest": bundle.digest,
        "source_base_samples": bundle.train.base_sample_count,
        "selected_samples": len(dataset),
    }
    return dataset, identity


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/audit a large class-folder disk index")
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--declaration", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--index", type=Path, required=True)
    audit.add_argument("--fast", action="store_true", help="Skip re-hashing the large TSV/offset files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "build":
        manifest = build_class_folder_index(
            source_root=args.source_root,
            declaration_path=args.declaration,
            output_dir=args.output,
        )
    else:
        manifest = verify_index(args.index, verify_large_files=not args.fast)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
