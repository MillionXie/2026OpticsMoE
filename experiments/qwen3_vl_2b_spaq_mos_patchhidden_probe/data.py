from __future__ import annotations

from typing import Any

from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.data_prepare import ensure_spaq_dataset
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.datasets import DatasetBundle, load_spaq
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.settings import Settings as SourceSettings


def load_probe_data(settings: Any) -> tuple[DatasetBundle, SourceSettings]:
    source = SourceSettings(
        data_root=settings.data_root,
        annotations_file=settings.annotations_file,
        image_dir=settings.image_dir,
        download=settings.download,
        train_fraction=settings.train_fraction,
        train_image_limit=settings.train_image_limit,
        test_image_limit=settings.test_image_limit,
        output_dir=settings.output_dir,
        model_id=settings.model_id,
        cache_dir=settings.cache_dir,
        local_files_only=settings.local_files_only,
        processor_min_pixels=settings.processor_min_pixels,
        processor_max_pixels=settings.processor_max_pixels,
        seed=settings.seed,
    )
    source.validate()
    ensure_spaq_dataset(source)
    bundle = load_spaq(source, persist_split=True)
    source.resolved_annotations_file = bundle.metadata["annotation_file"]
    source.split_digest = bundle.metadata["split_digest"]
    return bundle, source

