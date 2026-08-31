from __future__ import annotations

from pathlib import Path

import torch
import yaml

from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.stem import (
    STEM_FORMAT,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.model import (
    P11DownstreamModel,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.scratch_source import (
    SOURCE_FORMAT,
    SOURCE_REGIME,
    export_random_p11_source,
    inspect_scratch_source,
    render_scratch_config,
    sha256_file,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.settings import (
    EXPERIMENT_DIR,
    load_settings,
)


BASE_CONFIG = EXPERIMENT_DIR / "configs" / "base_50e.yaml"


def _fake_stem(path: Path) -> Path:
    generator = torch.Generator().manual_seed(907)
    torch.save(
        {
            "format": STEM_FORMAT,
            "conv2d_weight": torch.randn(
                1024, 3, 16, 16, generator=generator
            )
            * 0.001,
            "conv2d_bias": torch.zeros(1024),
            "position_embedding": torch.randn(
                196, 1024, generator=generator
            )
            * 0.01,
            "metadata": {
                "image_size": 224,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            },
        },
        path,
    )
    return path


def _p11_config(seed: int = 2026) -> dict[str, object]:
    config = load_settings(BASE_CONFIG).p11_config
    config["seed"] = seed
    return config


def _payload(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(value, dict)
    return value


def test_scratch_source_is_reproducible_fresh_and_strictly_loadable(
    tmp_path: Path,
) -> None:
    stem = _fake_stem(tmp_path / "stem.pt")
    first_path = tmp_path / "scratch_2026_a.pt"
    second_path = tmp_path / "scratch_2026_b.pt"
    other_path = tmp_path / "scratch_2027.pt"

    first = export_random_p11_source(
        stem_checkpoint=stem,
        output=first_path,
        p11_config=_p11_config(),
        init_seed=2026,
    )
    second = export_random_p11_source(
        stem_checkpoint=stem,
        output=second_path,
        p11_config=_p11_config(),
        init_seed=2026,
    )
    other = export_random_p11_source(
        stem_checkpoint=stem,
        output=other_path,
        p11_config=_p11_config(),
        init_seed=2027,
    )

    assert first["source_regime"] == SOURCE_REGIME
    assert first["init_seed"] == 2026
    assert first["stem_checkpoint_sha256"] == sha256_file(stem)
    assert first["backbone_state_sha256"] == second["backbone_state_sha256"]
    assert first["stem_state_sha256"] == second["stem_state_sha256"]
    assert first["non_stem_state_sha256"] == second["non_stem_state_sha256"]
    assert first["stem_state_sha256"] == other["stem_state_sha256"]
    assert first["non_stem_state_sha256"] != other["non_stem_state_sha256"]

    # Re-export to the same path is safe and idempotent; it cannot silently
    # replace a source created by a different initialization.
    reused = export_random_p11_source(
        stem_checkpoint=stem,
        output=first_path,
        p11_config=_p11_config(),
        init_seed=2026,
    )
    assert reused["status"] == "reused"

    first_payload = _payload(first_path)
    other_payload = _payload(other_path)
    assert first_payload["format"] == SOURCE_FORMAT
    assert first_payload["source_regime"] == SOURCE_REGIME
    assert first_payload["initialization"]["copied_from_imagenet_pretraining"] is False
    first_state = first_payload["backbone"]
    other_state = other_payload["backbone"]
    assert torch.equal(
        first_state["p11_separable_architecture_signature"],
        torch.tensor([11, 1, 2, 4]),
    )
    stem_keys = [name for name in first_state if name.startswith("stem.")]
    assert stem_keys
    assert all(torch.equal(first_state[name], other_state[name]) for name in stem_keys)
    assert not torch.equal(
        first_state["adapter.projection.weight"],
        other_state["adapter.projection.weight"],
    )
    assert not torch.equal(
        first_state["stages.0.raw_phase"], other_state["stages.0.raw_phase"]
    )
    mixer_key = next(
        name
        for name in first_state
        if name.startswith("stages.0.electronic_skip.")
        and name.endswith("weight")
    )
    assert not torch.equal(first_state[mixer_key], other_state[mixer_key])

    downstream = P11DownstreamModel(
        stem_checkpoint=stem,
        source_checkpoint=first_path,
        p11_config=_p11_config(),
        task="caltech101",
    )
    assert downstream.source_manifest["source_regime"] == SOURCE_REGIME
    assert downstream.source_manifest["init_seed"] == 2026
    assert downstream.source_manifest["backbone_state_sha256"] == first[
        "backbone_state_sha256"
    ]
    assert torch.equal(
        downstream.backbone.p11_separable_architecture_signature.cpu(),
        torch.tensor([11, 1, 2, 4]),
    )
    assert downstream.backbone.stem.checkpoint_sha256 == sha256_file(stem)


def test_rendered_config_uses_real_sha_and_an_isolated_output_root(
    tmp_path: Path,
) -> None:
    stem = _fake_stem(tmp_path / "stem.pt")
    source_path = tmp_path / "scratch.pt"
    source = export_random_p11_source(
        stem_checkpoint=stem,
        output=source_path,
        p11_config=_p11_config(),
        init_seed=2026,
    )

    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    raw["paths"]["stem_checkpoint"] = str(stem)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    output_root = tmp_path / "isolated_scratch_runs"
    rendered_path = tmp_path / "resolved_scratch.yaml"
    render_scratch_config(
        base_config=base_path,
        source_checkpoint=source_path,
        output=rendered_path,
        output_root=output_root,
    )

    rendered_raw = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
    assert rendered_raw["paths"]["source_backbone_sha256"] == sha256_file(
        source_path
    )
    assert rendered_raw["scratch_control"] == {
        "source_regime": SOURCE_REGIME,
        "init_seed": 2026,
        "no_imagenet_backbone_pretraining": True,
        "frozen_qwen_stem_checkpoint_sha256": sha256_file(stem),
        "backbone_state_sha256": source["backbone_state_sha256"],
        "non_stem_state_sha256": source["non_stem_state_sha256"],
        "fa_pretrained_method_label_in_this_control": "fa_source_init",
    }
    settings = load_settings(rendered_path, task="lsp", method="fa_random", seed=2028)
    assert settings.paths.source_backbone == source_path.resolve()
    assert settings.paths.source_backbone_sha256 == sha256_file(source_path)
    assert settings.paths.output_root == output_root.resolve()
    assert settings.paths.output_root != load_settings(BASE_CONFIG).paths.output_root
    assert inspect_scratch_source(settings.paths.source_backbone)["init_seed"] == 2026
