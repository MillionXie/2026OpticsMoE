from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from ..hardware_bridge import _write_language_global_offline_payload
from ..offline_quick_finetune import (
    finetune_offline_quick,
    load_offline_quick_data,
    load_offline_tail,
)
from ..offline_tail import LanguageGlobalOfflineTail
from ..optical_blocks import LanguageTwoBlockOpticalCore


def _construction(
    width: int = 4,
    max_tokens: int = 5,
    minimum_optical_fusion: float = 0.1,
) -> dict[str, object]:
    return {
        "width": width,
        "max_tokens": max_tokens,
        "expansion": 2.0,
        "dropout": 0.1,
        "initial_residual_weight": 0.1,
        "token_mixer_enabled": True,
        "token_mixer_type": "depthwise_conv1d",
        "token_mixer_kernel_size": 3,
        "detector_size": 478,
        "detector_output_size": max_tokens,
        "detector_layernorm_eps": 1.0e-5,
        "detector_layernorm_affine": False,
        "detector_layernorm_scope": "per_token",
        "detector_nonlinearity": "relu",
        "ccd_relative_clip": 12.0,
        "ccd_log_compression": 1.0,
        "minimum_optical_fusion": minimum_optical_fusion,
        "embedding_dim": 3,
    }


def _settings(construction: dict[str, object], class_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        electronic_expansion=construction["expansion"],
        electronic_dropout=construction["dropout"],
        electronic_initial_residual_weight=construction["initial_residual_weight"],
        electronic_token_mixer_enabled=construction["token_mixer_enabled"],
        electronic_language_token_mixer_kernel_size=construction[
            "token_mixer_kernel_size"
        ],
        hardware_ccd_target_size=construction["detector_size"],
        input_adapter_dim=construction["detector_output_size"],
        detector_layernorm_eps=construction["detector_layernorm_eps"],
        detector_layernorm_affine=construction["detector_layernorm_affine"],
        detector_layernorm_scope=construction["detector_layernorm_scope"],
        detector_nonlinearity=construction["detector_nonlinearity"],
        language_optical_normalization_clip=construction["ccd_relative_clip"],
        language_optical_log_compression=construction["ccd_log_compression"],
        optical_fusion_minimum=construction["minimum_optical_fusion"],
        embedding_dim=construction["embedding_dim"],
        hardware_ccd_flip_vertical=True,
        hardware_ccd_flip_horizontal=True,
        random_seed=7,
        pk_skus_per_batch=class_count,
        pk_images_per_sku=2,
        learning_rate=1.0e-4,
        readout_learning_rate=5.0e-5,
        weight_decay=0.01,
        temperature=0.07,
        gallery_temperature=0.15,
        gallery_aggregation="mean_prototype",
    )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_session(
    root: Path,
    class_count: int = 2,
    architecture: str = "vision2_language2_moe4_10cm_robust_bounded_fusion_v2",
) -> tuple[Path, list[dict[str, object]]]:
    minimum = 0.05 if "warmstart5" in architecture else 0.10
    construction = _construction(minimum_optical_fusion=minimum)
    settings = _settings(construction, class_count)
    session = root / "session"
    stage = session / "04_language_global"
    stage.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    groups: list[torch.Tensor] = []
    splits = (("gallery", 1), ("train", 2), ("test", 1))
    order = 0
    generator = torch.Generator().manual_seed(11)
    for split, count in splits:
        for label in range(class_count):
            for item in range(count):
                key = f"{split}__{label:02d}__sample_{item:02d}"
                rows.append(
                    {
                        "order": order,
                        "key": key,
                        "sample_id": key,
                        "split": split,
                        "sku_index": label,
                        "sku_name": f"class_{label}",
                        "image_path": f"unused/{key}.jpg",
                    }
                )
                groups.append(torch.randn(3 + item % 2, 4, generator=generator))
                order += 1
    _write_manifest(session / "manifest.csv", rows)
    checkpoint = root / "source_checkpoint.pt"
    checkpoint.write_bytes(b"unit-test-checkpoint")
    tail = LanguageGlobalOfflineTail(**construction)
    fake_core = SimpleNamespace(
        width=4,
        max_tokens=5,
        blocks=[None, tail.block2],
        minimum_optical_fusion=minimum,
        optical_branch=SimpleNamespace(
            ccd_normalizer=SimpleNamespace(
                active_size=478,
                relative_clip=12.0,
                log_compression=1.0,
            ),
            core=SimpleNamespace(readout=tail.ccd_readout),
        ),
    )
    replacement = SimpleNamespace(
        checkpoint_architecture=architecture,
        language_surrogate=SimpleNamespace(core=fake_core),
    )
    _write_language_global_offline_payload(
        settings=settings,
        replacement=replacement,
        checkpoint=checkpoint,
        session_dir=session,
        destination=stage,
        rows=rows,
        block2_input_groups=groups,
        tail_state={
            name: value.detach().cpu().clone()
            for name, value in tail.state_dict().items()
        },
        upstream_source="simulation",
        measured_upstream_stages=(),
    )
    ccd_dir = stage / "ccd_captured"
    ccd_dir.mkdir()
    for index, row in enumerate(rows):
        array = np.full((478, 478), 30 + index, dtype=np.uint8)
        array[0, 0] = index
        Image.fromarray(array, mode="L").save(ccd_dir / f"{row['key']}.png")
    return session, rows


def test_offline_tail_has_formal_parameter_count() -> None:
    construction = _construction(width=192, max_tokens=224)
    construction["token_mixer_kernel_size"] = 5
    construction["detector_output_size"] = 224
    construction["embedding_dim"] = 64
    tail = LanguageGlobalOfflineTail(**construction)
    assert sum(parameter.numel() for parameter in tail.parameters()) == 255_811


def test_batched_core_helper_matches_lightweight_tail() -> None:
    tail = LanguageGlobalOfflineTail(**_construction())
    tail.eval()
    groups = [torch.randn(3, 4), torch.randn(5, 4)]
    ccd = torch.rand(2, 478, 478)

    def decode(
        intensity: torch.Tensor, padding_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        readout = tail.ccd_readout(tail._normalize_ccd(intensity))
        delta = tail.optical_output_adapter(readout[:, : padding_mask.shape[1]])
        return delta.to(dtype).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    fake_core = SimpleNamespace(
        width=tail.width,
        max_tokens=tail.max_tokens,
        blocks=[None, tail.block2],
        optical_branch=SimpleNamespace(decode_measured_ccd=decode),
        output_norm=tail.output_norm,
        block2_optical_fusion=tail.block2_optical_fusion,
    )
    expected = tail.detector_features(groups, ccd)
    observed = LanguageTwoBlockOpticalCore.detector_features_from_block2_inputs(
        fake_core, groups, ccd
    )
    torch.testing.assert_close(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_offline_tail_batching_is_deterministic_and_keeps_gradients() -> None:
    tail = LanguageGlobalOfflineTail(**_construction())
    tail.eval()
    groups = [torch.randn(3, 4), torch.randn(5, 4)]
    ccd = torch.rand(2, 478, 478)
    batched = tail.detector_features(groups, ccd)
    repeated = tail.detector_features(groups, ccd)
    individual = torch.cat(
        [
            tail.detector_features([group], ccd[index : index + 1])
            for index, group in enumerate(groups)
        ],
        dim=0,
    )
    torch.testing.assert_close(batched, repeated, atol=0.0, rtol=0.0)
    torch.testing.assert_close(batched, individual, atol=1.0e-5, rtol=1.0e-5)
    loss = batched.square().mean()
    loss.backward()
    required = (
        tail.block2.token_pointwise.weight,
        tail.optical_output_adapter.weight,
        tail.output_norm.weight,
        tail.block2_optical_fusion_logit,
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in required
    )


def test_exported_packed_contract_round_trips_and_applies_configured_flips(
    tmp_path: Path,
) -> None:
    session, rows = _make_session(tmp_path)
    data = load_offline_quick_data(session)
    assert len(data.rows) == len(rows) == 8
    assert tuple(data.ccd_uint8.shape) == (8, 478, 478)
    # Source [0,0] moves to the opposite corner after both configured flips.
    assert int(data.ccd_uint8[3, -1, -1]) == 3
    assert {row["split"] for row in data.rows} == {"train", "gallery", "test"}
    tail = load_offline_tail(data, torch.device("cpu"))
    assert sum(parameter.numel() for parameter in tail.parameters()) == int(
        data.contract["tail_trainable_parameter_count"]
    )
    assert tail.training is False


def test_warmstart5_contract_uses_its_5_percent_floor(tmp_path: Path) -> None:
    session, _ = _make_session(
        tmp_path,
        architecture="vision2_language2_moe4_10cm_warmstart5_stage_b_v1",
    )
    data = load_offline_quick_data(session)
    tail = load_offline_tail(data, torch.device("cpu"))
    assert tail.minimum_optical_fusion == pytest.approx(0.05)


def test_strict_loader_rejects_wrong_ccd_shape(tmp_path: Path) -> None:
    session, rows = _make_session(tmp_path)
    bad = session / "04_language_global" / "ccd_captured" / f"{rows[0]['key']}.png"
    Image.fromarray(np.zeros((477, 478), dtype=np.uint8), mode="L").save(bad)
    with pytest.raises(RuntimeError, match="must be 478x478"):
        load_offline_quick_data(session)


def test_strict_loader_rejects_unexpected_ccd_key(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    extra = session / "04_language_global" / "ccd_captured" / "unexpected.png"
    Image.fromarray(np.zeros((478, 478), dtype=np.uint8), mode="L").save(extra)
    with pytest.raises(RuntimeError, match="key set does not match"):
        load_offline_quick_data(session)


def test_strict_loader_rejects_manifest_hash_drift(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    manifest = session / "manifest.csv"
    manifest.write_text(manifest.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest SHA-256 mismatch"):
        load_offline_quick_data(session)


def test_quick210_profile_rejects_only_total_count_style_validation(
    tmp_path: Path,
) -> None:
    session, _ = _make_session(tmp_path)
    contract_path = (
        session / "04_language_global" / "offline_downstream" / "contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["profile"] = "quick210"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="quick210 must contain"):
        load_offline_quick_data(session)


def test_cpu_offline_finetune_smoke_writes_tail_only_result(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    output = tmp_path / "result"
    report = finetune_offline_quick(
        session_dir=session,
        output_dir=output,
        device_name="cpu",
        epochs=1,
    )
    assert 0.0 <= report["top1_retrieval_accuracy"] <= 1.0
    assert report["best_epoch"] == 1
    assert (output / "best_offline_tail_state.pt").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "train_log.csv").is_file()
    assert (output / "ccd_inventory.json").is_file()
    state = torch.load(
        output / "best_offline_tail_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(state) == set(LanguageGlobalOfflineTail(**_construction()).state_dict())


def test_offline_module_import_does_not_load_transformers_or_qwen_modeling() -> None:
    repository = Path(__file__).resolve().parents[3]
    module = (
        "experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_"
        "10cm_robust.offline_quick_finetune"
    )
    script = (
        f"import {module}; import sys; "
        "assert 'transformers' not in sys.modules; "
        "assert not any(name.endswith('.modeling') and 'qwen3_vl_embedding' in name "
        "for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    help_result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "without loading Qwen" in help_result.stdout


def test_contract_hashes_are_standard_sha256(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    contract_path = (
        session / "04_language_global" / "offline_downstream" / "contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    cache = contract_path.parent / contract["cache_file"]
    assert contract["cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
