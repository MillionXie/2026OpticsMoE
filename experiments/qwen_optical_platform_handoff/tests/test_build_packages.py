from __future__ import annotations

from pathlib import Path

from experiments.qwen_optical_platform_handoff.build_packages import (
    HARDWARE_SEEDS,
    SIMULATION_SEEDS,
    build,
    dependency_closure,
    repo_root,
)


def test_simulation_dependency_closure_contains_core_retrieval() -> None:
    packages = dependency_closure(repo_root(), SIMULATION_SEEDS)
    assert "qwen3_vl_embedding_2b_grocery10_optical_retrieval" in packages
    assert "qwen_optical_platform_handoff" in packages


def test_hardware_dependency_closure_contains_four_layer_and_mnist() -> None:
    packages = dependency_closure(repo_root(), HARDWARE_SEEDS)
    assert (
        "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust"
        in packages
    )
    assert "d2nn_mnist4_single_layer_17um_10cm_v2" in packages


def test_repo_root_is_real() -> None:
    root = repo_root()
    assert (root / "experiments" / "__init__.py").is_file()
    assert Path(__file__).resolve().is_relative_to(root)


def test_default_build_never_embeds_qwen_weights(tmp_path: Path) -> None:
    report = build(tmp_path)
    assert report["simulation"]["offline_model_included"] is False
    assert report["hardware"]["offline_model_included"] is False
