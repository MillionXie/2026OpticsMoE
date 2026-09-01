"""Regression tests for the FixedFeedbackSFT physical-layout migration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import experiments

from FixedFeedbackSFT.paths import PROJECT_NAMES, PROJECTS_ROOT, REPOSITORY_ROOT


def test_projects_are_physically_collected_under_fixed_feedback_sft() -> None:
    assert not (PROJECTS_ROOT / "__init__.py").exists()
    for name in PROJECT_NAMES:
        assert (PROJECTS_ROOT / name / "__init__.py").is_file()
        assert not (REPOSITORY_ROOT / "experiments" / name).exists()


def test_legacy_python_module_names_resolve_to_the_new_physical_layout() -> None:
    assert PROJECTS_ROOT.resolve() in {Path(entry).resolve() for entry in experiments.__path__}
    for name in PROJECT_NAMES:
        spec = importlib.util.find_spec(f"experiments.{name}")
        assert spec is not None
        assert spec.origin is not None
        assert Path(spec.origin).resolve() == (PROJECTS_ROOT / name / "__init__.py").resolve()


def test_active_project_files_do_not_reference_removed_slash_paths() -> None:
    stale_prefixes = tuple(f"experiments/{name}" for name in PROJECT_NAMES)
    text_suffixes = {".py", ".ps1", ".sh", ".yaml", ".yml"}
    legacy_fallback = (
        PROJECTS_ROOT
        / "qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa"
        / "commands"
        / "p12_mechanism_audit.sh"
    )
    allowed_legacy_reference = (
        "experiments/"
        "qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/"
        "configs/base_50e.yaml"
    )
    for path in PROJECTS_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        stale_references = [prefix for prefix in stale_prefixes if prefix in text]
        if path == legacy_fallback:
            assert stale_references == [
                "experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa"
            ]
            assert text.count(allowed_legacy_reference) == 1
            continue
        assert not stale_references, path
