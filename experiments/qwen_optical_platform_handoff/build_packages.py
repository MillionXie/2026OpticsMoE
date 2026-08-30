"""Build portable simulation-code and hardware-code handoff archives.

The builder copies a dependency closure of experiment packages instead of
assuming that one long experiment directory is self-contained.  Runtime data,
caches, checkpoints and measured calibration values are excluded by default.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


SIMULATION_SEEDS = (
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff",
    "qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16",
    "qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224",
    "qwen3_vl_embedding_2b_salicon_vision_optical_saliency",
    "qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation",
    "qwen3_vl_embedding_2b_lsp_pose_optical_moe16",
    "vision2_hybrid_dense",
)
HARDWARE_SEEDS = (
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff",
    "d2nn_mnist4_single_layer_17um_10cm_v2",
)
HANDOFF_PACKAGE = "qwen_optical_platform_handoff"
SOURCE_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".csv",
    ".m",
}
GLOBAL_EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "runs",
    "cache",
    "caches",
    "data",
    "datasets",
    "hardware_sessions",
    "lab_exports",
    "lab_bundles",
    "validation_bundles",
    "results",
    "work",
    "generated",
    "artifacts",
}
LAB_EXCLUDED_PARTS = GLOBAL_EXCLUDED_PARTS | {
    "calib",
    "shape_agreement",
    "four",
    "four_accuracy_first",
    "four_accuracy_first_full",
    "four_balanced",
    "four_balanced_full",
    "last",
    "model",
}
CURRENT_VENDOR_DIRS = (
    "amplitude_meadowlark",
    "camera_tucam_mosaic",
    "phase_meadowlark",
)
HARDWARE_CORE_DIRS = (
    "workflows",
    "drivers",
    "demos",
    "tools",
    "tests",
    "configs",
    "generators",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _experiment_imports(path: Path) -> set[str]:
    packages: set[str] = set()
    for source in path.rglob("*.py"):
        if any(part in GLOBAL_EXCLUDED_PARTS for part in source.parts):
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            for name in names:
                if name.startswith("experiments."):
                    parts = name.split(".")
                    if len(parts) >= 2:
                        packages.add(parts[1])
    return packages


def dependency_closure(root: Path, seeds: Iterable[str]) -> list[str]:
    experiments = root / "experiments"
    selected = set(seeds) | {HANDOFF_PACKAGE}
    pending = list(selected)
    while pending:
        name = pending.pop()
        package = experiments / name
        if not package.is_dir():
            raise FileNotFoundError(f"Required experiment package is missing: {package}")
        for dependency in _experiment_imports(package):
            if (experiments / dependency).is_dir() and dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return sorted(selected)


def _is_source_file(path: Path, base: Path, *, excluded: set[str]) -> bool:
    relative = path.relative_to(base)
    return (
        path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and not any(part in excluded for part in relative.parts)
    )


def _copy_file(source: Path, destination: Path) -> None:
    _io_path(destination.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(_io_path(source), _io_path(destination))


def _io_path(path: Path) -> Path:
    """Use the Win32 extended path form for long experiment names."""
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _io_path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_source_tree(source: Path, destination: Path, *, excluded: set[str]) -> int:
    count = 0
    for path in source.rglob("*"):
        if _is_source_file(path, source, excluded=excluded):
            _copy_file(path, destination / path.relative_to(source))
            count += 1
    return count


def _copy_dependency_packages(
    root: Path, stage: Path, packages: Iterable[str], *, hardware: bool
) -> int:
    count = 0
    experiments = root / "experiments"
    _copy_file(experiments / "__init__.py", stage / "experiments" / "__init__.py")
    count += 1
    for name in packages:
        if name in {"hardware_sdk", "lab_qwen"}:
            continue
        excluded = set(GLOBAL_EXCLUDED_PARTS)
        if name == "d2nn_mnist4_single_layer_17um_10cm_v2" and hardware:
            excluded.discard("mask_candidates")
        count += _copy_source_tree(
            experiments / name,
            stage / "experiments" / name,
            excluded=excluded,
        )
        if name == "d2nn_mnist4_single_layer_17um_10cm_v2" and hardware:
            masks = experiments / name / "mask_candidates"
            if masks.is_dir():
                for path in masks.rglob("*.bmp"):
                    _copy_file(path, stage / "experiments" / name / path.relative_to(experiments / name))
                    count += 1
    return count


def _copy_hardware_sdk(root: Path, stage: Path, *, include_vendor: bool) -> int:
    source = root / "experiments" / "hardware_sdk"
    destination = stage / "experiments" / "hardware_sdk"
    count = 0
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
            _copy_file(path, destination / path.name)
            count += 1
    for name in HARDWARE_CORE_DIRS:
        directory = source / name
        if directory.is_dir():
            count += _copy_source_tree(directory, destination / name, excluded=GLOBAL_EXCLUDED_PARTS)
    if include_vendor:
        vendor = source / "vendor_sdk"
        vendor_destination = destination / "vendor_sdk"
        if (vendor / "README.md").is_file():
            _copy_file(vendor / "README.md", vendor_destination / "README.md")
            count += 1
        for name in CURRENT_VENDOR_DIRS:
            directory = vendor / name
            if not directory.is_dir():
                raise FileNotFoundError(f"Current vendor SDK is missing: {directory}")
            for path in directory.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    _copy_file(path, vendor_destination / name / path.relative_to(directory))
                    count += 1
    return count


def _copy_lab_sources(root: Path, stage: Path) -> int:
    source = root / "experiments" / "lab_qwen"
    destination = stage / "experiments" / "lab_qwen"
    count = _copy_source_tree(source, destination, excluded=LAB_EXCLUDED_PARTS)
    # A measured LAB_CONFIG is platform-specific and must not silently travel.
    measured = destination / "LAB_CONFIG.yaml"
    measured.unlink(missing_ok=True)
    template = (
        root
        / "experiments"
        / HANDOFF_PACKAGE
        / "templates"
        / "LAB_CONFIG.new_platform.yaml"
    )
    _copy_file(template, destination / "LAB_CONFIG.template.yaml")
    return count


def _copy_offline_model(model_dir: Path, stage: Path) -> int:
    required = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Offline model is incomplete: {model_dir}; missing={missing}")
    destination = stage / "models" / "Qwen3-VL-Embedding-2B"
    count = 0
    for path in model_dir.rglob("*"):
        if path.is_file():
            _copy_file(path, destination / path.relative_to(model_dir))
            count += 1
    return count


def _write_root_files(root: Path, stage: Path, *, kind: str, model_included: bool) -> None:
    handoff = root / "experiments" / HANDOFF_PACKAGE
    guide = "SIMULATION_PROJECT.md" if kind == "simulation" else "HARDWARE_PROJECT.md"
    readme = f"# Qwen Optical {kind.title()} Project\n\n"
    readme += f"Start with `experiments/qwen_optical_platform_handoff/{guide}`.\n\n"
    readme += "This archive intentionally excludes datasets, run outputs, CCD frames and checkpoints.\n"
    readme += (
        "The complete offline Qwen model is included.\n"
        if model_included
        else "The Qwen model is not duplicated; see MODEL_SETUP.md.\n"
    )
    (stage / "README_FIRST.md").write_text(readme, encoding="utf-8")
    (stage / "MODEL_SETUP.md").write_text(
        "# Qwen model setup\n\n"
        "For online servers, use an explicit Hugging Face revision and cache directory.\n"
        "For offline laboratory use, copy the complete snapshot to "
        "`models/Qwen3-VL-Embedding-2B` and require local-files-only mode.\n"
        "Required files include config.json, model.safetensors, "
        "preprocessor_config.json and tokenizer.json. Never silently fall back "
        "to the network during a formal experiment.\n",
        encoding="utf-8",
    )
    requirements = handoff / f"requirements-{kind}.txt"
    _copy_file(requirements, stage / "requirements.txt")


def _manifest(stage: Path, kind: str, packages: list[str]) -> dict[str, object]:
    rows = []
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        rows.append(
            {
                "path": path.relative_to(stage).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "dependency_packages": packages,
        "files": rows,
        "total_bytes": sum(int(row["bytes"]) for row in rows),
    }
    (stage / "PACKAGE_CONTENTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _zip_tree(stage: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as handle:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            handle.write(_io_path(path), path.relative_to(stage).as_posix())


def audit_archive(archive: Path, *, kind: str) -> dict[str, object]:
    required = {
        "README_FIRST.md",
        "MODEL_SETUP.md",
        "PACKAGE_CONTENTS.json",
        "requirements.txt",
        "experiments/__init__.py",
        "experiments/qwen_optical_platform_handoff/AI_MODIFICATION_RULES.md",
        "experiments/qwen_optical_platform_handoff/NEW_TASK_WORKFLOW.md",
        "experiments/qwen_optical_platform_handoff/COMMANDS.md",
    }
    if kind == "hardware":
        required |= {
            "experiments/lab_qwen/LAB_CONFIG.template.yaml",
            "experiments/hardware_sdk/workflows/acquire_folder.py",
            "experiments/hardware_sdk/workflows/amplitude_lut_calibration.py",
            "experiments/d2nn_mnist4_single_layer_17um_10cm_v2/lab_pipeline.py",
        }
    else:
        required |= {
            "experiments/hardware_sdk/workflows/reconstruct_slm.py",
            "experiments/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16/training.py",
            "experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/training.py",
        }
    forbidden_fragments = (
        "/ccd_captured/",
        "/hardware_sessions/",
        "/runs/",
        "/generated/",
        "/__pycache__/",
        "/.pytest_cache/",
    )
    with zipfile.ZipFile(archive) as handle:
        corrupt = handle.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt archive member: {corrupt}")
        names = set(handle.namelist())
        missing = sorted(required.difference(names))
        if missing:
            raise RuntimeError(f"Archive is missing required files: {missing}")
        forbidden = sorted(
            name
            for name in names
            if any(fragment in f"/{name}" for fragment in forbidden_fragments)
            or name.endswith("/LAB_CONFIG.yaml")
            or name.endswith(".pt")
            or name.endswith(".pth")
        )
        if forbidden:
            raise RuntimeError(f"Archive contains runtime/platform state: {forbidden[:20]}")
        if kind == "simulation" and any("/vendor_sdk/" in f"/{name}" for name in names):
            raise RuntimeError("Simulation archive must not contain vendor SDK binaries")
        if kind == "hardware":
            vendor_prefix = "experiments/hardware_sdk/vendor_sdk/"
            invalid_vendor = []
            for name in names:
                if not name.startswith(vendor_prefix):
                    continue
                relative = name[len(vendor_prefix) :]
                first = relative.split("/", 1)[0]
                if first and first not in {*CURRENT_VENDOR_DIRS, "README.md"}:
                    invalid_vendor.append(name)
            if invalid_vendor:
                raise RuntimeError(f"Archive contains legacy vendor SDK: {invalid_vendor[:20]}")
        compiled = 0
        for name in sorted(names):
            if not name.endswith(".py") or "/vendor_sdk/" in f"/{name}":
                continue
            compile(handle.read(name), name, "exec")
            compiled += 1
    return {
        "zip_members": len(names),
        "python_files_compiled": compiled,
        "corrupt_member": None,
        "platform_specific_lab_config_excluded": True,
        "runtime_outputs_excluded": True,
    }


def build(
    output_dir: str | Path,
    *,
    hardware_offline_model_dir: str | Path | None = None,
) -> dict[str, object]:
    root = repo_root()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    hardware_model = (
        None
        if hardware_offline_model_dir is None
        else Path(hardware_offline_model_dir).expanduser().resolve()
    )
    reports: dict[str, object] = {}
    for kind, seeds in (("simulation", SIMULATION_SEEDS), ("hardware", HARDWARE_SEEDS)):
        packages = dependency_closure(root, seeds)
        temporary_root = root / ".codex_transfer"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"pkg_{kind[0]}_", dir=temporary_root
        ) as temporary:
            stage = Path(temporary)
            copied = _copy_dependency_packages(
                root, stage, packages, hardware=(kind == "hardware")
            )
            if kind == "hardware":
                copied += _copy_hardware_sdk(root, stage, include_vendor=True)
                copied += _copy_lab_sources(root, stage)
            elif "hardware_sdk" in packages:
                copied += _copy_hardware_sdk(root, stage, include_vendor=False)
            model_included = kind == "hardware" and hardware_model is not None
            if model_included:
                copied += _copy_offline_model(hardware_model, stage)
            _write_root_files(root, stage, kind=kind, model_included=model_included)
            manifest = _manifest(stage, kind, packages)
            archive = destination / f"qwen_optical_{kind}_project.zip"
            _zip_tree(stage, archive)
            audit = audit_archive(archive, kind=kind)
            reports[kind] = {
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": _sha256_file(archive),
                "copied_files_before_manifest": copied,
                "manifest_files": len(manifest["files"]),
                "dependency_packages": packages,
                "offline_model_included": model_included,
                "audit": audit,
            }
    report_path = destination / "build_report.json"
    report_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "README_先看这里.md").write_text(
        "# Qwen 光学平台双工程交接包\n\n"
        "- `qwen_optical_simulation_project.zip`：服务器仿真、训练、评估和硬件 payload 导出。\n"
        "- `qwen_optical_hardware_project.zip`：实验室标定、采集、一致性评估和 measured-CCD 微调。\n\n"
        "先校验 `build_report.json` 中的 SHA256，再分别解压到两个不同目录。"
        "不要把两个 ZIP 互相覆盖解压。解压后从 `README_FIRST.md` 开始。\n\n"
        "两个代码包都不携带任务数据/checkpoint/CCD；硬件包是否携带离线 Qwen 权重，"
        "以 build_report.json 的 offline_model_included 为准。新任务还必须从仿真工程"
        "导出 task payload，详见包内 PROJECT_BOUNDARIES.md。\n",
        encoding="utf-8",
    )
    return {"output_dir": str(destination), "report": str(report_path), **reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="handoff_exports")
    parser.add_argument(
        "--hardware-offline-model-dir",
        default=None,
        help=(
            "Optional complete Qwen snapshot. It is embedded only in the hardware "
            "archive so the two archives never duplicate the 4+ GB weights."
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output_dir,
                hardware_offline_model_dir=args.hardware_offline_model_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
