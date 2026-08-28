"""Model-free geometric SLM benchmark for simulation-to-hardware agreement.

The generator creates deterministic 478x478 amplitude shapes, geometric phase
masks, native Meadowlark/phase-SLM BMPs, and paired ideal/transport-quantized
angular-spectrum CCD references.  Acquisition uses the normal audited
``formal_hardware.yaml`` homography.  Evaluation never registers, flips,
background-subtracts, or per-frame min-max normalizes a measured CCD frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_phase,
    reconstruct_directory,
    save_active_png,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate import (
    _load_ccd,
    orientation_diagnostics,
    pair_metrics,
)


SCHEMA_VERSION = 1
BENCHMARK_TYPE = "geometric_amplitude_phase_sim_to_real"
ACTIVE_SIZE = 478
CANVAS_SIZE = 518
ACTIVE_MARGIN = (CANVAS_SIZE - ACTIVE_SIZE) // 2
WAVELENGTH_M = 532.0e-9
PIXEL_PITCH_M = 17.0e-6
DISTANCE_M = 0.10
THETA_MAX_DEG = 0.65
AMPLITUDE_SLM_SIZE_WH = (1024, 1024)
AMPLITUDE_CENTER_XY = (512.0, 512.0)
PHASE_SLM_SIZE_WH = (1920, 1200)
PHASE_CENTER_XY = (980.0, 590.0)
PHASE_SLM_PIXEL_PITCH_UM = 8.0
LOGICAL_PIXEL_PITCH_UM = 17.0
PHASE_FLIP_VERTICAL = True
PHASE_FLIP_HORIZONTAL = False
METRIC_DOMAINS = ("linear", "network_input")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    return rows


def _mask_from_pil(draw_fn: Any) -> np.ndarray:
    image = Image.new("L", (ACTIVE_SIZE, ACTIVE_SIZE), 0)
    draw_fn(ImageDraw.Draw(image))
    return np.array(image, dtype=np.uint8, copy=True)


def amplitude_shapes() -> dict[str, np.ndarray]:
    """Six deterministic, asymmetric, low/mid-frequency amplitude probes."""

    s = ACTIVE_SIZE
    y, x = np.mgrid[:s, :s]
    cx = cy = (s - 1) / 2.0
    disk = (((x - (cx - 58)) ** 2 + (y - (cy - 34)) ** 2) <= 72**2)
    ring_radius = np.sqrt((x - (cx + 42)) ** 2 + (y - (cy - 40)) ** 2)
    ring = (ring_radius >= 58) & (ring_radius <= 91)

    square = np.zeros((s, s), dtype=np.uint8)
    square[116:270, 246:400] = 255

    cross = np.zeros((s, s), dtype=np.uint8)
    cross[92:370, 205:251] = 255
    cross[214:260, 118:382] = 255
    cross[92:150, 251:298] = 255

    letter_l = np.zeros((s, s), dtype=np.uint8)
    letter_l[86:376, 116:174] = 255
    letter_l[318:376, 116:350] = 255
    letter_l[86:132, 174:244] = 160

    triangle = _mask_from_pil(
        lambda draw: draw.polygon([(94, 362), (236, 82), (392, 340)], fill=255)
    )
    triangle[252:308, 214:270] = 0

    result = {
        "input_00_offset_disk": disk.astype(np.uint8) * 255,
        "input_01_offset_square": square,
        "input_02_offset_ring": ring.astype(np.uint8) * 255,
        "input_03_asymmetric_cross": cross,
        "input_04_letter_L": letter_l,
        "input_05_notched_triangle": triangle,
    }
    digests = {_sha256_bytes(value.tobytes()) for value in result.values()}
    if len(digests) != len(result) or any(value.max() == 0 for value in result.values()):
        raise RuntimeError("Amplitude shape suite is empty or duplicated")
    return result


def phase_shapes_rad() -> dict[str, np.ndarray]:
    """Six geometric phase masks in canonical model coordinates."""

    s = ACTIVE_SIZE
    y, x = np.mgrid[:s, :s]
    cx = cy = (s - 1) / 2.0

    def phase_from(mask: np.ndarray, depth_turns: float) -> np.ndarray:
        return mask.astype(np.float32) * np.float32(2.0 * np.pi * depth_turns)

    circle = (x - (cx - 38)) ** 2 + (y - (cy + 27)) ** 2 <= 112**2
    square = np.zeros((s, s), dtype=bool)
    square[104:326, 174:396] = True
    radius = np.sqrt((x - (cx + 28)) ** 2 + (y - (cy - 36)) ** 2)
    ring = (radius >= 74) & (radius <= 136)
    cross = np.zeros((s, s), dtype=bool)
    cross[88:390, 216:258] = True
    cross[200:248, 86:365] = True
    letter_l = np.zeros((s, s), dtype=bool)
    letter_l[72:388, 104:162] = True
    letter_l[330:388, 104:354] = True

    result = {
        "phase_00_zero": np.zeros((s, s), dtype=np.float32),
        "phase_01_circle_0p75turn": phase_from(circle, 0.75),
        "phase_02_square_0p625turn": phase_from(square, 0.625),
        "phase_03_ring_0p5turn": phase_from(ring, 0.5),
        "phase_04_cross_0p375turn": phase_from(cross, 0.375),
        "phase_05_letter_L_0p25turn": phase_from(letter_l, 0.25),
    }
    if any(value.shape != (s, s) or not np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Invalid phase shape suite")
    return result


def angular_spectrum_intensity(
    amplitude_uint8: np.ndarray,
    phase_rad: np.ndarray,
    *,
    k_space_enabled: bool = True,
) -> np.ndarray:
    """Match the current 518-grid, 17 um, 532 nm, 10 cm propagator."""

    amplitude = np.asarray(amplitude_uint8, dtype=np.float64)
    phase = np.asarray(phase_rad, dtype=np.float64)
    if amplitude.shape != (ACTIVE_SIZE, ACTIVE_SIZE) or phase.shape != amplitude.shape:
        raise ValueError("Shape agreement inputs must both be 478x478")
    if not np.isfinite(amplitude).all() or not np.isfinite(phase).all():
        raise ValueError("Shape agreement inputs must be finite")
    field = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.complex128)
    active = (amplitude / 255.0) * np.exp(1j * phase)
    m = ACTIVE_MARGIN
    field[m : m + ACTIVE_SIZE, m : m + ACTIVE_SIZE] = active

    frequency = np.fft.fftfreq(CANVAS_SIZE, d=PIXEL_PITCH_M)
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    two_pi = 2.0 * np.pi
    argument = two_pi**2 * (1.0 / WAVELENGTH_M**2 - fx**2 - fy**2)
    propagating = argument >= 0.0
    if k_space_enabled:
        radial_wave_number = two_pi * np.sqrt(fx**2 + fy**2)
        cutoff = (two_pi / WAVELENGTH_M) * math.sin(math.radians(THETA_MAX_DEG))
        propagating &= radial_wave_number <= cutoff
    transfer = np.zeros_like(field)
    transfer[propagating] = np.exp(
        1j * DISTANCE_M * np.sqrt(np.maximum(argument[propagating], 0.0))
    )
    detector = np.fft.ifft2(np.fft.fft2(field) * transfer)
    intensity = np.abs(detector) ** 2
    cropped = intensity[m : m + ACTIVE_SIZE, m : m + ACTIVE_SIZE]
    return np.asarray(cropped, dtype=np.float32)


def _save_reference(path: Path, intensity: np.ndarray) -> str:
    value = np.asarray(intensity, dtype=np.float32)
    if value.shape != (ACTIVE_SIZE, ACTIVE_SIZE) or np.any(value < 0) or not np.isfinite(value).all():
        raise ValueError("Theoretical CCD must be finite nonnegative 478x478")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, intensity=value)
    return _sha256_file(path)


def _safe_new_directory(target: Path, overwrite: bool) -> None:
    if not target.exists():
        target.mkdir(parents=True)
        return
    marker = target / "shape_agreement_manifest.json"
    if not overwrite:
        raise FileExistsError(f"Shape session already exists: {target}")
    if not marker.is_file():
        raise RuntimeError(f"Refusing to replace unowned directory: {target}")
    shutil.rmtree(target)
    target.mkdir(parents=True)


def _display_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix().replace("/", "\\")
    except ValueError:
        return str(path.resolve())


def _write_run_commands(root: Path, stages: list[dict[str, Any]]) -> None:
    root_display = _display_relative(root)
    lines = [
        "# Shape agreement acquisition order",
        "",
        "Run every command from the standalone repository root. CCD output must use",
        "the generated formal homography configuration. Each stage has one manual phase BMP.",
        "",
    ]
    for index, item in enumerate(stages, 1):
        stage = item["stage"]
        lines.extend(
            [
                f"## {index}. {stage}",
                "",
                "Manually load the only BMP under:",
                "",
                f"`{root_display}\\{stage}\\phase_to_play`",
                "",
                "```powershell",
                "python -m experiments.hardware_sdk.workflows.acquire_folder `",
                "  --config experiments\\lab_qwen\\generated\\formal_hardware.yaml `",
                f"  --stage-dir {root_display}\\{stage} `",
                "  --clear-output",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Evaluate and plot",
            "",
            "```powershell",
            "python -m experiments.lab_qwen.shape_agreement evaluate `",
            f"  --session-dir {root_display}",
            "```",
            "",
        ]
    )
    (root / "RUN_COMMANDS.md").write_text("\n".join(lines), encoding="utf-8")


def _write_protocol(root: Path) -> None:
    (root / "BENCHMARK_PROTOCOL.md").write_text(
        """# 形状输入 × 形状相位 mask：仿真—实测一致性基准

本会话固定使用 532 nm、17 µm 逻辑采样、518×518 传播画布、中心 478×478
有效光场、10 cm 角谱传播和 0.65° k 空间截止。6 个非对称振幅形状分别与
6 个几何相位 mask 组合，共采集 36 帧。非对称图形用于暴露左右/上下翻转错误，
而不是在评估时自动纠正方向。

## 正式主结果

- 主参考：`transport_quantized`，即相位量化为 8-bit 后的仿真结果。
- 主域：`linear`，即 CCD 原始非负强度域。
- 主方向：固定四点 homography 后的 `canonical_model_xy`；不逐帧配准、不搜索翻转。
- 不做背景扣除，不做逐帧 min-max 归一化。
- 全部 36 帧共用一个全局能量增益，仅用于报告绝对能量比例；PCC、SSIM、
  shape-NRMSE 和余弦相似度仍反映空间分布一致性。

## 指标

- `pcc_full`：整幅图 Pearson 相关系数。
- `pcc_signal`：理论光能 99% 信号区域内的 PCC。
- `ssim`：按单帧均值作无量纲化并固定截断后的结构相似度。
- `shape_nrmse`：双方各自按总能量归一化后的 NRMSE，越低越好。
- `cosine_similarity`：非负强度向量的余弦相似度。
- `centroid_distance_px`：实测与仿真光强质心距离，单位为 478 网格像素。
- `outside_energy_fraction`：实测能量落在理论 99% 信号区外的比例。
- `energy_ratio_raw/calibrated`：原始/全局增益校准后的能量比例。
- `saturation_fraction`：CCD 饱和像素比例。

`best_orientation_diagnostic` 只用于排查配置错误。它不会改变任何主指标；若大量
样本的最佳诊断方向不是 `identity`，应回去检查四个逻辑角点标签，不应从多个
翻转结果中挑最高值作为正式结果。
""",
        encoding="utf-8",
    )


def generate_session(
    output_dir: str | Path,
    *,
    phase_center_xy: tuple[float, float] = PHASE_CENTER_XY,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    _safe_new_directory(root, overwrite)
    amplitudes = amplitude_shapes()
    phases = phase_shapes_rad()
    stage_records: list[dict[str, Any]] = []

    for phase_name, ideal_phase in phases.items():
        stage = root / phase_name
        compact_amplitude = stage / "compact_amplitude"
        compact_phase = stage / "compact_phase"
        compact_amplitude.mkdir(parents=True)
        compact_phase.mkdir(parents=True)
        (stage / "ccd_captured").mkdir()
        (stage / "acquisition_logs").mkdir()

        encoded_logical = encode_active_phase(ideal_phase)
        decoded_logical = encoded_logical.astype(np.float32) * (2.0 * np.pi / 256.0)
        encoded_export = encoded_logical
        if PHASE_FLIP_VERTICAL:
            encoded_export = np.flip(encoded_export, axis=0)
        if PHASE_FLIP_HORIZONTAL:
            encoded_export = np.flip(encoded_export, axis=1)
        phase_compact_path = compact_phase / f"{phase_name}.png"
        save_active_png(np.ascontiguousarray(encoded_export), phase_compact_path)
        reconstruct_directory(
            compact_phase,
            stage / "phase_to_play",
            slm_size_wh=PHASE_SLM_SIZE_WH,
            scale_factor=None,
            center_xy=phase_center_xy,
            logical_pixel_pitch_um=LOGICAL_PIXEL_PITCH_UM,
            slm_pixel_pitch_um=PHASE_SLM_PIXEL_PITCH_UM,
        )

        probe_rows: list[dict[str, Any]] = []
        for order, (input_name, amplitude) in enumerate(amplitudes.items()):
            compact_path = compact_amplitude / f"{input_name}.png"
            save_active_png(amplitude, compact_path)
            ideal_path = stage / "theoretical_ccd" / "ideal_continuous" / f"{input_name}.npz"
            transport_path = stage / "theoretical_ccd" / "transport_quantized" / f"{input_name}.npz"
            ideal_sha = _save_reference(
                ideal_path, angular_spectrum_intensity(amplitude, ideal_phase)
            )
            transport_sha = _save_reference(
                transport_path, angular_spectrum_intensity(amplitude, decoded_logical)
            )
            probe_rows.append(
                {
                    "order": order,
                    "phase_name": phase_name,
                    "input_name": input_name,
                    "amplitude_file": f"{input_name}.bmp",
                    "compact_amplitude_file": f"compact_amplitude/{input_name}.png",
                    "compact_amplitude_sha256": _sha256_file(compact_path),
                    "ideal_reference_file": ideal_path.relative_to(stage).as_posix(),
                    "ideal_reference_sha256": ideal_sha,
                    "transport_reference_file": transport_path.relative_to(stage).as_posix(),
                    "transport_reference_sha256": transport_sha,
                }
            )
        reconstruct_directory(
            compact_amplitude,
            stage / "amplitude_to_play",
            slm_size_wh=AMPLITUDE_SLM_SIZE_WH,
            scale_factor=None,
            center_xy=AMPLITUDE_CENTER_XY,
            logical_pixel_pitch_um=LOGICAL_PIXEL_PITCH_UM,
            slm_pixel_pitch_um=LOGICAL_PIXEL_PITCH_UM,
        )
        probe_manifest = stage / "probe_manifest.csv"
        _write_csv(probe_manifest, probe_rows)
        phase_manifest = stage / "phase_to_play" / "reconstruction_manifest.csv"
        phase_reconstruction_rows = _read_csv(phase_manifest)
        if len(phase_reconstruction_rows) != 1:
            raise RuntimeError(f"Expected one reconstructed phase BMP: {phase_manifest}")
        phase_bmp_name = phase_reconstruction_rows[0]["output_bmp"]
        phase_bmp_sha = phase_reconstruction_rows[0]["output_sha256"]
        contract = {
            "schema_version": SCHEMA_VERSION,
            "type": BENCHMARK_TYPE,
            "stage": phase_name,
            "probe_count": len(probe_rows),
            "active_shape_hw": [ACTIVE_SIZE, ACTIVE_SIZE],
            "canvas_shape_hw": [CANVAS_SIZE, CANVAS_SIZE],
            "wavelength_nm": WAVELENGTH_M * 1.0e9,
            "logical_pixel_pitch_um": LOGICAL_PIXEL_PITCH_UM,
            "propagation_distance_m": DISTANCE_M,
            "k_space": {"enabled": True, "theta_max_deg": THETA_MAX_DEG},
            "phase_export": {
                "phase_slm_size_wh": list(PHASE_SLM_SIZE_WH),
                "phase_slm_pixel_pitch_um": PHASE_SLM_PIXEL_PITCH_UM,
                "phase_center_xy": list(phase_center_xy),
                "flip_vertical_before_rasterization": PHASE_FLIP_VERTICAL,
                "flip_horizontal_before_rasterization": PHASE_FLIP_HORIZONTAL,
                "quantization": "floor(mod(phase,2pi)/(2pi)*256)",
                "phase_bmp_file": f"phase_to_play/{phase_bmp_name}",
                "phase_bmp_sha256": phase_bmp_sha,
            },
            "phase_reconstruction_manifest": "phase_to_play/reconstruction_manifest.csv",
            "phase_reconstruction_manifest_sha256": _sha256_file(phase_manifest),
            "amplitude_reconstruction_manifest": "amplitude_to_play/reconstruction_manifest.csv",
            "amplitude_reconstruction_manifest_sha256": _sha256_file(
                stage / "amplitude_to_play" / "reconstruction_manifest.csv"
            ),
            "probe_manifest": probe_manifest.name,
            "probe_manifest_sha256": _sha256_file(probe_manifest),
            "ccd_contract": {
                "shape_hw": [ACTIVE_SIZE, ACTIVE_SIZE],
                "orientation": "canonical_model_xy",
                "homography_required": True,
                "downstream_flip_required": False,
                "background_subtraction": False,
                "per_frame_registration": False,
                "per_frame_minmax_normalization": False,
            },
        }
        contract_path = stage / "shape_contract.json"
        _write_json(contract_path, contract)
        stage_records.append(
            {
                "stage": phase_name,
                "directory": phase_name,
                "contract": f"{phase_name}/shape_contract.json",
                "contract_sha256": _sha256_file(contract_path),
            }
        )

    root_manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": BENCHMARK_TYPE,
        "phase_masks": len(phases),
        "amplitude_shapes_per_mask": len(amplitudes),
        "expected_captures": len(phases) * len(amplitudes),
        "stages": stage_records,
    }
    _write_json(root / "shape_agreement_manifest.json", root_manifest)
    _write_run_commands(root, stage_records)
    _write_protocol(root)
    return {"session_dir": str(root), **root_manifest}


def _manifest_bool(row: dict[str, str], field: str) -> bool:
    value = str(row.get(field, "")).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise RuntimeError(f"Capture manifest field {field!r} is not boolean: {value!r}")


def _load_reference(path: Path, expected_sha: str) -> np.ndarray:
    if _sha256_file(path) != expected_sha:
        raise RuntimeError(f"Theoretical reference hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"intensity"}:
            raise RuntimeError(f"Reference must contain only intensity: {path}")
        value = np.asarray(payload["intensity"], dtype=np.float32)
    if value.shape != (ACTIVE_SIZE, ACTIVE_SIZE) or np.any(value < 0) or not np.isfinite(value).all():
        raise RuntimeError(f"Invalid theoretical reference: {path}")
    return value


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.clip(np.asarray(left, dtype=np.float64), 0.0, None).reshape(-1)
    y = np.clip(np.asarray(right, dtype=np.float64), 0.0, None).reshape(-1)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator <= 1.0e-18 else float(np.dot(x, y) / denominator)


def _mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _median_or_none(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _plot_results(
    output: Path,
    primary_rows: list[dict[str, Any]],
    pair_images: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    phases = sorted({row["phase_name"] for row in primary_rows})
    inputs = sorted({row["input_name"] for row in primary_rows})
    lookup = {(row["phase_name"], row["input_name"]): row for row in primary_rows}
    generated: list[str] = []

    def compact(value: Any) -> str:
        if value is None or not math.isfinite(float(value)):
            return "--"
        return f"{float(value):.2f}"

    for metric, title, vmin, vmax, cmap in (
        ("pcc_full", "PCC (linear, transport-quantized)", -1.0, 1.0, "coolwarm"),
        ("ssim", "SSIM (linear, transport-quantized)", 0.0, 1.0, "viridis"),
        ("cosine_similarity", "Cosine similarity", 0.0, 1.0, "viridis"),
        ("shape_nrmse", "Shape NRMSE (lower is better)", 0.0, 2.0, "magma_r"),
    ):
        matrix = np.asarray(
            [[lookup[(phase, shape)].get(metric, np.nan) for shape in inputs] for phase in phases],
            dtype=np.float64,
        )
        fig, ax = plt.subplots(figsize=(8.6 / 2.54, 5.4 / 2.54), constrained_layout=True)
        image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(inputs)), [name.replace("input_", "") for name in inputs], rotation=35, ha="right")
        ax.set_yticks(range(len(phases)), [name.replace("phase_", "") for name in phases])
        ax.set_title(title)
        for y in range(len(phases)):
            for x in range(len(inputs)):
                value = matrix[y, x]
                ax.text(
                    x,
                    y,
                    "--" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=(
                        "white"
                        if np.isfinite(value) and value < (vmin + vmax) / 2
                        else "black"
                    ),
                )
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        stem = f"shape_{metric}_heatmap"
        for suffix, dpi in (("pdf", None), ("svg", None), ("png", 600)):
            path = figures / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            generated.append(str(path))
        plt.close(fig)

    amplitude_cache = amplitude_shapes()
    phase_cache = phase_shapes_rad()
    selected_inputs = inputs
    for phase in phases:
        fig, axes = plt.subplots(
            len(selected_inputs),
            5,
            figsize=(14.0 / 2.54, 6.0 / 2.54),
            constrained_layout=True,
        )
        for row_index, input_name in enumerate(selected_inputs):
            measured, theory, difference = pair_images[(phase, input_name)]
            amplitude = amplitude_cache[input_name]
            phase_map = np.mod(phase_cache[phase], 2.0 * np.pi) / (2.0 * np.pi)
            displays = (
                (amplitude, "Amplitude", "gray", 0.0, 255.0),
                (phase_map, "Phase / 2pi", "twilight", 0.0, 1.0),
                (np.log1p(theory / max(float(theory.mean()), 1.0e-12)), "Simulation", "magma", None, None),
                (np.log1p(measured / max(float(measured.mean()), 1.0e-12)), "Measured", "magma", None, None),
                (difference, "|shape diff|", "inferno", 0.0, None),
            )
            for column, (value, title, cmap, vmin, vmax) in enumerate(displays):
                ax = axes[row_index, column]
                ax.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_index == 0:
                    ax.set_title(title)
                if column == 0:
                    ax.set_ylabel(input_name.replace("input_", ""), fontsize=5.5)
            metric = lookup[(phase, input_name)]
            axes[row_index, 3].set_xlabel(
                f"PCC {compact(metric['pcc_full'])}  SSIM {compact(metric['ssim'])}\n"
                f"NRMSE {compact(metric['shape_nrmse'])}  cos {compact(metric['cosine_similarity'])}",
                fontsize=4.8,
            )
        fig.suptitle(phase, fontsize=7)
        stem = f"comparison_{phase}"
        for suffix, dpi in (("pdf", None), ("png", 600)):
            path = figures / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            generated.append(str(path))
        plt.close(fig)
    return generated


def evaluate_session(
    session_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    session = Path(session_dir).expanduser().resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else session / "shape_agreement_results"
    )
    root = _read_json(session / "shape_agreement_manifest.json")
    if root.get("type") != BENCHMARK_TYPE or root.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported shape agreement session")

    pairs: list[dict[str, Any]] = []
    ratios: list[float] = []
    detector_geometry_hashes: set[str] = set()
    capture_exposures: set[str] = set()
    for item in root["stages"]:
        stage = session / str(item["directory"])
        contract_path = session / str(item["contract"])
        if _sha256_file(contract_path) != item["contract_sha256"]:
            raise RuntimeError(f"Shape contract hash mismatch: {contract_path}")
        contract = _read_json(contract_path)
        probes = _read_csv(stage / str(contract["probe_manifest"]))
        captures = _read_csv(stage / "acquisition_logs" / "capture_manifest.csv")
        if len(captures) != len({row["amplitude_bmp"] for row in captures}):
            raise RuntimeError(f"Duplicate amplitude capture rows for {stage.name}")
        capture_by_amplitude = {row["amplitude_bmp"]: row for row in captures}
        reconstruction = _read_csv(stage / "amplitude_to_play" / "reconstruction_manifest.csv")
        reconstruct_by_output = {row["output_bmp"]: row for row in reconstruction}
        expected = {row["amplitude_file"] for row in probes}
        if set(capture_by_amplitude) != expected or set(reconstruct_by_output) != expected:
            raise RuntimeError(f"Capture/reconstruction set mismatch for {stage.name}")

        for probe in probes:
            amplitude_name = probe["amplitude_file"]
            capture = capture_by_amplitude[amplitude_name]
            if not _manifest_bool(capture, "orientation_canonicalized"):
                raise RuntimeError(f"Capture is not homography-canonicalized: {amplitude_name}")
            if capture.get("saved_frame_orientation") != "canonical_model_xy":
                raise RuntimeError(f"Capture orientation is not canonical: {amplitude_name}")
            if _manifest_bool(capture, "downstream_loader_flip_required"):
                raise RuntimeError(f"Capture requests a forbidden extra flip: {amplitude_name}")
            if _manifest_bool(capture, "background_subtraction") or _manifest_bool(
                capture, "per_frame_minmax_normalization"
            ):
                raise RuntimeError(f"Capture used forbidden intensity processing: {amplitude_name}")
            phase_export = contract["phase_export"]
            if capture.get("phase_mask_sha256") != phase_export["phase_bmp_sha256"]:
                raise RuntimeError(f"Captured with the wrong phase mask: {amplitude_name}")
            if not _manifest_bool(capture, "phase_manifest_verified"):
                raise RuntimeError(
                    f"Phase reconstruction manifest was not verified: {amplitude_name}"
                )
            detector_hash = str(
                capture.get("detector_geometry_payload_sha256", "")
            ).strip()
            if len(detector_hash) != 64:
                raise RuntimeError(
                    f"Capture has no valid detector homography hash: {amplitude_name}"
                )
            detector_geometry_hashes.add(detector_hash)
            capture_exposures.add(
                str(capture.get("camera_exposure_us", "")).strip()
            )
            amplitude_path = stage / "amplitude_to_play" / amplitude_name
            reconstruction_row = reconstruct_by_output[amplitude_name]
            amplitude_sha = _sha256_file(amplitude_path)
            if amplitude_sha != reconstruction_row["output_sha256"] or amplitude_sha != capture["amplitude_bmp_sha256"]:
                raise RuntimeError(f"Played amplitude hash mismatch: {amplitude_name}")
            ccd_path = stage / "ccd_captured" / capture["ccd_capture"]
            if _sha256_file(ccd_path) != capture["output_sha256"]:
                raise RuntimeError(f"CCD capture hash mismatch: {ccd_path}")
            measured, saturation_value = _load_ccd(ccd_path, (ACTIVE_SIZE, ACTIVE_SIZE))
            references = {
                "ideal_continuous": _load_reference(
                    stage / probe["ideal_reference_file"], probe["ideal_reference_sha256"]
                ),
                "transport_quantized": _load_reference(
                    stage / probe["transport_reference_file"], probe["transport_reference_sha256"]
                ),
            }
            transport = references["transport_quantized"]
            if float(transport.mean()) > 1.0e-12:
                ratios.append(float(measured.mean() / transport.mean()))
            pairs.append(
                {
                    "phase_name": stage.name,
                    "input_name": probe["input_name"],
                    "measured": measured,
                    "saturation_value": saturation_value,
                    "references": references,
                }
            )

    if len(detector_geometry_hashes) != 1:
        raise RuntimeError(
            "All shape captures must use the same detector homography; observed "
            f"{len(detector_geometry_hashes)} contracts"
        )
    if len(capture_exposures) != 1:
        raise RuntimeError(
            "All shape captures must use one fixed exposure; observed "
            f"{sorted(capture_exposures)}"
        )
    energy_gain = float(np.median(ratios)) if ratios else 1.0
    metric_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    pair_images: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pair in pairs:
        measured = pair["measured"]
        transport = pair["references"]["transport_quantized"]
        orientation = orientation_diagnostics(measured, transport, relative_clip=12.0)
        best_orientation = max(
            orientation,
            key=lambda row: -math.inf if row["pcc_full"] is None else float(row["pcc_full"]),
        )
        for reference_kind, theory in pair["references"].items():
            for domain in METRIC_DOMAINS:
                metrics = pair_metrics(
                    measured,
                    theory,
                    domain=domain,
                    energy_gain=energy_gain,
                    saturation_value=float(pair["saturation_value"]),
                    signal_energy_fraction=0.99,
                    relative_clip=12.0,
                    log_compression=1.0,
                )
                row = {
                    "phase_name": pair["phase_name"],
                    "input_name": pair["input_name"],
                    "reference_kind": reference_kind,
                    "domain": domain,
                    **metrics,
                    "cosine_similarity": _cosine_similarity(measured, theory),
                    "best_orientation_diagnostic": best_orientation["orientation"],
                    "best_orientation_pcc_diagnostic": best_orientation["pcc_full"],
                    "primary_orientation": "identity_canonical_model_xy",
                }
                metric_rows.append(row)
                if reference_kind == "transport_quantized" and domain == "linear":
                    primary_rows.append(row)
        measured_shape = measured / max(float(measured.sum()), 1.0e-12)
        theory_shape = transport / max(float(transport.sum()), 1.0e-12)
        pair_images[(pair["phase_name"], pair["input_name"])] = (
            measured,
            transport,
            np.abs(measured_shape - theory_shape),
        )

    _write_csv(output / "metrics_per_pair.csv", metric_rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(row["phase_name"], row["reference_kind"], row["domain"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    summary_metrics = (
        "pcc_full",
        "pcc_signal",
        "ssim",
        "shape_nrmse",
        "cosine_similarity",
        "centroid_distance_px",
        "energy_ratio_raw",
        "outside_energy_fraction",
        "saturation_fraction",
    )
    for (phase, reference, domain), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "phase_name": phase,
            "reference_kind": reference,
            "domain": domain,
            "pairs": len(rows),
        }
        for metric in summary_metrics:
            summary[f"{metric}_mean"] = _mean_or_none(row.get(metric) for row in rows)
            summary[f"{metric}_median"] = _median_or_none(row.get(metric) for row in rows)
        summary_rows.append(summary)
    _write_csv(output / "metrics_summary_by_phase.csv", summary_rows)
    figures = _plot_results(output, primary_rows, pair_images) if make_plots else []
    primary_identity_count = sum(
        row["best_orientation_diagnostic"] == "identity" for row in primary_rows
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "type": BENCHMARK_TYPE,
        "session_dir": str(session),
        "pairs": len(pairs),
        "metric_rows": len(metric_rows),
        "global_energy_gain_measured_over_simulation": energy_gain,
        "detector_geometry_payload_sha256": next(iter(detector_geometry_hashes)),
        "camera_exposure_us": next(iter(capture_exposures)),
        "primary_reference": "transport_quantized",
        "primary_domain": "linear",
        "primary_pcc_mean": _mean_or_none(row["pcc_full"] for row in primary_rows),
        "primary_pcc_median": _median_or_none(row["pcc_full"] for row in primary_rows),
        "primary_ssim_mean": _mean_or_none(row["ssim"] for row in primary_rows),
        "primary_cosine_mean": _mean_or_none(row["cosine_similarity"] for row in primary_rows),
        "identity_best_orientation_pairs": primary_identity_count,
        "orientation_diagnostic_note": (
            "Alternative flips are diagnostic only and are never applied to primary metrics."
        ),
        "processing_contract": {
            "fixed_detector_homography": True,
            "per_frame_registration": False,
            "background_subtraction": False,
            "per_frame_minmax_normalization": False,
            "primary_orientation": "canonical_model_xy identity",
        },
        "outputs": {
            "per_pair_csv": str(output / "metrics_per_pair.csv"),
            "summary_csv": str(output / "metrics_summary_by_phase.csv"),
            "figures": figures,
        },
    }
    _write_json(output / "shape_agreement_summary.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="create shape inputs, masks, and simulations")
    generate.add_argument("--output-dir", default="experiments/lab_qwen/shape_agreement")
    generate.add_argument("--phase-center-x", type=float, default=PHASE_CENTER_XY[0])
    generate.add_argument("--phase-center-y", type=float, default=PHASE_CENTER_XY[1])
    generate.add_argument("--overwrite", action="store_true")
    evaluate = subparsers.add_parser("evaluate", help="score acquired canonical CCD frames")
    evaluate.add_argument("--session-dir", default="experiments/lab_qwen/shape_agreement")
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate":
        report = generate_session(
            args.output_dir,
            phase_center_xy=(args.phase_center_x, args.phase_center_y),
            overwrite=args.overwrite,
        )
    else:
        report = evaluate_session(
            args.session_dir,
            output_dir=args.output_dir,
            make_plots=not args.no_plots,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
