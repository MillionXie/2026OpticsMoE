"""Make honest, paired visualizations of the Qwen optical CCD feature planes.

Raw nonnegative FP32 intensity remains in the agreement NPZ files.  This tool
adds the exact network-input map and explicitly labelled display-only PNGs; it
never overwrites or rescales the scientific reference arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_common import (
    STAGES,
    read_json,
    stage_directory,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate import (
    network_input_map,
    pair_metrics,
)


REFERENCE_COLUMNS = {
    "ideal_model_fp32": "ideal_reference_file",
    "transport_quantized": "transport_reference_file",
}

STAGE_DISPLAY_NAMES = {
    "vision_expert": "01 Vision expert (MoE4)",
    "vision_global": "02 Vision global",
    "language_expert": "03 Language expert (MoE4)",
    "language_global": "04 Language global",
}

# Compact approximation of matplotlib's perceptually ordered viridis map.  The
# scientific arrays are never colorized; this LUT is used only for PNG display.
VIRIDIS_ANCHORS = np.asarray(
    [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ],
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write an empty CCD feature manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_intensity(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        value = np.asarray(payload["intensity"], dtype=np.float32)
    if value.ndim != 2 or np.any(value < 0) or not np.isfinite(value).all():
        raise RuntimeError(f"Invalid theoretical CCD: {path}")
    return value


def _save_l(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="L").save(path)
    return _sha256(path)


def _viridis(value: np.ndarray) -> np.ndarray:
    gray = np.asarray(value, dtype=np.float64).clip(0.0, 255.0) / 255.0
    position = gray * (len(VIRIDIS_ANCHORS) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(VIRIDIS_ANCHORS) - 1)
    fraction = (position - lower)[..., None]
    rgb = VIRIDIS_ANCHORS[lower] * (1.0 - fraction) + VIRIDIS_ANCHORS[upper] * fraction
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def _save_rgb(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB").save(path)
    return _sha256(path)


def _display_linear(value: np.ndarray, relative_clip: float) -> np.ndarray:
    source = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    mean = max(float(source.mean()), 1.0e-12)
    shape = np.minimum(source / mean, relative_clip) / relative_clip
    return np.rint(255.0 * shape).clip(0, 255).astype(np.uint8)


def _display_network(value: np.ndarray, relative_clip: float, log_compression: float) -> np.ndarray:
    maximum = math.log1p(log_compression * relative_clip)
    return np.rint(255.0 * np.asarray(value) / maximum).clip(0, 255).astype(np.uint8)


def _contact_sheet(
    paths: list[tuple[Path, str]], destination: Path, *, columns: int = 5
) -> str | None:
    if not paths:
        return None
    tile, label_height = 224, 28
    rows = int(math.ceil(len(paths) / columns))
    sheet = Image.new("RGB", (columns * tile, rows * (tile + label_height)), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(paths):
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize((tile, tile), Image.Resampling.BILINEAR)
        x = (index % columns) * tile
        y = (index // columns) * (tile + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 3, y + tile + 3), label[:34], fill=(255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return _sha256(destination)


def _four_stage_sheet(
    keys: list[str],
    stage_images: dict[str, dict[str, Path]],
    destination: Path,
) -> str | None:
    """Lay the same capture keys out as rows and the four optical stages as columns."""
    stages = [stage for stage in STAGES if stage in stage_images]
    keys = [key for key in keys if all(key in stage_images[stage] for stage in stages)]
    if not stages or not keys:
        return None
    tile, header, row_label = 224, 36, 190
    sheet = Image.new(
        "RGB",
        (row_label + len(stages) * tile, header + len(keys) * tile),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(sheet)
    for column, stage in enumerate(stages):
        draw.text(
            (row_label + column * tile + 5, 11),
            STAGE_DISPLAY_NAMES.get(stage, stage),
            fill=(255, 255, 255),
        )
    for row, key in enumerate(keys):
        y = header + row * tile
        draw.text((5, y + 8), key[:29], fill=(255, 255, 255))
        for column, stage in enumerate(stages):
            with Image.open(stage_images[stage][key]) as opened:
                image = opened.convert("RGB").resize(
                    (tile, tile), Image.Resampling.BILINEAR
                )
            sheet.paste(image, (row_label + column * tile, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return _sha256(destination)


def _representative_keys(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    designed = [row["capture_key"] for row in rows if row["source_kind"] == "designed"]
    # One fixed test example per class produces a compact, unbiased visual index.
    model: list[str] = []
    observed_skus: set[str] = set()
    for row in rows:
        if row["source_kind"] == "designed":
            continue
        sku = row.get("sku_index", "")
        if sku not in observed_skus:
            observed_skus.add(sku)
            model.append(row["capture_key"])
    return designed[:8], model[:10]


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None],
        dtype=np.float64,
    )
    return None if values.size == 0 else float(values.mean())


def export_gallery(session_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir).expanduser().resolve()
    root_contract = read_json(session / "agreement_manifest.json")
    reports: list[dict[str, Any]] = []
    easy_view = session / "VIEW_THEORETICAL_CCD"
    easy_view.mkdir(parents=True, exist_ok=True)
    stage_images: dict[str, dict[str, Path]] = {}
    representative_designed: list[str] = []
    representative_model: list[str] = []
    for stage in STAGES:
        stage_dir = stage_directory(session, stage)
        if not (stage_dir / "probe_manifest.csv").is_file():
            continue
        contract = read_json(stage_dir / "agreement_contract.json")
        agreement = contract["agreement"]
        relative_clip = float(agreement["relative_clip"])
        log_compression = float(agreement["log_compression"])
        output = stage_dir / "ccd_feature_visualization"
        rows: list[dict[str, Any]] = []
        quantization_rows: list[dict[str, Any]] = []
        sheets: list[dict[str, Any]] = []
        source_rows = _read_csv(stage_dir / "probe_manifest.csv")
        designed_keys, model_keys = _representative_keys(source_rows)
        if not representative_designed:
            representative_designed = designed_keys
        if not representative_model:
            representative_model = model_keys
        easy_stage = easy_view / STAGE_DISPLAY_NAMES.get(stage, stage)
        stage_images[stage] = {}
        all_easy_tiles: dict[str, list[tuple[Path, str]]] = {
            "designed": [],
            "model": [],
        }
        for reference_kind, column in REFERENCE_COLUMNS.items():
            designed_tiles: list[tuple[Path, str]] = []
            model_tiles: list[tuple[Path, str]] = []
            for source in source_rows:
                intensity_path = stage_dir / source[column]
                intensity = _load_intensity(intensity_path)
                network = network_input_map(
                    intensity,
                    relative_clip=relative_clip,
                    log_compression=log_compression,
                )
                key = source["capture_key"]
                linear_png = output / reference_kind / "linear_shape_478" / f"{key}.png"
                network_npz = output / reference_kind / "network_input_224" / f"{key}.npz"
                network_png = output / reference_kind / "network_input_224" / f"{key}.png"
                linear_sha = _save_l(
                    linear_png, _display_linear(intensity, relative_clip)
                )
                network_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(network_npz, network_input=network.astype(np.float32))
                network_sha = _save_l(
                    network_png,
                    _display_network(network, relative_clip, log_compression),
                )
                if reference_kind == "transport_quantized":
                    linear_display = _display_linear(intensity, relative_clip)
                    network_display = _display_network(
                        network, relative_clip, log_compression
                    )
                    easy_gray = easy_stage / "01_linear_gray_478" / f"{key}.png"
                    easy_color = easy_stage / "02_linear_color_478" / f"{key}.png"
                    easy_network = easy_stage / "03_network_input_color_224" / f"{key}.png"
                    _save_l(easy_gray, linear_display)
                    _save_rgb(easy_color, _viridis(linear_display))
                    _save_rgb(easy_network, _viridis(network_display))
                    stage_images[stage][key] = easy_color
                    kind = "designed" if source["source_kind"] == "designed" else "model"
                    all_easy_tiles[kind].append((easy_color, key))
                rows.append(
                    {
                        "stage": stage,
                        "capture_key": key,
                        "source_kind": source["source_kind"],
                        "role": source["role"],
                        "sku_index": source["sku_index"],
                        "sku_name": source["sku_name"],
                        "reference_kind": reference_kind,
                        "raw_intensity_npz": source[column],
                        "raw_mean": float(intensity.mean()),
                        "raw_std": float(intensity.std()),
                        "raw_max": float(intensity.max()),
                        "linear_shape_png": linear_png.relative_to(stage_dir).as_posix(),
                        "linear_shape_png_sha256": linear_sha,
                        "network_input_npz": network_npz.relative_to(stage_dir).as_posix(),
                        "network_input_npz_sha256": _sha256(network_npz),
                        "network_input_png": network_png.relative_to(stage_dir).as_posix(),
                        "network_input_png_sha256": network_sha,
                        "network_mean": float(network.mean()),
                        "network_std": float(network.std()),
                    }
                )
                target = designed_tiles if source["source_kind"] == "designed" else model_tiles
                limit = 8 if source["source_kind"] == "designed" else 20
                if len(target) < limit:
                    target.append((network_png, f"{source['source_kind']}:{key}"))
            for name, tiles in (("designed", designed_tiles), ("model", model_tiles)):
                path = output / f"CONTACT_SHEET_{reference_kind}_{name}.png"
                digest = _contact_sheet(tiles, path)
                if digest:
                    sheets.append(
                        {
                            "file": path.relative_to(stage_dir).as_posix(),
                            "sha256": digest,
                            "reference_kind": reference_kind,
                            "source_kind": name,
                        }
                    )
        for source in source_rows:
            ideal = _load_intensity(stage_dir / source["ideal_reference_file"])
            transport = _load_intensity(stage_dir / source["transport_reference_file"])
            row: dict[str, Any] = {
                "stage": stage,
                "capture_key": source["capture_key"],
                "source_kind": source["source_kind"],
                "role": source["role"],
                "sku_index": source["sku_index"],
                "sku_name": source["sku_name"],
            }
            for domain in ("linear", "network_input"):
                metrics = pair_metrics(
                    transport,
                    ideal,
                    domain=domain,
                    energy_gain=1.0,
                    saturation_value=0.0,
                    signal_energy_fraction=float(agreement["signal_energy_fraction"]),
                    relative_clip=relative_clip,
                    log_compression=log_compression,
                )
                row.update({f"{domain}_{key}": value for key, value in metrics.items()})
            quantization_rows.append(row)
        manifest = output / "ccd_feature_manifest.csv"
        _write_csv(manifest, rows)
        quantization_manifest = output / "transport_vs_ideal_metrics.csv"
        _write_csv(quantization_manifest, quantization_rows)
        quantization_summary = {
            "samples": len(quantization_rows),
            "linear_pcc_mean": _mean_metric(quantization_rows, "linear_pcc_full"),
            "linear_ssim_mean": _mean_metric(quantization_rows, "linear_ssim"),
            "linear_shape_nrmse_mean": _mean_metric(
                quantization_rows, "linear_shape_nrmse"
            ),
            "network_input_pcc_mean": _mean_metric(
                quantization_rows, "network_input_pcc_full"
            ),
            "network_input_ssim_mean": _mean_metric(
                quantization_rows, "network_input_ssim"
            ),
            "network_input_shape_nrmse_mean": _mean_metric(
                quantization_rows, "network_input_shape_nrmse"
            ),
        }
        guide = output / "README.md"
        guide.write_text(
            "# CCD feature visualization\n\n"
            "`theoretical_ccd/*/*.npz` is the scientific raw linear intensity.\n"
            "`network_input_224/*.npz` is the exact nonnegative -> frame-mean -> "
            "relative-clip -> log1p -> AdaptiveAvgPool(224) map consumed before "
            "the learned readout. PNGs and contact sheets are display-only. No "
            "background subtraction or per-frame min-max enters the network.\n",
            encoding="utf-8",
        )
        for kind, tiles in all_easy_tiles.items():
            _contact_sheet(
                tiles,
                easy_stage / f"OPEN_ALL_{kind.upper()}_LINEAR_COLOR.png",
                columns=5,
            )
        reports.append(
            {
                "stage": stage,
                "features": len(rows),
                "manifest": str(manifest),
                "manifest_sha256": _sha256(manifest),
                "contact_sheets": sheets,
                "transport_vs_ideal_manifest": str(quantization_manifest),
                "transport_vs_ideal_manifest_sha256": _sha256(
                    quantization_manifest
                ),
                "transport_vs_ideal_summary": quantization_summary,
            }
        )
    open_first = {
        "designed": _four_stage_sheet(
            representative_designed,
            stage_images,
            easy_view / "OPEN_ME_FIRST_DESIGNED_PROBES.png",
        ),
        "test": _four_stage_sheet(
            representative_model,
            stage_images,
            easy_view / "OPEN_ME_FIRST_TEST_ONE_PER_CLASS.png",
        ),
    }
    (easy_view / "README_先看这里.md").write_text(
        "# Qwen 四层理论 CCD：肉眼查看入口\n\n"
        "先打开 `OPEN_ME_FIRST_TEST_ONE_PER_CLASS.png` 和 "
        "`OPEN_ME_FIRST_DESIGNED_PROBES.png`。每一行是同一个输入，每一列依次是 "
        "vision expert、vision global、language expert、language global。\n\n"
        "各层文件夹内：\n\n"
        "- `01_linear_gray_478`：478×478，单帧均值归一化后按固定相对强度上限显示的灰度图；\n"
        "- `02_linear_color_478`：与上一项数值完全相同，仅套 viridis 色表，方便看弱结构；\n"
        "- `03_network_input_color_224`：网络在归一化、截断、log1p 和 224×224 池化后实际读取的形状；\n"
        "- `OPEN_ALL_*.png`：该层所有样本的联系表。\n\n"
        "显示 PNG 只用于主观观察，不能用于 PCC/SSIM。正式指标仍读取 "
        "`theoretical_ccd/transport_quantized/*.npz`，实测图也必须走相同的网络映射。"
        "伪彩图的紫色表示低强度，黄色表示高强度。\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "session": str(session),
        "agreement_manifest_checkpoint_sha256": root_contract["checkpoint_sha256"],
        "stages": reports,
        "easy_view_directory": str(easy_view),
        "open_first_sha256": open_first,
    }
    (session / "ccd_feature_gallery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(export_gallery(args.session_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
