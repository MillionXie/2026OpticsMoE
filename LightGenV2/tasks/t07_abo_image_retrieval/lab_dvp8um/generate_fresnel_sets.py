"""Generate the teacher Fresnel mask and 8 um / 10 cm calibration layouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WAVELENGTH_NM = 532.0
PANEL_WH = (1920, 1200)
PHASE_CENTER_INDEX_XY = (959.5, 599.5)
ACTIVE_SIZE = 1016
ACTIVE_BOUNDS_XYXY = (452, 92, 1468, 1108)  # half-open
IDEAL_PHYSICAL_WIDTH_PX = 478 * 17 / 8  # 1015.75
CCD_CORNERS_TL_TR_BR_BL = ((900, 142), (4310, 142), (4297, 3551), (874, 3544))


def phase_gray(x: np.ndarray, y: np.ndarray, cx: float, cy: float,
               pitch_um: float, distance_cm: float) -> np.ndarray:
    pitch_m = pitch_um * 1e-6
    lam_m = WAVELENGTH_NM * 1e-9
    z_m = distance_cm * 1e-2
    turns = -((x - cx) ** 2 + (y - cy) ** 2) * pitch_m**2 / (2 * lam_m * z_m)
    return np.rint(np.mod(turns, 1.0) * 255.0).astype(np.uint8)


def teacher_original() -> tuple[np.ndarray, list[list[float]], list[list[int]]]:
    """Reproduce generate_fresnel_lens.m: 2x2 of 350, 9.2 um, 30 cm."""
    canvas = np.zeros((1152, 1920), dtype=np.uint8)
    centers = []
    bounds = []
    for row in range(2):
        for col in range(2):
            left = 610 + col * 350
            top = 226 + row * 350
            yy, xx = np.mgrid[top:top + 350, left:left + 350]
            # MATLAB X,Y = 1:350, center_point = 175: local zero-index 174.
            cx, cy = left + 174, top + 174
            canvas[top:top + 350, left:left + 350] = phase_gray(
                xx, yy, cx, cy, 9.2, 30.0)
            centers.append([float(cx), float(cy)])
            bounds.append([left, top, left + 350, top + 350])
    return canvas, centers, bounds


def normal_layout(grid: int) -> tuple[np.ndarray, list[list[float]], list[list[int]]]:
    """Adjacent lenslets whose centers are inside their assigned active cell."""
    x0, y0, x1, y1 = ACTIVE_BOUNDS_XYXY
    assert x1 - x0 == y1 - y0 == ACTIVE_SIZE
    sizes = ([1016] if grid == 1 else [508, 508] if grid == 2 else [339, 338, 339])
    edges = [0]
    for size in sizes:
        edges.append(edges[-1] + size)
    canvas = np.zeros((PANEL_WH[1], PANEL_WH[0]), dtype=np.uint8)
    centers, bounds = [], []
    for row in range(grid):
        for col in range(grid):
            left, right = x0 + edges[col], x0 + edges[col + 1]
            top, bottom = y0 + edges[row], y0 + edges[row + 1]
            cx, cy = (left + right - 1) / 2, (top + bottom - 1) / 2
            yy, xx = np.mgrid[top:bottom, left:right]
            canvas[top:bottom, left:right] = phase_gray(xx, yy, cx, cy, 8.0, 10.0)
            centers.append([cx, cy])
            bounds.append([left, top, right, bottom])
    return canvas, centers, bounds


def roi_layout(grid: int) -> tuple[np.ndarray, list[list[float]], list[list[int]]]:
    """Adjacent lenslets centered on physical ROI vertices / 3x3 ROI nodes.

    The outer lens tiles may extend beyond the phase panel; only the displayed
    part is clipped. No wrapped bitmap is resized after encoding.
    """
    if grid not in (2, 3):
        raise ValueError("ROI layout requires a 2x2 or 3x3 grid")
    cx, cy = PHASE_CENTER_INDEX_XY
    half = IDEAL_PHYSICAL_WIDTH_PX / 2
    axis_x = np.linspace(cx - half, cx + half, grid)
    axis_y = np.linspace(cy - half, cy + half, grid)
    x_edges = [-0.5]
    y_edges = [-0.5]
    x_edges += [(axis_x[i] + axis_x[i + 1]) / 2 for i in range(grid - 1)]
    y_edges += [(axis_y[i] + axis_y[i + 1]) / 2 for i in range(grid - 1)]
    x_edges += [PANEL_WH[0] - 0.5]
    y_edges += [PANEL_WH[1] - 0.5]
    canvas = np.zeros((PANEL_WH[1], PANEL_WH[0]), dtype=np.uint8)
    centers, bounds = [], []
    for row in range(grid):
        for col in range(grid):
            left = max(0, int(np.ceil(x_edges[col] + 0.5)))
            right = min(PANEL_WH[0], int(np.ceil(x_edges[col + 1] + 0.5)))
            top = max(0, int(np.ceil(y_edges[row] + 0.5)))
            bottom = min(PANEL_WH[1], int(np.ceil(y_edges[row + 1] + 0.5)))
            if right <= left or bottom <= top:
                raise ValueError("Empty ROI-centered Fresnel tile")
            target_x, target_y = float(axis_x[col]), float(axis_y[row])
            yy, xx = np.mgrid[top:bottom, left:right]
            canvas[top:bottom, left:right] = phase_gray(
                xx, yy, target_x, target_y, 8.0, 10.0)
            centers.append([target_x, target_y])
            bounds.append([left, top, right, bottom])
    return canvas, centers, bounds


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_gray(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.ascontiguousarray(array), mode="L").save(path, "BMP")
    with Image.open(path) as image:
        if image.format != "BMP" or image.mode != "L" or image.size != (array.shape[1], array.shape[0]):
            raise RuntimeError(f"BMP verification failed: {path}")


def preview(path: Path, array: np.ndarray, centers: list[list[float]]) -> None:
    image = Image.fromarray(array, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, (x, y) in enumerate(centers, 1):
        r = 12
        draw.line((x-r, y, x+r, y), fill=(255, 0, 0), width=3)
        draw.line((x, y-r, x, y+r), fill=(255, 0, 0), width=3)
        draw.text((x+15, y+8), str(index), fill=(255, 240, 0))
    image.thumbnail((960, 600), Image.Resampling.LANCZOS)
    image.save(path)


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("00_teacher_original_30cm_9p2um_1920x1152", *teacher_original(), 30.0, 9.2,
         "original MATLAB geometry and physical parameters"),
        ("01_single_10cm_8um", *normal_layout(1), 10.0, 8.0,
         "one lens centered in the 1016x1016 active square"),
        ("02_four_normal_10cm_8um", *normal_layout(2), 10.0, 8.0,
         "four adjacent cells, lens center at each cell center"),
        ("03_four_roi_vertices_10cm_8um", *roi_layout(2), 10.0, 8.0,
         "four adjacent lenses centered on the physical active ROI vertices"),
        ("04_nine_normal_10cm_8um", *normal_layout(3), 10.0, 8.0,
         "nine adjacent cells, lens center at each cell center"),
        ("05_nine_roi_nodes_10cm_8um", *roi_layout(3), 10.0, 8.0,
         "nine adjacent lenses centered on ROI vertices, edge midpoints and center"),
    ]
    records = []
    for stem, array, centers, bounds, distance_cm, pitch_um, description in cases:
        normal = output / f"{stem}_normal.bmp"
        inverse = output / f"{stem}_inverse255.bmp"
        save_gray(normal, array)
        save_gray(inverse, 255 - array)
        with Image.open(normal) as a, Image.open(inverse) as b:
            if not np.all(np.asarray(a, dtype=np.uint16) +
                          np.asarray(b, dtype=np.uint16) == 255):
                raise RuntimeError(f"Inverse pair check failed: {stem}")
        preview(output / f"{stem}_preview.png", array, centers)
        records.append({
            "name": stem, "description": description,
            "normal_bmp": normal.name, "normal_sha256": sha256(normal),
            "inverse255_bmp": inverse.name, "inverse255_sha256": sha256(inverse),
            "preview_png": f"{stem}_preview.png",
            "size_wh": [array.shape[1], array.shape[0]],
            "wavelength_nm": WAVELENGTH_NM,
            "pixel_pitch_um": pitch_um, "focal_distance_cm": distance_cm,
            "lens_centers_panel_pixel_xy_row_major": centers,
            "clipped_tile_bounds_xyxy_row_major": bounds,
            "gray_min_max": [int(array.min()), int(array.max())],
        })
    white = output / "A_WHITE_1920x1080.bmp"
    save_gray(white, np.full((1080, 1920), 255, np.uint8))
    report = {
        "schema": 1, "phase_size_wh": list(PANEL_WH),
        "active_bounds_panel_xyxy_half_open": list(ACTIVE_BOUNDS_XYXY),
        "phase_center_panel_pixel_xy": list(PHASE_CENTER_INDEX_XY),
        "ideal_active_width_phase_pixels": IDEAL_PHYSICAL_WIDTH_PX,
        "current_ccd_roi_full_sensor_TL_TR_BR_BL": CCD_CORNERS_TL_TR_BR_BL,
        "teacher_source": "C:/Users/Xml12/OneDrive/2026OpticsModel/generate_fresnel_lens.m",
        "formula": "turns=-pitch_m^2*((x-xc)^2+(y-yc)^2)/(2*lambda_m*f_m); uint8=round(255*mod(turns,1))",
        "inverse_formula": "uint8_inverse=255-uint8_normal, including the flat background",
        "warning": "10 cm, 8 um, and 1016-pixel support exceed quadratic Nyquist sampling near the outer edges; teacher formula retained without clipping.",
        "amplitude_white_bmp": white.name,
        "files": records,
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# 菲涅尔标定图（2026-09-23）\n\n"
        "正常/反相两套 BMP 成对输出；反相是整张图逐像素 255-normal。"
        "相位 LUT 的实际方向仍须看焦点实验，不要把文件名当成实测结论。\n\n"
        "00 是老师 MATLAB 原版：30 cm、9.2 µm、1920×1152，保留作对照；"
        "01–05 是当前 8 µm、10 cm、532 nm、1920×1200 相位 SLM 可用文件。\n\n"
        "02/04 的焦点中心在各相邻小格中心；03/05 的焦点中心分别位于"
        "有效相位区域四角、以及四角/边中点/中心。当前相位有效区域"
        "为 x=[452,1468)、y=[92,1108)，这对应此前 478×17 µm "
        "模型光学宽度在 8 µm 面板上的约 1016 像素。CCD 中记录的"
        "四角 [900,142]、[4310,142]、[4297,3551]、[874,3544]"
        "是相机坐标，不直接填到相位 BMP。精确中心见 manifest.json。\n\n"
        "建议振幅 SLM 播放 A_WHITE_1920x1080.bmp，先试 01 聚焦距离，"
        "再用 03 的四焦点对应当前 ROI。预览 PNG 上红十字标示设计中心，"
        "不应加载到 SLM；硬件只加载 BMP。\n\n"
        "10 cm 及这么大的口径在边缘存在相位采样混叠风险；如焦点不理想，"
        "要把有效照明范围与相位符号作为诊断变量，勿直接宣称焦距错。\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "pairs": len(report["files"]),
                      "files": [x["normal_bmp"] for x in report["files"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
