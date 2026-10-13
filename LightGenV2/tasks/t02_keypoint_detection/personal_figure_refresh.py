"""Redraw an existing handoff from original photos, preserving model-space metrics."""
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .personal_figure_delivery import EDGES, GT_COLOR, MODEL_COLOR
from .personal_prepare import sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def geometry(keypoints, row, original_size):
    """224 model -> working frame -> upright original -> bounded display crop."""
    ww, wh = row["size_wh"]
    ow, oh = original_size
    left, top, right, bottom = keypoints["crop_box_source_xyxy"]
    clipped = [max(0, left), max(0, top), min(ww, right), min(wh, bottom)]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("Crop has no real photo pixels")
    sx, sy = ow / ww, oh / wh
    bounds = [max(0, math.floor(clipped[0] * sx)), max(0, math.floor(clipped[1] * sy)),
              min(ow, math.ceil(clipped[2] * sx)), min(oh, math.ceil(clipped[3] * sy))]
    def transform(values):
        xy = np.asarray(values, dtype=np.float64).copy()
        xy[:, 0] = (left + xy[:, 0] * (right - left) / 224) * sx - bounds[0]
        xy[:, 1] = (top + xy[:, 1] * (bottom - top) / 224) * sy - bounds[1]
        return xy
    return bounds, transform(keypoints["gt_xy_224"]), transform(keypoints["LightGen2_xy_224"])


def draw_skeleton(base, xy, valid, color, cross=False, line_fraction=.014):
    # Supersample only the transparent drawing layer; the photograph stays sharp.
    scale = 2
    layer = Image.new("RGBA", (base.width * scale, base.height * scale))
    draw = ImageDraw.Draw(layer)
    width = max(6, round(min(base.size) * line_fraction)) * scale
    radius = max(5, round(width / scale * .70)) * scale
    p = xy * scale
    for a, b in EDGES:
        if valid[a] and valid[b]:
            draw.line([tuple(p[a]), tuple(p[b])], fill=color, width=width)
    for (x, y), usable in zip(p, valid):
        if not usable:
            continue
        if cross:
            draw.line((x-radius, y, x+radius, y), fill="white", width=width+scale*2)
            draw.line((x, y-radius, x, y+radius), fill="white", width=width+scale*2)
            draw.line((x-radius, y, x+radius, y), fill=color, width=width)
            draw.line((x, y-radius, x, y+radius), fill=color, width=width)
        else:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="white", width=scale*2)
    layer = layer.resize(base.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


def font(size):
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def run(delivery, dataset, originals, work_dir, max_side=2000, apply=False):
    delivery, dataset, originals, work_dir = map(lambda p: Path(p).resolve(), (delivery, dataset, originals, work_dir))
    if max_side < 896:
        raise ValueError("Use at least 896px export limit")
    if work_dir.exists():
        raise FileExistsError(work_dir)
    work_dir.mkdir(parents=True)
    stage = work_dir / "rendered"
    stage.mkdir()
    annotations = read_json(dataset / "annotations_provisional.json")
    lookup = {r["id"]: r for r in annotations["images"]}
    folders = sorted(p for p in delivery.iterdir() if p.is_dir() and (p / "keypoints.json").exists())
    protected = [delivery / n for n in ["SUMMARY.json", "SUMMARY.md", "SAMPLE_METRICS.json"]]
    protected += [p / n for p in folders for n in ["keypoints.json", "metrics.json", "metrics.md"]]
    protected_sha = {str(p.relative_to(delivery)): sha256(p) for p in protected}
    old_summary = read_json(delivery / "SUMMARY.json")
    rows, audit, padding_count = [], [], 0
    for folder in folders:
        row = lookup[folder.name.rsplit("_", 1)[0]]
        key = read_json(folder / "keypoints.json")
        path = (originals / row["original_name"]).resolve()
        if not path.is_relative_to(originals) or sha256(path) != row["original_sha256"]:
            raise ValueError(f"Original identity mismatch: {folder.name}")
        with Image.open(path) as raw:
            original = ImageOps.exif_transpose(raw).convert("RGB")
        if row.get("extra_rotation_ccw", 0):
            original = original.rotate(row["extra_rotation_ccw"], expand=True)
        if list(original.size) != row["upright_size_wh"]:
            raise ValueError("Upright orientation differs from annotation reference")
        bounds, gt, pred = geometry(key, row, original.size)
        native_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        base = original.crop(bounds)
        base.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        ratio = np.asarray(base.size) / np.asarray(native_size)
        gt, pred = gt * ratio, pred * ratio
        valid = np.asarray(key["GT_valid"], dtype=bool)
        pred_valid = np.ones(len(pred), dtype=bool)
        gt_image = draw_skeleton(base, gt, valid, GT_COLOR)
        ours_image = draw_skeleton(base, pred, pred_valid, MODEL_COLOR)
        overlay = draw_skeleton(draw_skeleton(base, gt, valid, GT_COLOR), pred, pred_valid, MODEL_COLOR, cross=True)
        out = stage / folder.name
        out.mkdir()
        panels = [base, gt_image, ours_image, overlay]
        for name, panel in zip(["input", "GT", "LightGen2", "overlay"], panels):
            panel.save(out / (name + ".png"), compress_level=4)
        previews = []
        for panel in panels:
            small = panel.copy()
            small.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
            previews.append(small)
        pw, ph = previews[0].size
        header = 58
        sheet = Image.new("RGB", (4 * pw, ph + header), "white")
        d = ImageDraw.Draw(sheet)
        for i, (panel, title) in enumerate(zip(previews, ["Input", "GT", "LightGen2", "GT + LightGen2"])):
            sheet.paste(panel, (pw * i, header))
            d.text((pw * i + 14, 12), title, fill="#202020", font=font(30))
        sheet.save(out / "preview.png", compress_level=4)
        old = key["crop_box_source_xyxy"]
        ww, wh = row["size_wh"]
        padded = old[0] < 0 or old[1] < 0 or old[2] > ww or old[3] > wh
        padding_count += int(padded)
        inside = lambda xy: (xy[:, 0] >= 0) & (xy[:, 0] < base.width) & (xy[:, 1] >= 0) & (xy[:, 1] < base.height)
        display = {"original_file": row["original_name"], "original_sha256": row["original_sha256"],
                   "upright_original_size_wh": list(original.size), "display_crop_original_xyxy": bounds,
                   "export_size_wh": list(base.size), "artificial_padding_removed": padded,
                   "line_width_px": max(6, round(min(base.size) * .014)), "line_width_short_edge_fraction": .014,
                   "GT_xy_display": gt.tolist(), "LightGen2_xy_display": pred.tolist(),
                   "GT_visible_on_display": (inside(gt) & valid).tolist(), "prediction_visible_on_display": inside(pred).tolist(),
                   "model_coordinates_and_metrics_unchanged": True}
        write_json(out / "display_geometry.json", display)
        audit.append({"sample_id": folder.name, **display})
        m = read_json(folder / "metrics.json")
        cards = "".join(f'<figure><a href="{folder.name}/{name}.png" target="_blank"><img loading="lazy" src="{folder.name}/{name}.png?v=20261013"></a><figcaption>{title}</figcaption></figure>'
                        for name, title in zip(["input", "GT", "LightGen2", "overlay"], ["输入", "GT", "LightGen2", "GT + LightGen2"]))
        rows.append(f'<section id="{folder.name}"><h2>{html.escape(folder.name)}</h2><p>PCK@0.2：{m["pck_at_0.2_torso"]:.2%} · 定位误差：{m["mean_pixel_error_224"]:.2f} px（224模型坐标） · 原图导出：{base.width} × {base.height}</p><div class="grid">{cards}</div><p><a href="{folder.name}/preview.png" target="_blank">四列拼图</a> · <a href="{folder.name}/metrics.md">样本指标</a></p></section>')
        print(f'{folder.name}: {base.width}x{base.height}, line={display["line_width_px"]}, padding_removed={padded}', flush=True)
    notice = "GT 由自动预标注生成，人工校验状态按用户确认记录。仅展示12个四肢关节；蓝色：GT，橙色：LightGen2。"
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LightGen2 原图高清展示</title><style>body{font:16px system-ui;color:#202020;background:#f2f4f7;margin:24px}main{max-width:1800px;margin:auto}section{background:white;padding:20px;margin:26px 0;border-radius:8px}p{line-height:1.7}h2{font-size:20px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;align-items:start}figure{margin:0}img{display:block;width:100%;height:auto}figcaption{text-align:center;font-weight:600;padding:10px}a{color:#0066a0}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:560px){.grid{grid-template-columns:1fr}body{margin:10px}}</style><main><h1>LightGen2 · 原图高清展示</h1><p>'+notice+f'</p><p>{old_summary["photos"]} 张照片 / {len(folders)} 个人体 · PCK@0.2 {old_summary["pck_at_0.2_torso"]:.2%} · <a href="SUMMARY.md">总体指标</a></p><p>点击单图查看原尺寸。图片从手机原照片直接裁取，去除裁剪越界形成的黑框；所有模型坐标、预测及指标保持原记录。页面采用自然宽高比，照片不拉伸。</p>'+"".join(rows)+'</main></html>'
    (stage / "PREVIEW.html").write_text(page, encoding="utf-8")
    (stage / "README.md").write_text('# LightGen2 高清画图交付\n\n打开 PREVIEW.html；点击单图可查看或保存原尺寸。每个人体目录有 input.png、GT.png、LightGen2.png、overlay.png 四张无标题图，以及 preview.png 四列拼图。\n\n手机原照片经原来的EXIF方向处理后直接裁取；只保留原模型裁剪框与真实照片的交集，不再经过224缩小，不进行照片拉伸、生成填充或锐化。长边最多2000像素，不放大小尺寸原图。线宽按短边1.4%设置，绘图透明层抗锯齿。\n\nGT蓝色，LightGen2橙色；叠加中GT圆点、LightGen2十字。GT由RTMPose预标注后人工校验（用户确认），本次未修改标注文件。历史指标文件的标签来源说明原样保留。\n\nmetrics.md/JSON、keypoints.json保持原文件；SUMMARY.md/JSON、SAMPLE_METRICS.json为原总体/逐样本指标，仍在224模型坐标计算。display_geometry.json仅记录展示裁剪和坐标映射：去黑边会改变图片边界，不会改变预测或PCK。相机原本未拍到的身体部分不会补画；位于照片外的预测仍保留在坐标文件中。\n\nRENDER_AUDIT.json记录原图SHA、方向、真实像素裁剪范围、导出分辨率和旧版恢复包；FILES_SHA256.json是新版文件清单。旧版完整ZIP在交付目录之外单独保存。\n', encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rendering = {"date": "2026-10-13", "source_commit_at_execution": commit,
                 "renderer_sha256": sha256(Path(__file__)), "samples": len(folders),
                 "padding_removed_samples": padding_count, "maximum_export_side": max_side,
                 "source": "original phone photograph; no 224px intermediate", "protected_files_sha256": protected_sha,
                 "gt_human_review": "user-confirmed in 2026-09-29 conversation; annotation file not altered",
                 "samples_geometry": audit}
    write_json(stage / "RENDER_AUDIT.json", rendering)
    if not apply:
        print(f"Prepared reviewable images: {stage}")
        return
    # Recoverable replacement: verify a complete archive before updating presentation assets.
    backup_dir = delivery.parent / "figure_delivery_backups"
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / (delivery.name + "_before_highres_20261013.zip")
    if backup.exists():
        raise FileExistsError(backup)
    old_files = sorted(p for p in delivery.rglob("*") if p.is_file())
    with zipfile.ZipFile(backup, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for p in old_files:
            archive.write(p, str(p.relative_to(delivery)))
    with zipfile.ZipFile(backup) as archive:
        if len(archive.infolist()) != len(old_files) or archive.testzip() is not None:
            raise ValueError("Backup verification failed")
    write_json(backup.with_suffix(".json"), {"archive": str(backup), "sha256": sha256(backup), "file_count": len(old_files)})
    rendering.update(previous_delivery_backup=str(backup), backup_sha256=sha256(backup))
    write_json(stage / "RENDER_AUDIT.json", rendering)
    for p in stage.rglob("*"):
        if p.is_file():
            target = delivery / p.relative_to(stage)
            temp = target.with_name(target.name + ".render-tmp")
            shutil.copyfile(p, temp)
            temp.replace(target)
    for relative, expected in protected_sha.items():
        if sha256(delivery / relative) != expected:
            raise ValueError(f"Scientific result changed: {relative}")
    write_json(delivery / "FILES_SHA256.json", {str(p.relative_to(delivery)).replace("\\", "/"): sha256(p)
               for p in sorted(delivery.rglob("*")) if p.is_file() and p.name != "FILES_SHA256.json"})
    write_json(work_dir / "receipt.json", {"delivery": str(delivery), "samples": len(folders),
        "padding_removed_samples": padding_count, "backup": str(backup), "metrics_and_model_coordinates_unchanged": True})
    print(f"Updated {delivery}; original scientific files verified unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=2000)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(args.delivery, args.dataset, args.originals, args.work_dir, args.max_side, args.apply)
