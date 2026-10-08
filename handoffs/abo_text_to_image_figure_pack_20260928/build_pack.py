"""Build auditable ABO text-to-image Top-3 figure assets from saved rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import textwrap
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
MODEL_ORDER = ("Ours (10 cm)", "Qwen3-VL-2B", "DeepSeek-VL2-Tiny", "CLIP ViT-B/32")
CHOSEN_LABELS = (16, 17, 19, 20, 35, 52, 71, 82, 96, 99)
SHORT_NAMES = {
    16: "step_stool", 17: "nightstand", 19: "drive_case", 20: "speaker_mount",
    35: "hdmi_dvi_cable", 52: "waste_basket", 71: "tablet_stand",
    82: "disc_binder", 96: "dresser", 99: "dining_table",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_image_path(value: str) -> str:
    parts = PurePosixPath(value).parts
    if not parts or parts[0] != "images" or ".." in parts:
        raise ValueError(f"Unsafe dataset image path: {value}")
    return PurePosixPath(value).as_posix()


def prepare() -> None:
    test = read_csv(SOURCE / "test.csv")
    titles = read_csv(SOURCE / "titles.csv")
    ours = read_csv(SOURCE / "ours_10cm_predictions.csv")
    qwen = json.loads((SOURCE / "qwen_dynamic_2048_predictions.json").read_text(encoding="utf-8"))
    deepseek = json.loads((SOURCE / "deepseek_predictions.json").read_text(encoding="utf-8"))
    clip = json.loads((SOURCE / "clip_predictions.json").read_text(encoding="utf-8"))
    assert len(test) == 2400 and len(titles) == len(ours) == len(qwen) == 100
    assert len(deepseek["text_to_image_top10"]) == len(clip["text_to_image_top10"]) == 100
    assert all(int(row["label"]) == i for i, row in enumerate(titles))
    by_id = {row["sample_id"]: row for row in test}
    assert len(by_id) == 2400
    by_label = {int(row["label"]): row for row in titles}

    runs = {
        "Ours (10 cm)": [json.loads(row["top10_sample_ids"]) for row in ours],
        "Qwen3-VL-2B": [row["top10_sample_ids"] for row in qwen],
        "DeepSeek-VL2-Tiny": deepseek["text_to_image_top10"],
        "CLIP ViT-B/32": clip["text_to_image_top10"],
    }
    cases = []
    overview_rows = []
    flat = []
    selected_paths: set[str] = set()
    for label in CHOSEN_LABELS:
        title = by_label[label]
        assert title["product_id"] == ours[label]["product_id"] == qwen[label]["product_id"]
        case = {
            "label": label,
            "short_name": SHORT_NAMES[label],
            "query_product_id": title["product_id"],
            "prompt": title["title"],
            "models": {},
        }
        for model in MODEL_ORDER:
            ids = runs[model][label]
            assert len(ids) >= 3 and len(set(ids[:3])) == 3
            predictions = []
            for rank, sample_id in enumerate(ids[:3], 1):
                row = by_id[sample_id]
                retrieved_label = int(row["label"])
                item = {
                    "rank": rank,
                    "sample_id": sample_id,
                    "image_path": verified_image_path(row["image_path"]),
                    "retrieved_label": retrieved_label,
                    "retrieved_product_id": row["product_id"],
                    "retrieved_title": by_label[retrieved_label]["title"],
                    "correct": retrieved_label == label,
                }
                predictions.append(item)
                selected_paths.add(item["image_path"])
                flat.append({
                    "query_label": label,
                    "case": SHORT_NAMES[label],
                    "prompt": title["title"],
                    "query_product_id": title["product_id"],
                    "model": model,
                    **item,
                })
            case["models"][model] = predictions
        assert case["models"][MODEL_ORDER[0]][0]["correct"]
        assert all(not case["models"][model][0]["correct"] for model in MODEL_ORDER[1:])
        assert int(ours[label]["first_positive_rank"]) == 1
        cases.append(case)
        overview_rows.append({
            "case": SHORT_NAMES[label],
            "query_label": label,
            "query_product_id": title["product_id"],
            "prompt": title["title"],
            "ours_top1_product_id": case["models"]["Ours (10 cm)"][0]["retrieved_product_id"],
            "qwen_top1_product_id": case["models"]["Qwen3-VL-2B"][0]["retrieved_product_id"],
            "deepseek_top1_product_id": case["models"]["DeepSeek-VL2-Tiny"][0]["retrieved_product_id"],
            "clip_top1_product_id": case["models"]["CLIP ViT-B/32"][0]["retrieved_product_id"],
        })

    source_hashes = {
        path.name: sha256(path)
        for path in sorted(SOURCE.iterdir())
        if path.is_file() and path.suffix in {".csv", ".json"}
    }
    selection = {
        "protocol": "ABO easy100: 100 official product titles query 2400 test images",
        "ours": "10 cm compact e0p5; best epoch 12; simulation Hit@1 0.86",
        "baseline_variant": "frozen Qwen dynamic 2048D; frozen DeepSeek-VL2-Tiny and CLIP ViT-B/32",
        "selection_rule": "Ours Top-1 correct, all three frozen baseline Top-1 wrong",
        "source_sha256": source_hashes,
        "cases": cases,
    }
    (ROOT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(ROOT / "rankings.csv", flat)
    write_csv(ROOT / "cases_overview.csv", overview_rows)
    (ROOT / "image_paths.txt").write_text("\n".join(sorted(selected_paths)) + "\n", encoding="utf-8")
    print(f"Prepared {len(cases)} cases, {len(flat)} ranking rows, {len(selected_paths)} unique original images")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ("arialbd.ttf", "arial.ttf") if bold else ("arial.ttf", "arialbd.ttf")
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def lines(draw: ImageDraw.ImageDraw, value: str, face: ImageFont.ImageFont, width: int, max_lines: int) -> list[str]:
    words = value.split()
    output: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=face) <= width:
            current = candidate
        else:
            if current:
                output.append(current)
            current = word
    if current:
        output.append(current)
    if len(output) > max_lines:
        output = output[:max_lines]
        while draw.textlength(output[-1] + "...", font=face) > width and output[-1]:
            output[-1] = output[-1][:-1]
        output[-1] += "..."
    return output


def source_image(item: dict[str, object]) -> Path:
    path = ROOT / "raw_dataset" / str(item["image_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.contain(rgb, size, Image.Resampling.LANCZOS)


def draw_photo(canvas: Image.Image, item: dict[str, object], box: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill="#ffffff", outline="#17964c" if item["correct"] else "#cb4545", width=6)
    image = thumbnail(source_image(item), (x1 - x0 - 18, y1 - y0 - 18))
    canvas.paste(image, (x0 + (x1 - x0 - image.width) // 2, y0 + (y1 - y0 - image.height) // 2))


def render_case(case: dict[str, object]) -> Path:
    width, height = 2350, 1370
    canvas = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    heading, body, small = font(36, True), font(27), font(22)
    draw.text((42, 28), f"ABO text-to-image | Case {int(case['label']):02d}: {case['short_name'].replace('_', ' ')}", font=heading, fill="#193b5c")
    for j, line in enumerate(lines(draw, str(case["prompt"]), body, 2240, 2)):
        draw.text((42, 78 + 37 * j), line, font=body, fill="#17212b")
    draw.text((42, 154), "Green = same product ID   |   Red = different product ID   |   Gallery: 2,400 test images", font=small, fill="#5a6673")
    row_top, row_h = 205, 286
    card_x = (365, 1018, 1671)
    for row_idx, model in enumerate(MODEL_ORDER):
        y = row_top + row_idx * row_h
        draw.rounded_rectangle((30, y, 2320, y + row_h - 10), radius=15, fill="#e7edf4" if row_idx % 2 == 0 else "#eef2f7")
        for j, label_line in enumerate(textwrap.wrap(model, width=17)):
            draw.text((48, y + 85 + 31 * j), label_line, font=font(26, True), fill="#17466f")
        for rank_idx, item in enumerate(case["models"][model]):
            x = card_x[rank_idx]
            draw_photo(canvas, item, (x, y + 17, x + 224, y + 222))
            caption = f"Top {rank_idx + 1}  {'CORRECT' if item['correct'] else 'WRONG'}"
            draw.text((x + 240, y + 25), caption, font=font(23, True), fill="#168044" if item["correct"] else "#bd3c3c")
            draw.text((x + 240, y + 64), str(item["retrieved_product_id"]), font=small, fill="#2b3948")
            for k, line in enumerate(lines(draw, str(item["retrieved_title"]), small, 365, 4)):
                draw.text((x + 240, y + 100 + 29 * k), line, font=small, fill="#344454")
    output = ROOT / "panels" / f"case_{int(case['label']):02d}_{case['short_name']}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return output


def render_overview(cases: list[dict[str, object]]) -> Path:
    width, row_h = 2240, 258
    canvas = Image.new("RGB", (width, 125 + len(cases) * row_h), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 21), "ABO text-to-image | Top-1 across four models", font=font(38, True), fill="#173e62")
    draw.text((28, 73), "Ours: 10 cm simulation, Hit@1 0.86   |   Green: correct SKU   Red: wrong SKU", font=font(25), fill="#51606f")
    x_positions = (460, 905, 1350, 1795)
    for i, case in enumerate(cases):
        y = 125 + i * row_h
        draw.rectangle((0, y, width, y + row_h), fill="#f2f6fa" if i % 2 == 0 else "#ffffff")
        draw.text((24, y + 25), f"{int(case['label']):02d} | {case['short_name'].replace('_', ' ')}", font=font(27, True), fill="#163f61")
        for j, line in enumerate(lines(draw, str(case["prompt"]), font(20), 400, 4)):
            draw.text((24, y + 71 + 25 * j), line, font=font(20), fill="#3c4b5a")
        for model, x in zip(MODEL_ORDER, x_positions):
            item = case["models"][model][0]
            draw.text((x, y + 14), model, font=font(22, True), fill="#17466f")
            draw_photo(canvas, item, (x, y + 52, x + 164, y + 222))
            draw.text((x + 178, y + 95), "RIGHT" if item["correct"] else "WRONG", font=font(20, True), fill="#168044" if item["correct"] else "#bd3c3c")
            draw.text((x + 178, y + 129), str(item["retrieved_product_id"]), font=font(17), fill="#394959")
    output = ROOT / "overview_top1.png"
    canvas.save(output, optimize=True)
    return output


def render() -> None:
    selection = json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))
    cases = selection["cases"]
    index = []
    for case in cases:
        panel = render_case(case)
        case_folder = ROOT / "originals_by_case" / f"case_{int(case['label']):02d}_{case['short_name']}"
        case_folder.mkdir(parents=True, exist_ok=True)
        for model_idx, model in enumerate(MODEL_ORDER):
            for item in case["models"][model]:
                src = source_image(item)
                dst = case_folder / f"{model_idx + 1:02d}_{model.split()[0].lower()}_top{item['rank']}_{item['sample_id']}{src.suffix.lower()}"
                shutil.copy2(src, dst)
                index.append({
                    "case": case["short_name"], "query_label": case["label"],
                    "model": model, "rank": item["rank"], "correct": item["correct"],
                    "sample_id": item["sample_id"], "original_relative_path": dst.relative_to(ROOT).as_posix(),
                    "source_dataset_path": item["image_path"], "sha256": sha256(dst),
                    "panel_relative_path": panel.relative_to(ROOT).as_posix(),
                })
    write_csv(ROOT / "asset_index.csv", index)
    overview = render_overview(cases)
    print(f"Rendered {len(cases)} Top-3 panels and {overview.name}; {len(index)} original-image copies indexed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "render"))
    arguments = parser.parse_args()
    prepare() if arguments.stage == "prepare" else render()
