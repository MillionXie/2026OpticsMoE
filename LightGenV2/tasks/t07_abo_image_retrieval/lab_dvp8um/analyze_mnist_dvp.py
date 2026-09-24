"""Read-only MNIST CCD analysis; phase/camera acquisition never depends on labels."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
REFERENCE = ROOT / "ABO_Lab_SHS_8um" / "assets" / "mnist_native8_completed_20260913" / "paired"


def pcc(a, b):
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x.dot(y) / norm) if norm else None


def variants(a):
    return dict(identity=a, flip_h=np.fliplr(a), flip_v=np.flipud(a), rot180=np.rot90(a, 2),
                transpose=a.T, rot90=np.rot90(a, 1), rot270=np.rot90(a, 3),
                anti_transpose=np.rot90(a.T, 2))


def energies(a, bounds):
    return [float(a[y0:y1, x0:x1].sum()) for x0, y0, x1, y1 in bounds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixed-orientation", choices=list(variants(np.empty((1, 1))).keys()))
    parser.add_argument("--simulation-arm", choices=["A", "B", "none"], default="B")
    args = parser.parse_args()
    sim = None
    if args.simulation_arm != "none":
        with np.load(REFERENCE / "pair_reference.npz") as z:
            sim = z["ccd_" + args.simulation_arm].copy()
    ref = json.loads((REFERENCE / "pair_reference.json").read_text(encoding="utf-8"))
    lookup = {r["key"]: i for i, r in enumerate(ref["rows"])}
    paths = sorted(args.images.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No CCD PNGs in {args.images}")
    per_variant = {k: [] for k in variants(np.empty((1, 1)))}
    for path in paths:
        match = re.search(r"mnist_i\d+_y[0-3]", path.stem)
        if not match or match.group() not in lookup:
            continue
        key = match.group()
        i = lookup[key]
        a = np.asarray(Image.open(path), dtype=np.float32)
        if a.shape != (478, 478):
            raise ValueError(f"Expected 478x478 canonical CCD: {path}")
        for name, v in variants(a).items():
            energy = energies(v, ref["detector_bounds"])
            per_variant[name].append(dict(key=key, label=int(ref["rows"][i]["label"]),
                                          prediction=int(np.argmax(energy)), energy=energy,
                                          simulation_prediction=(int(ref["predictions"][args.simulation_arm][i])
                                                                 if sim is not None else None),
                                          pcc=(pcc(v, sim[i]) if sim is not None else None), mean=float(v.mean()),
                                          p99=float(np.percentile(v, 99)),
                                          saturated_fraction=float(np.mean(v >= 255))))
    summaries = {}
    for name, rows in per_variant.items():
        valid_pcc = [r["pcc"] for r in rows if r["pcc"] is not None]
        summaries[name] = dict(n=len(rows), accuracy=float(np.mean([r["label"] == r["prediction"] for r in rows])),
                               mean_pcc=(float(np.mean(valid_pcc)) if valid_pcc else None))
    if sim is None and args.fixed_orientation is None:
        raise ValueError("Flat phase needs a predeclared fixed orientation")
    choice = args.fixed_orientation or max(summaries, key=lambda k: summaries[k]["mean_pcc"])
    report = dict(orientation=choice, orientation_selection="fixed_predeclared" if args.fixed_orientation else "highest_mean_pcc_on_input_set",
                  summary=summaries, selected_rows=per_variant[choice],
                  reference=str(REFERENCE / "pair_reference.npz"), image_directory=str(args.images.resolve()),
                  simulation_arm=args.simulation_arm,
                  photometry="linear Mono8; no per-image normalization or background subtraction")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(orientation=choice, summary=summaries), indent=2))


if __name__ == "__main__":
    main()
