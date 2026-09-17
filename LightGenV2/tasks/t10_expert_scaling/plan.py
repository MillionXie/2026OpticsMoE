"""Generate an auditable design matrix. Does not load data or launch training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_protocol():
    return json.loads((ROOT / "configs/study.json").read_text(encoding="utf-8"))


def geometry(cfg, n, fixed_global=None):
    g = math.isqrt(n)
    if g * g != n:
        raise ValueError("Expert count must form a square grid")
    geom, model = cfg["geometry"], cfg["model"]
    e, gap = geom["expert_side_px"], geom["gap_px"]
    occupied = g * e + (g - 1) * gap
    active = occupied if fixed_global is None else fixed_global
    if active < occupied or (active - occupied) % 2:
        raise ValueError("Expert array must fit and center on the active plane")
    layers = model["feature_layers_provisional"]
    if layers < 2 or layers % 2:
        raise ValueError("Feature depth must consist of expert/global pairs")
    cycles = layers // 2
    router = model["router_side_px"] ** 2
    expert_params = cycles * n * e * e
    global_params = cycles * active * active
    target = expert_params + global_params + router
    ideal = math.sqrt(target / layers)
    candidates = {max(2, 2 * math.floor(ideal / 2)), 2 * math.ceil(ideal / 2)}
    side = min((s for s in candidates if s <= active), key=lambda s: abs(layers * s * s - target))
    return dict(experts=n, grid=g, expert_side_px=e, gap_px=gap,
                occupied_side_px=occupied, active_side_px=active,
                canvas_side_px=active + 2 * geom["outer_margin_px"],
                active_side_mm=round(active * geom["pixel_pitch_um"] / 1000, 4),
                feature_layers=layers, router_phase_parameters=router,
                expert_phase_parameters=expert_params,
                global_phase_parameters=global_params,
                moe_phase_parameters=target, d2nn_parameter_side_px=side,
                d2nn_parameter_count=layers * side * side,
                d2nn_parameter_relative_error=(layers * side * side - target) / target,
                d2nn_aperture_count=layers * active * active)


def k_values(cfg, n, pilot=False):
    sweep = cfg["sweep"]
    if pilot:
        values = set(sweep["pilot_fixed_k"] + [n])
    else:
        values = set(sweep["fixed_k"])
        values.update(math.ceil(n / d) for d in sweep["fractional_k_denominators"])
    return sorted(k for k in values if 1 <= k <= n)


def router_regions(cfg):
    m = cfg["model"]
    g, width, pitch, side = (m[k] for k in
                             ["router_detector_grid", "router_detector_side_px",
                              "router_detector_pitch_px", "router_side_px"])
    start = (side - ((g - 1) * pitch + width)) // 2
    cells = [(y, x) for y in range(g) for x in range(g)]
    cells.sort(key=lambda v: ((2*v[0]-(g-1))**2 + (2*v[1]-(g-1))**2, v[0], v[1]))
    return [[start+y*pitch, start+y*pitch+width, start+x*pitch, start+x*pitch+width]
            for y, x in cells]


def matrix(cfg, datasets, pilot=False, fixed_global=False):
    sweep = cfg["sweep"]
    ns = sweep["fixed_global_control_expert_counts"] if fixed_global else (
        sweep["pilot_expert_counts"] if pilot else cfg["expert_counts"])
    global_side = geometry(cfg, max(cfg["expert_counts"]))["active_side_px"] if fixed_global else None
    rows = []
    for dataset in datasets:
        for seed in cfg["training_proposal"]["initial_seeds"]:
            for n in ns:
                geo = geometry(cfg, n, global_side)
                ks = sorted({min(4, n), n}) if fixed_global else k_values(cfg, n, pilot)
                arms = [("moe_oeo", k) for k in ks]
                arms += [(arm, None) for arm in sweep["baselines"]]
                # With a fixed global plane this baseline is identical for all N.
                if fixed_global and n != ns[0]:
                    arms = [(a, k) for a, k in arms if a != "d2nn_same_aperture"]
                for arm, k in arms:
                    row = dict(dataset=dataset, seed=seed, protocol=cfg["protocol"],
                               geometry_contract="fixed_global" if fixed_global else "growing_global",
                               model=arm, reference_experts=n, top_k=k,
                               selected_fraction=None if k is None else k/n,
                               active_side_px=geo["active_side_px"],
                               canvas_side_px=geo["canvas_side_px"],
                               phase_side_px=(geo["d2nn_parameter_side_px"] if arm == "d2nn_total_parameter"
                                              else geo["active_side_px"] if arm == "d2nn_same_aperture" else 224),
                               feature_layers=geo["feature_layers"],
                               status="planned_not_started")
                    rows.append(row)
    return rows


def check(cfg):
    assert cfg["geometry"]["expert_side_px"] == 224
    assert cfg["geometry"]["gap_px"] == 30
    assert cfg["geometry"]["pixel_pitch_um"] == 17
    assert cfg["geometry"]["phase_device_pixel_pitch_um"] == 8
    assert cfg["geometry"]["hardware_size_limit_applied"] is False
    assert 2 * cfg["model"]["rgb_channel_side_px"] == 224
    assert cfg["model"]["oeo_after_every_feature_layer"]
    regions = router_regions(cfg)
    assert len(regions) == 49
    pixels = set()
    for y0, y1, x0, x1 in regions:
        assert 0 <= y0 < y1 <= 224 and 0 <= x0 < x1 <= 224
        covered = {(y, x) for y in range(y0, y1) for x in range(x0, x1)}
        assert not pixels.intersection(covered)
        pixels.update(covered)
    for n in cfg["expert_counts"]:
        g = geometry(cfg, n)
        assert len(set(k_values(cfg, n))) == len(k_values(cfg, n))
        assert 1 in k_values(cfg, n) and n in k_values(cfg, n)
        assert abs(g["d2nn_parameter_relative_error"]) <= cfg["model"]["phase_parameter_tolerance_fraction"]
        assert g["d2nn_parameter_side_px"] % 2 == 0
        # Non-overlapping expert areas are contained in their active plane.
        assert n * 224**2 <= g["occupied_side_px"] ** 2
        # Unit total entrance power, independent of N and k; no cloned power.
        p = [(i + 1) / (n * (n + 1) / 2) for i in range(n)]
        for k in k_values(cfg, n):
            selected = sorted(range(n), key=lambda i: p[i], reverse=True)[:k]
            amplitude = [math.sqrt(p[i] / sum(p[j] for j in selected)) for i in selected]
            assert math.isclose(sum(a*a for a in amplitude), 1.0, abs_tol=1e-12)
    assert geometry(cfg, 4)["active_side_px"] == 478
    assert geometry(cfg, 49)["active_side_px"] == 1748
    # This verifies planning arithmetic, not optical gradients or accuracy.


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "reports/design")
    args = ap.parse_args()
    cfg = load_protocol()
    check(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = cfg["datasets"]["primary"]
    geo = [dict(geometry(cfg, n), top_k="/".join(map(str, k_values(cfg, n)))) for n in cfg["expert_counts"]]
    pilot = matrix(cfg, datasets, pilot=True)
    main_rows = matrix(cfg, datasets)
    controls = matrix(cfg, datasets, fixed_global=True)
    write_csv(args.output / "geometry.csv", geo)
    write_csv(args.output / "pilot_matrix.csv", pilot)
    write_csv(args.output / "main_matrix.csv", main_rows)
    write_csv(args.output / "fixed_global_matrix.csv", controls)
    regions = {"contract": "fixed_49_ports_center_out_first_N_row_major_ties",
               "bounds_y0_y1_x0_x1": router_regions(cfg)}
    (args.output / "router_regions.json").write_text(json.dumps(regions, indent=2)+"\n", encoding="utf-8")
    summary = dict(status="design_only_no_training_results", checks="passed_geometry_matrix_entrance_power",
                   protocol_sha256=hashlib.sha256((ROOT / "configs/study.json").read_bytes()).hexdigest(),
                   primary_datasets=datasets, pilot_runs=len(pilot), main_runs_including_pilot=len(main_rows),
                   main_models=dict(Counter(r["model"] for r in main_rows)),
                   fixed_global_control_runs=len(controls),
                   optional_dataset_main_runs=len(matrix(cfg, cfg["datasets"]["optional"])),
                   excluded_from_counts=["depth_and_learning_rate_calibration", "routing_ablation",
                                         "joint_task_runs", "noise_sensitivity", "additional_seeds"],
                   hardware_limit_applied=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
