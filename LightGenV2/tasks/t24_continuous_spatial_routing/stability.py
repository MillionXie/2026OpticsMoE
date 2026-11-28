"""Apply the preregistered T24 validation-accuracy stopping rule to the paired cohort."""
import argparse
from itertools import product
import json
import math
from pathlib import Path
from statistics import fmean


PROTOCOL_PATH = Path(__file__).resolve().parent / "configs/longrun_matched_20261128.json"


def load_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def run_names(protocol):
    return [f"{data}_s{seed}_{mode}" for data, seed, mode in
            product(protocol["datasets"], protocol["seeds"], protocol["modes"])]


def assess_run(history, protocol=None, *, epoch=None):
    cfg = load_protocol() if protocol is None else protocol
    epoch = len(history) if epoch is None else epoch
    if not 1 <= epoch <= min(len(history), cfg["max_epochs"]):
        raise ValueError("Requested epoch must exist and be within the long-run budget")
    rows = history[:epoch]
    if any(row["epoch"] != i for i, row in enumerate(rows, start=1)):
        raise ValueError("Supply consecutive validation history starting at epoch 1")
    acc = [100.0 * row["validation"]["accuracy"] for row in rows]
    ce = [row["validation"]["cross_entropy"] for row in rows]
    if any(not 0 <= x <= 100 for x in acc) or any(not math.isfinite(x) or x < 0 for x in ce):
        raise ValueError("Invalid validation accuracy or cross entropy")
    window = cfg["adjacent_window_epochs"]
    span = cfg["gain_window_epochs"]
    smooth = cfg["smoothing_epochs"]
    minimum = cfg["min_epochs"]
    eligible = epoch >= minimum and (epoch - minimum) % cfg["decision_interval"] == 0
    result = dict(epoch=epoch, eligible=eligible, plateau=False,
                  accuracy_previous10_pct=None, accuracy_latest10_pct=None, tail_delta_pp=None,
                  best5_previous20_pct=None, best5_latest20_pct=None, best5_gain_pp=None,
                  recent_smoothed_range_pp=None, val_ce_previous10=None, val_ce_latest10=None)
    if epoch < max(2 * window, 2 * span):
        return result
    previous_mean, latest_mean = fmean(acc[-2*window:-window]), fmean(acc[-window:])

    def smoothed(values):
        return [fmean(values[i:i+smooth]) for i in range(len(values)-smooth+1)]

    previous = smoothed(acc[-2*span:-span])
    latest = smoothed(acc[-span:])
    delta = latest_mean - previous_mean
    gain = max(latest) - max(previous)
    spread = max(latest) - min(latest)
    # Inclusive thresholds admit exact decimal ties despite floating-point summation.
    eps = 1e-10
    passed = (abs(delta) <= cfg["max_abs_adjacent_mean_delta_pp"] + eps
              and gain <= cfg["max_best_smoothed_gain_pp"] + eps
              and spread <= cfg["max_recent_smoothed_range_pp"] + eps)
    result.update(plateau=eligible and passed, accuracy_previous10_pct=previous_mean,
                  accuracy_latest10_pct=latest_mean, tail_delta_pp=delta,
                  best5_previous20_pct=max(previous), best5_latest20_pct=max(latest),
                  best5_gain_pp=gain, recent_smoothed_range_pp=spread,
                  val_ce_previous10=fmean(ce[-2*window:-window]), val_ce_latest10=fmean(ce[-window:]))
    return result


def assess_cohort(histories, protocol=None, *, epoch=None):
    cfg = load_protocol() if protocol is None else protocol
    expected = run_names(cfg)
    if set(histories) != set(expected):
        raise ValueError("A cohort decision requires all 12 specified dataset/seed/mode runs")
    epochs = {len(h) for h in histories.values()}
    if epoch is None:
        if len(epochs) != 1:
            raise ValueError("Unequal run lengths; specify a completed common checkpoint")
        epoch = epochs.pop()
    runs = {name: assess_run(histories[name], cfg, epoch=epoch) for name in expected}
    missing = [name for name in expected if not runs[name]["plateau"]]
    if not missing:
        decision, target = "stop_plateau", None
    elif epoch >= cfg["max_epochs"]:
        decision, target = "stop_cap", None
    else:
        checks = range(cfg["min_epochs"], cfg["max_epochs"]+1, cfg["decision_interval"])
        decision, target = "continue", next(e for e in checks if e > epoch)
    return dict(epoch=epoch, decision=decision, next_epoch=target, all_plateau=not missing,
                runs_without_plateau=missing, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path, help="Parent of all twelve long-run directories")
    parser.add_argument("--parents", type=Path, help="Supply archived epochs 1-8 if histories start at 9")
    parser.add_argument("--epoch", type=int, help="Common completed endpoint to assess")
    parser.add_argument("--out", type=Path, help="Optional destination for the JSON decision")
    args = parser.parse_args()
    cfg = load_protocol()
    histories = {}
    for name in run_names(cfg):
        rows = json.loads((args.runs / name / "history.json").read_text(encoding="utf-8"))
        if rows and rows[0]["epoch"] == cfg["parent_epoch"] + 1 and args.parents:
            prior = json.loads((args.parents / name / "history.json").read_text(encoding="utf-8"))
            rows = prior + rows
        histories[name] = rows
    result = assess_cohort(histories, cfg, epoch=args.epoch)
    report = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
