"""CPU-only error decomposition of preserved predictions/features; no model run."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from . import evaluate_temporal_baseline_batches as frozen


def softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def error_statistics(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("Incomparable/nonfinite arrays")
    delta = candidate - reference
    return dict(max_abs=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))),
                relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)),
                changed_entries=int(np.count_nonzero(delta)), entries=int(delta.size))


def pair_order_changes(reference, candidate):
    """Count strict prediction-pair inversions; ties reported separately."""
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("Expected aligned prediction vectors")
    indices = np.triu_indices(len(reference), 1)
    left = np.sign(reference[:, None] - reference[None, :])[indices]
    right = np.sign(candidate[:, None] - candidate[None, :])[indices]
    return dict(strict_order_reversals=int(np.count_nonzero(left * right < 0)),
                tie_state_changes=int(np.count_nonzero((left == 0) != (right == 0))),
                compared_pairs=len(left))


def decompose(reference, candidate, weights, scores):
    """Exact linear identity in CPU FP64 separates changed features from readout residual."""
    f1, f2 = reference["features"], candidate["features"]
    l1, l2 = reference["logits"], candidate["logits"]
    projected1, projected2 = f1 @ weights.T, f2 @ weights.T
    explained_delta = (f2-f1) @ weights.T
    residual = (l2-l1) - explained_delta
    p1, p2 = softmax(l1), softmax(l2)
    mos1, mos2 = p1 @ scores, p2 @ scores
    projection_mos1, projection_mos2 = softmax(projected1) @ scores, softmax(projected2) @ scores
    return dict(feature_difference=error_statistics(f1, f2),
                saved_logit_difference=error_statistics(l1, l2),
                feature_explained_logit_delta_max_abs=float(np.abs(explained_delta).max()),
                unexplained_cross_batch_logit_residual_max_abs=float(np.abs(residual).max()),
                single_projection_residual_max_abs=float(np.abs(l1-projected1).max()),
                batch_projection_residual_max_abs=float(np.abs(l2-projected2).max()),
                probability_max_abs_delta=float(np.abs(p2-p1).max()),
                max_probability_total_variation=float(np.abs(p2-p1).sum(-1).max()/2),
                saved_logits_cpu64_mos_max_abs_delta=float(np.abs(mos2-mos1).max()),
                changed_features_cpu64_head_mos_max_abs_delta=float(np.abs(projection_mos2-projection_mos1).max()),
                cpu64_reprojection_max_abs_mos_difference=float(max(np.abs(mos1-projection_mos1).max(), np.abs(mos2-projection_mos2).max())),
                fixed_features_replayed_batch_head_logit_max_abs_delta=float(max(
                    np.abs(np.concatenate([f1[i:i+b] @ weights.T for i in range(0,len(f1),b)])-projected1).max()
                    for b in (1,2,4))),
                per_video_mos_delta=(mos2-mos1).tolist(),
                per_video_feature_max_abs=np.abs(f2-f1).max(-1).tolist())


def audit_full_predictions(root, baseline):
    from scipy.stats import spearmanr, rankdata
    source, result = {}, {}
    for batch in (1,2,4,8):
        path = root / f"{baseline}_batch{batch:02d}" / "predictions.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 558 or len({r["sample_id"] for r in rows}) != 558:
            raise ValueError("Full TEST identity/count mismatch")
        source[batch] = rows
    first = source[1]
    p1 = np.array([float(r["prediction"]) for r in first])
    target = np.array([float(r["target_mos"]) for r in first])
    identity = [(r["sample_id"],r["target_mos"],r["selected_frame_positions"]) for r in first]
    for batch, rows in source.items():
        if identity != [(r["sample_id"],r["target_mos"],r["selected_frame_positions"]) for r in rows]:
            raise ValueError("Full TEST sample order/target/frame identity changed")
        p = np.array([float(r["prediction"]) for r in rows])
        result[str(batch)] = dict(srcc=float(spearmanr(p,target).statistic),
            mos_difference=error_statistics(p1,p), prediction_order=pair_order_changes(p1,p),
            max_abs_rank_shift=float(np.abs(rankdata(p)-rankdata(p1)).max()),
            sha256=frozen.sha256(root/f"{baseline}_batch{batch:02d}"/"predictions.csv"))
    return result


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU-only analysis requires CUDA_VISIBLE_DEVICES=-1")
    if args.output.exists():
        raise FileExistsError("Preserve existing analysis")
    task = Path(__file__).resolve().parent
    if not args.output.is_relative_to((task/"runs/simulation").resolve()):
        raise ValueError("Output must be in canonical task run")
    import torch
    torch.set_num_threads(2)
    diagnosis = json.loads((args.diagnostic/"diagnosis.json").read_text())
    if diagnosis["checkpoint_sha256"] != frozen.sha256(args.checkpoint):
        raise ValueError("Checkpoint not bound to preserved diagnosis")
    payload = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    weights = payload["state_dict"]["weight"].double().numpy()
    scores = payload["level_scores"].double().numpy()
    modes = {}
    sources = {"diagnosis.json": frozen.sha256(args.diagnostic/"diagnosis.json")}
    for mode in diagnosis["probes"]:
        snapshots = {}
        sample_ids = None
        for batch in (1,2,4):
            path = args.diagnostic/f"{mode}_b{batch}.pt"
            saved = torch.load(path,map_location="cpu",weights_only=False)
            if len(saved["sample_ids"]) != 8 or (sample_ids is not None and sample_ids != saved["sample_ids"]):
                raise ValueError("Preserved probe sample identity changed")
            sample_ids = saved["sample_ids"]
            snapshots[batch] = {key:saved[key].double().numpy() for key in ("features","logits")}
            sources[path.name] = frozen.sha256(path)
        modes[mode] = {str(b):decompose(snapshots[1],snapshots[b],weights,scores) for b in (1,2,4)}
    result = dict(baseline=args.baseline,scope="CPU-only preserved eight-video feature/readout decomposition and full558 prediction-rank audit",
        checkpoint_sha256=frozen.sha256(args.checkpoint), manifest_sha256=diagnosis["manifest_sha256"],
        preserved_source_sha256=sources, analysis_source_sha256=frozen.sha256(Path(__file__)),
        modes=modes, full558=audit_full_predictions(args.formal_root,args.baseline),
        limits=["No new backbone forward, training, GPU measurements, formal score or cost replacement",
                "A changed final feature locates error upstream of five-row head, not its first internal layer",
                "FP64 CPU readout is diagnostic recomputation, not a new formal precision protocol"])
    args.output.mkdir(parents=True,exist_ok=False)
    frozen.write_json(args.output/"analysis.json",result)
    print(json.dumps({"baseline":args.baseline,"native_B4":modes["native"]["4"],"full558":result["full558"]},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",choices=("qwen","deepseek"),required=True)
    for name in ("diagnostic","checkpoint","formal-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,key,value.resolve())
    run(args)


if __name__ == "__main__":
    main()
