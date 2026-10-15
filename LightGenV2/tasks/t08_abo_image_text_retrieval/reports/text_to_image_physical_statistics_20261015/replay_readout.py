"""CPU-only replay of the adopted 85% readout on sealed measured features.

Never loads Qwen, optical model or hardware. Never trains or modifies sources.
Refuses different assets and refuses to overwrite an existing output.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

BODY = "cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae"
READOUT = "89e25360c8f12c9d13dbceba49e8d81073d4dd75c6acf667ab12a38b8fd293c5"
CACHE = "f0387d9ad5187f728cc2bc7da56c4ff0a27c179992045d7b3c22e9744ac8d2ed"
CSV_SHA = {"test": "4ca5834abf4bb4d614b41f05b5b2481b3d41720dd78c1c37b05923c97657dfb8",
           "titles": "c97e8da3c6c608a706f2cc0d284aeafb873fc6cc17d04a7c9f9f66b2135846c1"}
TENSOR_SHA = {"image_features": "8dd1a89dc5eb1e0d52c2a56ed39bb4968976cdac4f546e30116fb4e53303ecec",
              "title_features": "cd814710dc625c7278301e4e3dd92086f49a5476ab9f7275eb3226d8d406392e"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cache_path, checkpoint_path, data_root):
    assert sha(cache_path) == CACHE, "Wrong measured cache"
    assert sha(checkpoint_path) == READOUT, "Wrong adopted readout"
    rows = {}
    for name, expected in CSV_SHA.items():
        src = data_root / f"{name}.csv"
        assert sha(src) == expected, f"Wrong {name} CSV"
        with src.open(encoding="utf-8-sig", newline="") as f:
            rows[name] = list(csv.DictReader(f))
    assert len(rows["test"]) == 2400 and len(rows["titles"]) == 100
    assert [int(r["label"]) for r in rows["titles"]] == list(range(100))
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert cache["checkpoint_sha256"] == BODY and cache["split"] == "test"
    assert cache["image_keys"] == [f"image_{i:04d}" for i in range(2400)]
    assert cache["title_keys"] == [f"title_{i:03d}" for i in range(100)]
    assert cache["image_labels"] == [int(r["label"]) for r in rows["test"]]
    for key, count in (("image_features", 2400), ("title_features", 100)):
        t = cache[key]
        assert list(t.shape) == [count, 384] and t.dtype == torch.float32 and t.device.type == "cpu"
        assert bool(torch.isfinite(t).all())
        assert hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() == TENSOR_SHA[key]
    state = checkpoint["retrieval_readout"]
    assert set(state) == {"norm.weight", "norm.bias", "projection.weight", "projection.bias"}
    assert list(state["projection.weight"].shape) == [64, 384]
    assert all(bool(torch.isfinite(v).all()) for v in state.values())

    def readout(x):
        # Exact ElectronicRetrievalReadout: default LayerNorm eps=1e-5,
        # signed Linear(384,64), then default F.normalize eps=1e-12.
        normalized = F.layer_norm(x.float(), (384,), state["norm.weight"], state["norm.bias"], eps=1e-5)
        return F.normalize(F.linear(normalized, state["projection.weight"], state["projection.bias"]), dim=-1)

    with torch.no_grad():
        scores = readout(cache["title_features"]) @ readout(cache["image_features"]).T
        order = scores.argsort(dim=1, descending=True, stable=True)
    labels = torch.tensor(cache["image_labels"], dtype=torch.long)
    relevant = labels[order].eq(torch.arange(100)[:, None])
    assert bool(relevant.sum(1).eq(24).all())
    predictions = []
    for label, title in enumerate(rows["titles"]):
        positive_ranks = (relevant[label].nonzero().flatten() + 1).tolist()
        first = positive_ranks[0]
        top = order[label, :10].tolist()
        assert all(rows["test"][i]["product_id"] == title["product_id"] for i in order[label][relevant[label]].tolist())
        ap = sum((i + 1) / rank for i, rank in enumerate(positive_ranks)) / 24
        predictions.append({"title_label": label, "product_id": title["product_id"], "title": title["title"],
                            "first_positive_rank": first, "hit_at_1": int(first <= 1),
                            "hit_at_5": int(first <= 5), "hit_at_10": int(first <= 10),
                            "reciprocal_rank": 1 / first, "average_precision": ap,
                            "positive_ranks": positive_ranks,
                            "top10_sample_ids": [rows["test"][i]["sample_id"] for i in top],
                            "top10_product_ids": [rows["test"][i]["product_id"] for i in top],
                            "top10_scores": scores[label, top].tolist()})
    metrics = {f"hit_at_{k}": sum(p[f"hit_at_{k}"] for p in predictions) / 100 for k in (1, 5, 10)}
    metrics.update(mrr=sum(p["reciprocal_rank"] for p in predictions) / 100,
                   map=sum(p["average_precision"] for p in predictions) / 100)
    assert [sum(p[f"hit_at_{k}"] for p in predictions) for k in (1, 5, 10)] == [85, 97, 97], "Replay differs from adopted result"
    assert abs(metrics["mrr"] - .8952448964118958) < 2e-6
    assert abs(metrics["map"] - .7978265285491943) < 2e-6
    assert sha(cache_path) == CACHE and sha(checkpoint_path) == READOUT
    return {"mode": "fixed_adopted_readout_cpu_replay", "torch": torch.__version__,
            "device": "cpu", "dtype": "float32", "threads": 1, "training": False,
            "body_sha256": BODY, "readout_sha256": READOUT, "cache_sha256": CACHE,
            "data_sha256": CSV_SHA, "query_count": 100, "gallery_count": 2400,
            "relevant_images_per_query": 24, "metrics": metrics, "predictions": predictions,
            "limitation": "Replays sealed measured-feature cache; does not revalidate raw CCD-to-feature reconstruction or independent generalization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.cache, args.checkpoint, args.data_root)
    result["evaluator_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
