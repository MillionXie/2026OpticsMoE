"""Render a few audited RGB/SAR pairs and their actual optical input fields."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from LightGenV2.tasks.t14_shared_readout_lifelong.single_task_eurosat import source_paths

from .data import PairedEuroSatFields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    trainval, holdout = source_paths(args.protocol)
    data = PairedEuroSatFields(trainval, holdout, "train")
    protocol = json.loads(args.protocol.read_text())
    expected = protocol.get("source_protocol", protocol)["pair_counts"]
    pair_sets = {"train": set(map(str, data.pair_ids))}
    assert len(data) == expected["train"] == len(pair_sets["train"])
    for split, source_name in (("val", "validation"), ("test", "test")):
        other = PairedEuroSatFields(trainval, holdout, split)
        pair_sets[split] = set(map(str, other.pair_ids))
        assert len(other) == expected[source_name] == len(pair_sets[split])
        del other
    assert not (pair_sets["train"] & pair_sets["val"]
                or pair_sets["train"] & pair_sets["test"]
                or pair_sets["val"] & pair_sets["test"])
    print("verified paired RGB/SAR split sizes:", expected, flush=True)
    if args.count < 1 or args.count > 8:
        raise ValueError("count must be in 1..8")
    indices = np.linspace(0, len(data) - 1, args.count, dtype=np.int64)
    fig, axes = plt.subplots(args.count, 3, figsize=(10, 3 * args.count),
                             squeeze=False, constrained_layout=True)
    for row, index in enumerate(indices):
        rgb = data.rgb[index]
        sar = data.sar[index]
        amplitude = data[int(index)].numpy()
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"RGB; pair {data.pair_ids[index]}; class {data.labels[index]}")
        axes[row, 1].imshow((sar[..., 0].astype(np.float32)
                              + sar[..., 1].astype(np.float32)) / (2 * 255),
                            cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("SAR: mean of normalized VV/VH")
        axes[row, 2].imshow(amplitude, cmap="magma")
        axes[row, 2].axvline(111.5, color="cyan", lw=.8)
        axes[row, 2].axhline(111.5, color="cyan", lw=.8)
        axes[row, 2].set_title("Optical field: R | G / B | SAR")
        for ax in axes[row]:
            ax.axis("off")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
