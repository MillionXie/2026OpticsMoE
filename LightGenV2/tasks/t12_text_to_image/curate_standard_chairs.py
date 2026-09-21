"""Create a reproducible shape-clean chair subset from an ABO CleanRender dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .dataset import sha256, validate_split_contract, write_manifest
from .settings import TASK_DIR


def curate(source: Path, output: Path, selection_path: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    split_counts = {}
    identity_counts = {}
    for split in ("train", "val", "test"):
        allowed = set(selection[split])
        rows = [json.loads(line) for line in (source / f"{split}.jsonl").read_text().splitlines()]
        available = {row["sequence_id"] for row in rows}
        missing = allowed - available
        if missing:
            raise ValueError(f"Missing {split} sequences: {sorted(missing)}")
        chosen = [row for row in rows if row["sequence_id"] in allowed]
        for row in chosen:
            source_image = source / row["image_path"]
            target_image = output / row["image_path"]
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, target_image)
        write_manifest(output / f"{split}.jsonl", chosen)
        split_counts[split] = len(chosen)
        identity_counts[split] = len({row["sequence_id"] for row in chosen})
    contract = validate_split_contract(output)
    summary = {
        **contract,
        "variant": "ABO-CleanRender-StandardChair",
        "selection_policy": selection["selection_policy"],
        "source_dataset": str(source.resolve()),
        "selection_file": str(selection_path.resolve()),
        "selection_sha256": sha256(selection_path),
        "images": split_counts,
        "identities": identity_counts,
        "one_object_per_image": True,
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path,
        default=TASK_DIR / "configs/abo_standard_chair_sequences.json",
    )
    args = parser.parse_args()
    result = curate(
        args.source.expanduser().resolve(), args.output.expanduser().resolve(),
        args.selection.expanduser().resolve(),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
