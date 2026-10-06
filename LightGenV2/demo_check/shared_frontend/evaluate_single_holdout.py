"""Evaluate one validation-selected shared-frontend run on its sealed test split."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("holdout_runner", HERE / "run.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    status = json.loads((args.run / "status.json").read_text())
    assert status["state"] == "complete"
    source = json.loads((args.run / "metadata.json").read_text())
    cfg, optical_cfg = source["config"], source["optical_config"]
    selected = []
    for architecture in ("dynamic_four", "full_d2nn"):
        summary = json.loads((args.run / architecture / "summary.json").read_text())
        checkpoint = args.run / architecture / "best_checkpoint.pt"
        assert r.sha(checkpoint) == summary["checkpoint_sha256"]
        selected.append(dict(
            architecture=architecture,
            path=str(checkpoint),
            checkpoint_sha256=r.sha(checkpoint),
            selected_epoch=summary["selected_epoch"],
            validation=summary["validation"],
            train_unaugmented=summary["train_unaugmented"],
        ))
    frontend_path = args.run / "frontend" / "best_checkpoint.pt"
    frontend_sha = r.sha(frontend_path)
    r.save(args.out / "selection_lock.json", dict(
        selection="validation only; no tuning after test access",
        models=selected,
        frontend_sha256=frontend_sha,
    ))

    manifest_path = args.data.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert r.sha(args.data) == manifest["data_sha256"]
    assert manifest["original_split_sha256"] == source["split_sha256"]
    assert manifest["train_validation_spatial_overlap"] == 0
    with np.load(args.data, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    assert all(key.startswith("test_") for key in arrays)
    test = tuple(torch.from_numpy(arrays["test_" + key]) for key in ("images", "labels", "domains"))
    for domain in (0, 1):
        assert torch.bincount(test[1][test[2] == domain], minlength=10).tolist() == [100] * 10

    frontend_payload = torch.load(frontend_path, map_location="cpu", weights_only=False)
    frontend = r.SharedFrontend(frontend_payload["model"]).cuda()
    frozen_sha = r.tensors_sha(frontend.state_dict())
    r.save(args.out / "metadata.json", dict(
        command=sys.argv,
        config=cfg,
        optical_config=optical_cfg,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        python=sys.version,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        data_sha256=r.sha(args.data),
        data_manifest_sha256=r.sha(manifest_path),
        operation="locked validation-selected checkpoints evaluated once on the independent spatial test subset",
    ))
    shutil.copyfile(manifest_path, args.out / "test_manifest.json")
    results = []
    for entry in selected:
        model = r.FrontendOptics(frontend, entry["architecture"], optical_cfg).cuda()
        checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=False)
        assert checkpoint["frontend_tensors_sha256"] == frozen_sha
        model.optical.load_state_dict(checkpoint["model"])
        metrics, probabilities = r.base.evaluate(model, test, cfg)
        with (args.out / f"{entry['architecture']}_predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "domain", "label", "prediction"] + [f"p{i}" for i in range(10)])
            writer.writerows(
                [str(index), int(domain), int(label), int(probs.argmax()), *probs.tolist()]
                for index, domain, label, probs in zip(
                    arrays["test_ids"], arrays["test_domains"], arrays["test_labels"], probabilities
                )
            )
        results.append(dict(**entry, test=metrics))
        print(json.dumps(dict(architecture=entry["architecture"], test_accuracy=metrics["accuracy"])), flush=True)
        del model
        torch.cuda.empty_cache()
    assert r.tensors_sha(frontend.state_dict()) == frozen_sha
    r.save(args.out / "results.json", results)
    r.save(args.out / "status.json", dict(state="complete"))


if __name__ == "__main__":
    main()
