"""Test T06 candidate Git blobs on CPU without altering any existing checkout.

Original backend tests run from an existing fixture directory only after exact
test-source SHA comparison. Imports come from candidate blobs, not that directory.
Two existing fixed PTs are strict-loaded; no dataset evaluation or devices used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

PREFIXES = (
    "LightGenV2/tasks/t06_video_quality_assessment/",
    "experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/",
    "experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/",
)


class TreeImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, blobs, fixture_root):
        self.blobs, self.fixture_root = blobs, fixture_root

    def location(self, name):
        path = name.replace(".", "/")
        for candidate, package in ((path + "/__init__.py", True), (path + ".py", False)):
            if candidate in self.blobs:
                return candidate, package
        if any(key.startswith(path + "/") for key in self.blobs):
            return path + "/__init__.py", True
        return None

    def find_spec(self, fullname, path=None, target=None):
        if not any(fullname == prefix.rstrip("/").replace("/", ".") or
                   fullname.startswith(prefix.replace("/", ".")) for prefix in PREFIXES):
            return None
        found = self.location(fullname)
        return importlib.util.spec_from_loader(fullname, self, is_package=found[1]) if found else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, package = self.location(module.__name__)
        data = self.blobs.get(path, b"")
        module.__file__ = str(self.fixture_root / path)
        module.__git_blob_sha256__ = hashlib.sha256(data).hexdigest()
        if package:
            module.__path__ = [str((self.fixture_root / path).parent)]
        exec(compile(data, module.__file__, "exec"), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--adaptation-tests", action="store_true",
                        help="Also test the candidate's original offline readout protocol")
    args = parser.parse_args()
    def git(*parts):
        return subprocess.check_output(["git", "-C", str(args.repository), *parts])
    entries = git("ls-tree", "-r", "-z", args.commit, "--", *PREFIXES).split(b"\0")
    rows = [entry.split(b"\t", 1) for entry in entries if entry]
    rows = [(meta.split()[-1], path.decode()) for meta, path in rows if path.endswith(b".py")]
    if not rows:
        raise RuntimeError("Candidate has no T06 source")
    payload = subprocess.check_output(
        ["git", "-C", str(args.repository), "cat-file", "--batch"],
        input=b"".join(oid + b"\n" for oid, _ in rows))
    offset, blobs = 0, {}
    for _, path in rows:
        end = payload.index(b"\n", offset)
        size = int(payload[offset:end].split()[-1])
        blobs[path] = payload[end+1:end+1+size]
        compile(blobs[path], path, "exec")
        offset = end + size + 2
    test_path = PREFIXES[1] + "tests/test_model_and_training.py"
    fixture = args.fixture_root / test_path
    if fixture.read_bytes() != blobs[test_path]:
        raise RuntimeError("Original test fixture is not byte-identical to candidate")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    import torch
    import pytest
    torch.set_num_threads(2)
    for name in ("LightGenV2", "LightGenV2.tasks", "experiments"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules["LightGenV2"].tasks = sys.modules["LightGenV2.tasks"]
    sys.meta_path.insert(0, TreeImporter(blobs, args.fixture_root))
    runtime = importlib.import_module("LightGenV2.tasks.t06_video_quality_assessment.lab_runtime")
    reloads = []
    for target in ("spatial", "temporal"):
        checkpoint = args.checkpoint_root / runtime.PINS[target]["run"] / "best_checkpoint.pt"
        model, settings = runtime.load_model(target, checkpoint, "cpu")
        reloads.append({"target": target, "strict": True, "sha256": runtime.sha(checkpoint),
                        "architecture": settings.architecture_label,
                        "parameters": sum(p.numel() for p in model.parameters())})
        del model
    code = int(pytest.main([str(fixture), "-q", "-p", "no:cacheprovider"]))
    adaptation = None
    if args.adaptation_tests:
        import unittest
        test_module = importlib.import_module(
            "LightGenV2.tasks.t06_video_quality_assessment.test_adapt_measured_readout")
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if result.testsRun != 12 or result.skipped or not result.wasSuccessful():
            code = 1
        adaptation = {"tests_run": result.testsRun, "skipped": len(result.skipped),
                      "successful": result.wasSuccessful(),
                      "source_sha256": test_module.__git_blob_sha256__}
    imported = sys.modules["experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.modeling"]
    expected = hashlib.sha256(blobs[PREFIXES[1] + "modeling.py"]).hexdigest()
    if imported.__git_blob_sha256__ != expected:
        raise RuntimeError("Tests did not use candidate model")
    print(json.dumps({"commit": args.commit, "python_blobs_compiled": len(blobs),
                      "backend_test_exit_code": code, "strict_reloads": reloads,
                      "adaptation_protocol_tests": adaptation,
                      "candidate_model_sha256": expected, "new_checkout_created": False,
                      "dataset_evaluated": False, "cuda_visible_devices": ""}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
