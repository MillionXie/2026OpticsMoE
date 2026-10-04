"""Run T11 CPU contracts directly from pinned Git blobs, without a new checkout."""
import argparse
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest


class TreeImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, blobs):
        self.blobs = blobs

    def location(self, name):
        if name.startswith("tasks."):
            name = "LightGenV2." + name
        path = name.replace(".", "/")
        for candidate, package in ((path + "/__init__.py", True), (path + ".py", False)):
            if candidate in self.blobs:
                return candidate, package
        if any(key.startswith(path + "/") for key in self.blobs):
            return path + "/__init__.py", True
        return None

    def find_spec(self, fullname, path=None, target=None):
        if not (fullname.startswith("LightGenV2.tasks.t11_lifelong_optics") or
                fullname.startswith("tasks.t11_lifelong_optics")):
            return None
        found = self.location(fullname)
        return importlib.util.spec_from_loader(fullname, self, is_package=found[1]) if found else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, package = self.location(module.__name__)
        module.__file__ = "git-tree/" + path
        if package:
            module.__path__ = []
        exec(compile(self.blobs.get(path, b""), module.__file__, "exec"), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repository", help="Repository path when running the tool from Git via stdin")
    args = parser.parse_args()
    root = Path(args.repository).resolve() if args.repository else Path(__file__).resolve().parents[2]
    prefix = "LightGenV2/tasks/t11_lifelong_optics/"
    entries = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "-z", args.commit, "--", prefix]).split(b"\0")
    rows = [entry.split(b"\t", 1) for entry in entries if entry]
    rows = [(metadata.split()[-1], path.decode()) for metadata, path in rows if path.endswith(b".py")]
    payload = subprocess.check_output(["git", "-C", str(root), "cat-file", "--batch"], input=b"".join(oid + b"\n" for oid, _ in rows))
    offset, blobs = 0, {}
    for _, path in rows:
        end = payload.index(b"\n", offset)
        size = int(payload[offset:end].split()[-1])
        blobs[path] = payload[end+1:end+1+size]
        compile(blobs[path], path, "exec")
        offset = end + size + 2
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(2)
    for name in ("LightGenV2", "LightGenV2.tasks", "tasks"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules["LightGenV2"].tasks = sys.modules["LightGenV2.tasks"]
    sys.meta_path.insert(0, TreeImporter(blobs))
    suite = unittest.TestSuite()
    for name in ("LightGenV2.tasks.t11_lifelong_optics.tests.test_contract",
                 "LightGenV2.tasks.t11_lifelong_optics.tests.test_crc9_contract"):
        module = importlib.import_module(name)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({"commit": args.commit, "python_blobs_compiled": len(blobs),
                      "tests_run": result.testsRun, "success": result.wasSuccessful(),
                      "cuda_visible_devices": "", "new_checkout_created": False}))
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
