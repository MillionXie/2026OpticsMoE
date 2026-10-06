"""CPU metrics on small fixtures; never call historical main/train or CUDA."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "LightGenV2/demo_check"


def function(path, name, environment):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), environment)
    return environment[name]


class DemoEntryTests(unittest.TestCase):
    def test_original_runtime_source_identity(self):
        manifest = json.loads((BASE / "HISTORICAL_RUNTIME_IDENTITY_20261006.json").read_text())
        self.assertEqual(len(manifest["sources"]), 3)
        for row in manifest["sources"]:
            payload = (ROOT / row["path"]).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), row["sha256"])
            ast.parse(payload)

    def test_original_frozen_fusion_config(self):
        config = json.loads((BASE / "frozen_electronic/config.json").read_text())
        self.assertEqual(config["electronic_weight"], .5)
        self.assertEqual(config["optical_weight"], .5)
        self.assertFalse(config["test_set_used"])
        self.assertEqual(config["architectures"], ["dynamic_four", "full_d2nn"])

    def test_fusion_metrics_use_all_rows_and_domain_split(self):
        import numpy as np
        metric = function(BASE / "frozen_electronic/run.py", "metric", {"np": np})
        probabilities = np.full((4, 10), .01)
        probabilities[range(4), [0, 1, 3, 3]] = .91
        labels = np.array([0, 1, 2, 3])
        domains = np.array([0, 0, 1, 1])
        report = metric(probabilities, labels, domains)
        self.assertEqual(report["accuracy"], .75)
        self.assertEqual(report["domain_accuracy"], {"0": 1., "1": .5})
        self.assertEqual(sum(map(sum, report["confusion_matrix"])), 4)
        self.assertAlmostEqual(report["loss"], float(-np.log([.91,.91,.01,.91]).mean()))

    def test_phase_only_holdout_metric_on_cpu_fixture(self):
        import numpy as np
        import torch
        evaluate = function(BASE / "pure_optical/evaluate_holdout.py", "evaluate",
                            {"np": np, "torch": torch})
        probabilities = torch.tensor([[.8,.2],[.1,.9],[.6,.4],[.2,.8]])
        class Model:
            def __call__(self, indices):
                return {"probabilities": probabilities[indices]}
        data = (torch.arange(4), torch.tensor([0,1,1,1]), torch.tensor([0,0,1,1]))
        report, prediction = evaluate(Model(), data, batch=2)
        self.assertEqual(report["accuracy"], .75)
        self.assertEqual(report["domain_accuracy"], {"0": 1., "1": .5})
        np.testing.assert_array_equal(prediction, probabilities.numpy())


if __name__ == "__main__":
    unittest.main()
