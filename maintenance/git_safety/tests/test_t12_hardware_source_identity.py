"""Small CPU checks only: no task model, data, checkpoint, CUDA or SDK."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "LightGenV2/tasks/t12_text_to_image/lab_shs8um"


class HardwareSourceTests(unittest.TestCase):
    def test_all_seventeen_deployed_sources(self):
        receipt = json.loads((BASE / "SOURCE_IDENTITY_20261006.json").read_text())
        self.assertEqual(len(receipt["sources"]), 17)
        for row in receipt["sources"]:
            data = (ROOT / row["path"]).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256_lf"], row["path"])
            ast.parse(data, filename=row["path"])

    def test_train_capture_replacements_match(self):
        tree = ast.parse((BASE / "run_train_capture.py").read_text())
        replacements = next(
            ast.literal_eval(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "changes" for t in node.targets)
        )
        source = (BASE / "run_layerwise.py").read_text()
        for old, new in replacements.items():
            self.assertIn(old, source)
            source = source.replace(old, new)
        ast.parse(source)
        self.assertIn("test_product_overlap']==0", source)
        self.assertIn("test_source_hash_overlap']==0", source)
        self.assertIn("assert len(dataset)==20736", source)
        self.assertIn("choices=('train',),default='train'", source)

    def boundary(self):
        spec = importlib.util.spec_from_file_location(
            "t12_boundary_fixture", BASE / "physical_decoder_boundary.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self):
        import torch
        model = torch.nn.Module()
        model.editor = torch.nn.Module()
        for name in ("up3", "up2", "up1", "to_delta", "source_gate", "bottleneck"):
            setattr(model.editor, name, torch.nn.Linear(2, 2))
        model.language = torch.nn.Linear(2, 2)
        model.register_buffer("phase", torch.ones(2))
        return model

    def test_only_original_decoder_is_trainable(self):
        boundary = self.boundary()
        model = self.fixture()
        selected = boundary.freeze_for_adaptation(model)
        self.assertEqual(len(selected), 10)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, boundary.is_decoder(name))
        self.assertFalse(model.training)
        self.assertFalse(boundary.is_decoder("editor.bottleneck.optical.weight"))

    def test_decoder_update_does_not_change_protected_hash(self):
        import torch
        boundary = self.boundary()
        model = self.fixture()
        before = boundary.protected_hash(model)
        with torch.no_grad():
            model.editor.to_delta.weight.add_(1)
        self.assertEqual(before, boundary.protected_hash(model))

    def test_upstream_and_phase_update_detected(self):
        import torch
        boundary = self.boundary()
        model = self.fixture()
        before = boundary.protected_hash(model)
        with torch.no_grad():
            model.phase.add_(1)
        self.assertNotEqual(before, boundary.protected_hash(model))
        before = boundary.protected_hash(model)
        with torch.no_grad():
            model.editor.bottleneck.weight.add_(1)
        self.assertNotEqual(before, boundary.protected_hash(model))


if __name__ == "__main__":
    unittest.main()
