"""Dependency/contract tests without importing Torch or touching a GPU/video."""
from __future__ import annotations

import ast
import argparse
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[2]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    if {node.name for node in selected} != set(names):
        raise AssertionError("Missing migration dependency")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class MeasurementIdentityTests(unittest.TestCase):
    def setUp(self):
        self.cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "NVIDIA A100-PCIE-40GB")
        self.calls = []
        def run(command, **kwargs):
            self.calls.append((command, kwargs))
            return SimpleNamespace(stdout="250.0\n")
        self.ns = load_functions(
            ROOT / "LightGenV2/common/baseline_measurement.py",
            ["nvidia_smi_gpu_id", "gpu_power_limit_w", "validate_cuda_device"],
            {"os": os, "torch": SimpleNamespace(cuda=self.cuda), "subprocess": SimpleNamespace(run=run)},
        )

    def test_physical_gpu_mapping(self):
        for visible, expected in [("", "0"), ("5,2", "5"), (" GPU-test , 2", "GPU-test"), ("MIG-test", "MIG-test")]:
            with self.subTest(visible=visible), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}):
                self.assertEqual(self.ns["nvidia_smi_gpu_id"](), expected)

    def test_disabled_cuda_mapping_rejected(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            with self.assertRaises(RuntimeError):
                self.ns["nvidia_smi_gpu_id"]()

    def test_power_query_uses_visible_physical_gpu(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}):
            self.assertEqual(self.ns["gpu_power_limit_w"](), 250.0)
        self.assertIn("--id=5", self.calls[-1][0])
        self.assertIn("--query-gpu=power.limit", self.calls[-1][0])

    def test_explicit_power_query_identity(self):
        self.ns["gpu_power_limit_w"]("GPU-explicit")
        self.assertIn("--id=GPU-explicit", self.calls[-1][0])

    def test_invalid_power_output_rejected(self):
        self.ns["subprocess"] = SimpleNamespace(run=lambda *a, **k: SimpleNamespace(stdout="N/A\n"))
        with self.assertRaises(RuntimeError):
            self.ns["gpu_power_limit_w"]()

    def test_gpu_contract_enforced(self):
        self.assertEqual(self.ns["validate_cuda_device"]("a100"), "NVIDIA A100-PCIE-40GB")
        with self.assertRaises(RuntimeError):
            self.ns["validate_cuda_device"]("5090")
        self.cuda.is_available = lambda: False
        with self.assertRaises(RuntimeError):
            self.ns["validate_cuda_device"]()

    def test_sampler_keeps_default_and_accepts_uuid(self):
        namespace = load_functions(ROOT / "LightGenV2/common/baseline_measurement.py",
            ["NvidiaSmiPowerSampler"], {"threading": __import__("threading")})
        sampler = namespace["NvidiaSmiPowerSampler"]
        self.assertEqual(sampler().gpu_index, 0)
        self.assertEqual(sampler(gpu_index=5).gpu_index, 5)
        self.assertEqual(sampler(gpu_index="GPU-explicit").gpu_index, "GPU-explicit")
        with self.assertRaises(ValueError):
            sampler(interval_ms=51)


class DecodeAuditTests(unittest.TestCase):
    def decode(self, trace=None):
        frames = []
        class Frame:
            shape = (62, 100, 3)
            def __getitem__(self, item):
                frames.append((item[0].start, item[0].stop, item[1].start, item[1].stop))
                return self
        class Capture:
            def get(self, key):
                return {1: 100, 2: 25.0, 3: 100, 4: 62}[key]
            def set(self, key, value):
                self.position = value
            def read(self):
                return True, Frame()
            def release(self):
                pass
        cv = SimpleNamespace(VideoCapture=lambda path: Capture(), CAP_PROP_FRAME_COUNT=1, CAP_PROP_FPS=2,
            CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_POS_FRAMES=5, COLOR_BGR2RGB=6,
            INTER_AREA=7, cvtColor=lambda frame, mode: frame,
            resize=lambda frame, size, interpolation: (size, interpolation))
        namespace = load_functions(TASK / "quality_token_common.py", ["decode_random_seek"],
            {"cv2": cv, "math": __import__("math"), "Image": SimpleNamespace(fromarray=lambda frame: frame)})
        modules = {"transformers": SimpleNamespace(), "transformers.video_utils": SimpleNamespace(VideoMetadata=lambda **kw: kw)}
        source = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=123), __str__=lambda: "mock-video")
        with patch.dict(sys.modules, modules):
            result = namespace["decode_random_seek"](source, [0.10, 0.37, 0.63, 0.90], 448, audit_trace=trace)
        return result, frames

    def test_trace_does_not_change_decoding(self):
        before, before_crops = self.decode()
        trace = {}
        after, after_crops = self.decode(trace)
        self.assertEqual(before, after)
        self.assertEqual(before_crops, after_crops)
        self.assertEqual(before[2], [10, 37, 62, 89])
        self.assertEqual(before_crops, [(11, 51, 30, 70)] * 4)
        self.assertEqual(before[0], [((448, 448), 7)] * 4)
        self.assertEqual(trace["source_file_bytes"], 123)
        self.assertEqual(trace["container_reported_size_wh"], [100, 62])
        self.assertEqual(trace["selected_frame_positions"], [10, 37, 62, 89])


class BatchSourceContractTests(unittest.TestCase):
    def test_missing_dependencies_now_present(self):
        source = ast.parse((TASK / "quality_token_batch_benchmark.py").read_text(encoding="utf8"))
        common = ast.parse((TASK / "quality_token_common.py").read_text(encoding="utf8"))
        exports = {node.name for node in common.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        uses = {node.attr for node in ast.walk(source) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "core"}
        constants = {target.id for node in common.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
        self.assertFalse(uses - exports - constants)

    def test_output_guard_precedes_gpu_validation(self):
        source = ast.parse((TASK / "quality_token_batch_benchmark.py").read_text(encoding="utf8"))
        main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        guard = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name) and node.exc.func.id == "FileExistsError")
        device = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "validate_cuda_device")
        self.assertLess(guard, device)

    def test_existing_batch_output_rejected_without_gpu(self):
        namespace = load_functions(TASK / "quality_token_batch_benchmark.py", ["main"],
            {"argparse": argparse, "Path": Path,
             "validate_cuda_device": lambda *a: self.fail("GPU queried before output guard")})
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "formal/batch_16").mkdir(parents=True)
            argv = ["benchmark", "formal", "--model", "unused", "--manifest", "unused", "--checkpoint", "unused", "--output", directory]
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                namespace["main"]()

    def test_existing_dataset_once_output_rejected_without_gpu(self):
        namespace = load_functions(TASK / "quality_token_dataset_once.py", ["main"],
            {"argparse": argparse, "Path": Path, "SCHEME1": "scheme1_scalar_linear",
             "SCHEME2": "scheme2_five_quality_tokens", "core": SimpleNamespace(PROMPTS={"temporal": "prompt"}),
             "validate_cuda_device": lambda *a: self.fail("GPU queried before output guard")})
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "resolution448/frames4/scheme2_five_quality_tokens").mkdir(parents=True)
            argv = ["dataset", "--frames", "4", "--scheme", "scheme2_five_quality_tokens", "--image-size", "448", "--model", "unused", "--manifest", "unused", "--checkpoint", "unused", "--output", directory]
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                namespace["main"]()


if __name__ == "__main__":
    unittest.main()
