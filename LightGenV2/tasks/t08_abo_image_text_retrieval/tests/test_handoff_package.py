import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from LightGenV2.tasks.t08_abo_image_text_retrieval.build_lab_package import source_paths
from LightGenV2.tasks.t08_abo_image_text_retrieval.handoff import verify_package


class HandoffTests(unittest.TestCase):
    def test_legacy_dependencies_are_present(self):
        root = Path(__file__).resolve().parents[4]
        files = {p.as_posix() for p in source_paths(root)}
        for suffix in ("balanced_optical_fusion_ablation", "electronic_retrieval",
                       "four_layer_optical_retrieval_10cm_robust",
                       "four_layer_optical_retrieval_10cm_warmstart5",
                       "four_layer_optical_router_retrieval", "robust_hybrid_retrieval"):
            prefix = f"experiments/qwen3_vl_embedding_2b_caltech101_{suffix}/"
            self.assertTrue(any(p.startswith(prefix) and p.endswith(".py") for p in files), prefix)
        self.assertFalse(any("/runs/" in p or "/vendor_sdk/" in p for p in files))

    def test_manifest_detects_tampering_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "payload.txt"
            path.write_bytes(b"example")
            manifest = {"source_commit": "unit-test", "files": [{"path": "payload.txt",
                         "sha256": hashlib.sha256(b"example").hexdigest()}]}
            (root / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_package(root)["verified_files"], 1)
            path.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                verify_package(root)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_package(root)


if __name__ == "__main__":
    unittest.main()
