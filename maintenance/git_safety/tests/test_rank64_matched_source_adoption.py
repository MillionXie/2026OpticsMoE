"""Source-preserving adoption checks; never import Torch or run experiments."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "LightGenV2/tasks/t04_openmoji_robust_ablation/train_rank64_matched.py"


def test_original_runner_preserved_except_old_checkout_injection():
    text = SOURCE.read_text(encoding="utf-8")
    restored = text.replace("import json\n", "import json\nimport sys\n", 1)
    restored = restored.replace(
        "from LightGenV2.tasks.t04_openmoji_robust_ablation.train import",
        "SOURCE_ROOT = Path('/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t04_openmoji_robust_20260928')\n"
        "sys.path.insert(0, str(SOURCE_ROOT))\n"
        "from LightGenV2.tasks.t04_openmoji_robust_ablation.train import", 1,
    )
    assert hashlib.sha256(restored.encode()).hexdigest() == (
        "f8e0786a90a1b45a911020251dfddffa34436aa1890a3ba622c2a7b44d23eef1"
    )


def test_runner_has_no_old_checkout_override_or_top_level_execution():
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert "sys.path" not in text and ".worktrees/" not in text
    assert isinstance(tree.body[-1], ast.If)
    assert ast.unparse(tree.body[-1].test) == "__name__ == '__main__'"
    assert "'selection': 'clean TEST every5 epochs development'" in text
    assert "assert len(train.dataset) == 5000 and len(test.dataset) == 1000" in text
    assert "cfg.shared_readout_variant, cfg.output_dir, cfg.num_workers = 'lowrank64', output, 0" in text


def test_runner_is_registered_and_bound_to_adopted_source():
    registry = json.loads((ROOT / "LightGenV2/TASK_REGISTRY.json").read_text(encoding="utf-8"))
    task = next(t for t in registry["tasks"] if t["id"] == "t04_openmoji_robust_ablation")
    assert ROOT / "LightGenV2" / task["matched_rank64_simulation_entry"] == SOURCE
    manifest = json.loads((ROOT / "LightGenV2" / task["matched_rank64_source_import"]).read_text(encoding="utf-8"))
    row = next(r for r in manifest["files"] if r["path"] == SOURCE.relative_to(ROOT).as_posix())
    assert row["hash_normalization"] == "crlf_to_lf"
    assert hashlib.sha256(SOURCE.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == row["sha256"]
