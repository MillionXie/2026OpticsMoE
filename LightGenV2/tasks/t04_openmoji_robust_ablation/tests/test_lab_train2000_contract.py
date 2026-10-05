"""Static and pure-identity checks; no model, dataset or SDK execution."""
import ast
import importlib.util
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("identity", TASK / "lab_checkpoint_identity.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)
tune = ast.parse((TASK / "lab_tune2000.py").read_text(encoding="utf-8"))
calibration = ast.parse((TASK / "lab_calibrate_rank64.py").read_text(encoding="utf-8"))


@pytest.mark.parametrize("group", ["g2", "g5"])
def test_group_uses_audited_rank64_profile(group):
    function = next(n for n in tune.body if isinstance(n, ast.FunctionDef) and n.name == "group_checkpoint")
    def selected(name, path):
        return identity.load_identity(path, name)
    namespace = {"Path": Path, "__file__": str(TASK / "lab_tune2000.py"), "checkpoint_identity": selected}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "group_selection", "exec"), namespace)
    row = identity.load_identity(TASK / "configs/lab/rank64_20261002.json", group)
    assert namespace["group_checkpoint"](group) == (row["filename"], row["sha256"])


def test_original_training_scope_and_selection_retained():
    text = ast.unparse(tune)
    assert "fit_idx = list(range(2000))" in text
    assert "test_metrics = evaluate(test, list(range(1000))) if epoch % 5 == 0 else None" in text
    assert "score = test_metrics['overall']['changed_cell_accuracy'] if test_metrics else None" in text
    assert "if score is not None and score > best_score:" in text
    assert "test_gradient" in text and "former_validation_in_train" in text
    loop = next(n for n in ast.walk(tune) if isinstance(n, ast.For) and ast.unparse(n.target) == "epoch")
    batch_loop = next(n for n in loop.body if isinstance(n, ast.For))
    assert "mini(train, indices)" in ast.unparse(batch_loop)
    assert "mini(test," not in ast.unparse(batch_loop)
    assert "loss.backward()" in ast.unparse(batch_loop)


def test_original_loss_and_physical_contract_retained():
    text = ast.unparse(tune)
    assert "loss = 0.5 * changed.mean() + 0.5 * composed.mean() + 0.2 * preserved.mean()" in text
    assert "F.binary_cross_entropy_with_logits(edit, mask, pos_weight=edit.new_tensor(8.0)) + 0.05 * anchor" in text
    assert "contract['exposure_us'] == 2000" in text
    assert "contract['gain'] == 'Gain_X4'" in text
    assert "contract['wait_ms'] == 240" in text
    assert "len(train['ids']) == len(set(train['ids'])) == 2000" in text
    assert "assert protected_sha(model) == upstream_sha" in text


def test_calibration_changes_existing_bias_and_strict_reloads():
    text = ast.unparse(calibration)
    assert "from .lab_tune2000 import config, protected_sha" in text
    assert "sys.path.insert" not in text
    assert "config(args.project, torch.device('cpu'), args.group)" in text
    assert "selected['model']['shared_readout.decoder.edit_head.bias'] -= best['offset']" in text
    assert "strict=True" in text
    assert "assert protected_sha(model) == original_protected" in text
    assert "threshold_equivalent" in text and "test_gradient" in text
