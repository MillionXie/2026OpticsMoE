import ast
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import audit_batch_semantics as audit


def test_no_backbone_forward_training_or_gpu_transfer():
    tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    methods={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {"to","cuda","backward","step","train","forward"}.intersection(methods)
    pretrained=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and n.func.attr=="from_pretrained"]
    assert all(isinstance(n.func.value,ast.Name) and n.func.value.id=="AutoConfig" for n in pretrained)


def test_inventory_explicitly_discriminates_batchnorm_running_stats():
    source=Path(audit.__file__).read_text(encoding="utf-8")
    assert "track_running_stats" in source and "running_mean_present" in source
    assert "_BatchNorm" in source and "model.eval().requires_grad_(False)" in source
