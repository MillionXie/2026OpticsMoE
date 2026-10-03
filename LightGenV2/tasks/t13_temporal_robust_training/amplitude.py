"""Six-plane pre-modulation bounded amplitude, matching T12 tanh/0.5.

The immutable teacher file is never edited. Six audited methods are compiled
from its hash-pinned AST with exactly one pre-complex amplitude insertion each.
No encoding is applied to a field that already contains coherent DC modulation.
"""
import ast
from contextlib import contextmanager
from pathlib import Path

CONTRACT = {"kind": "tanh", "scale": 0.5, "bmp_scale": 255,
            "per_frame_peak_normalization": False, "position": "before_phase_and_coherent_DC"}
METHODS = {"FrameOpticalRouter": ("forward",), "FrameOpticalPath": ("expert", "global_path"),
           "VideoOpticalRouter": ("forward",), "VideoOpticalPath": ("expert", "global_path")}


def encode(value):
    import torch
    if value.is_complex() or not torch.isfinite(value).all() or bool((value < 0).any()):
        raise ValueError("Pre-modulation amplitude must be finite, real and nonnegative")
    return torch.tanh(value / CONTRACT["scale"])


def quantize_amplitude(value):
    import torch
    if not torch.isfinite(value).all() or bool((value < 0).any()) or bool((value > 1.000001).any()):
        raise ValueError("Physical amplitude must already be bounded; no implicit normalization")
    return (value.clamp(0, 1) * 255).round().to(torch.uint8)


@contextmanager
def bounded_graph():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))
    from verify_source import verify
    from LightGenV2.tasks.t06_video_quality_assessment.models import multivideo9x4 as optics
    verify()
    source = Path(optics.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    originals = []
    replacements = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in METHODS:
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name not in METHODS[cls.name]:
                continue
            class Transform(ast.NodeTransformer):
                count = 0
                def visit_Call(self, node):
                    self.generic_visit(node)
                    if (isinstance(node.func, ast.Attribute) and node.func.attr == "to"
                            and len(node.args) == 1 and isinstance(node.args[0], ast.Attribute)
                            and isinstance(node.args[0].value, ast.Name) and node.args[0].value.id == "torch"
                            and node.args[0].attr == "complex64"):
                        node.func.value = ast.Call(func=ast.Name(id="_bounded_encode", ctx=ast.Load()),
                                                   args=[node.func.value], keywords=[])
                        self.count += 1
                    return node
            transform = Transform()
            method = transform.visit(method)
            if transform.count != 1:
                raise RuntimeError(f"Expected one pre-modulation insertion: {cls.name}.{method.name}")
            namespace = dict(vars(optics), _bounded_encode=encode)
            unit = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
            exec(compile(unit, str(Path(__file__)), "exec"), namespace)
            owner = getattr(optics, cls.name)
            originals.append((owner, method.name, getattr(owner, method.name)))
            replacements.append((owner, method.name, namespace[method.name]))
    if len(originals) != 6:
        raise RuntimeError("Incomplete six-plane amplitude graph")
    try:
        for owner, name, replacement in replacements:
            setattr(owner, name, replacement)
        yield CONTRACT
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
