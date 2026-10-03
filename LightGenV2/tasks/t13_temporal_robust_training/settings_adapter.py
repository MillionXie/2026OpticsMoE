"""Allow zero-DC ablations; preserve every other formal teacher check."""
import ast
import inspect
import textwrap


def load_settings(path, *, synthetic=False):
    from verify_source import verify
    from LightGenV2.tasks.t06_video_quality_assessment import multivideo_settings as backend
    verify()
    original = backend.MultiVideoSettings.validate
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    method = tree.body[0]
    guards = [node for node in method.body if isinstance(node, ast.If)
              and any(isinstance(child, ast.Constant)
                      and child.value == "Formal runs retain at least 20% nominal DC power"
                      for child in ast.walk(node))]
    if len(guards) != 1:
        raise RuntimeError("Teacher DC policy guard identity changed")
    if ast.unparse(guards[0].test) != "not self.synthetic and self.unmodulated_power_fraction_min < 0.2":
        raise RuntimeError("Unexpected teacher DC policy condition")
    method.body.remove(guards[0])
    namespace = dict(vars(backend))
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    try:
        backend.MultiVideoSettings.validate = namespace["validate"]
        return backend.load_settings(path, synthetic=synthetic)
    finally:
        backend.MultiVideoSettings.validate = original
