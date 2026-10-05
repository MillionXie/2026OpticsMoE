"""Configuration imports must not initialize the model runtime."""
import subprocess
import sys
import unittest


class LazyImportTests(unittest.TestCase):
    def run_isolated(self, source):
        subprocess.run([sys.executable, '-c', source], check=True)

    def test_settings_import_does_not_load_torch(self):
        self.run_isolated('''
import sys
from LightGenV2.tasks.t12_text_to_image.settings import Settings
assert 'torch' not in sys.modules
assert 'LightGenV2.tasks.t12_text_to_image.modeling' not in sys.modules
''')

    def test_public_exports_keep_identity(self):
        self.run_isolated('''
import sys, types
import LightGenV2.tasks.t12_text_to_image as task
model = types.ModuleType(task.__name__ + '.modeling')
model.TextConditionedVAE = type('TextConditionedVAE', (), {})
model.build_model = lambda: None
sys.modules[model.__name__] = model
assert task.TextConditionedVAE is model.TextConditionedVAE
assert task.build_model is model.build_model
try:
    task.undefined_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export must fail')
''')


if __name__ == '__main__':
    unittest.main()
