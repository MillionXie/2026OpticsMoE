import unittest
from maintenance.git_safety.check_tracked_imports import dependencies, module_source_paths


class ImportAuditTests(unittest.TestCase):
    def test_transitive_targets_include_module_alias_and_initializers(self):
        tracked={'LightGenV2/__init__.py','LightGenV2/common/__init__.py','LightGenV2/common/model.py'}
        rows=dependencies('LightGenV2/tasks/example/run.py','from LightGenV2.common import model',tracked,include_present=True)
        self.assertEqual({r['module'] for r in rows},{'LightGenV2.common','LightGenV2.common.model'})
        self.assertEqual(module_source_paths('LightGenV2.common.model',tracked),
                         ['LightGenV2/__init__.py','LightGenV2/common/__init__.py','LightGenV2/common/model.py'])

    def test_transitive_relative_module_does_not_append_imported_class(self):
        rows=dependencies('LightGenV2/tasks/example/run.py','from .model import Model',
                          {'LightGenV2/tasks/example/model.py'},include_present=True)
        self.assertEqual([r['module'] for r in rows],['LightGenV2.tasks.example.model'])
    def test_relative_dependency_missing_from_git(self):
        rows=dependencies('LightGenV2/tasks/example/run.py','from .model import Model',set())
        self.assertEqual(rows[0]['module'],'LightGenV2.tasks.example.model')

    def test_tracked_dependency_and_external_import_accepted(self):
        rows=dependencies('LightGenV2/tasks/example/run.py','from .model import Model\nimport torch',
                          {'LightGenV2/tasks/example/model.py'})
        self.assertEqual(rows,[])

    def test_parent_relative_resolution(self):
        rows=dependencies('LightGenV2/tasks/example/tests/check.py','from ..model import Model',set())
        self.assertEqual(rows[0]['module'],'LightGenV2.tasks.example.model')

    def test_package_attribute_ambiguity_is_explicit(self):
        rows=dependencies('LightGenV2/tasks/example/run.py','from . import value',set())
        self.assertTrue(rows[0]['alias_may_be_package_attribute'])
