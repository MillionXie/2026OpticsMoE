import unittest
from maintenance.git_safety.check_tracked_imports import dependencies


class ImportAuditTests(unittest.TestCase):
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
