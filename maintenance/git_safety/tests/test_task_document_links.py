import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('links',Path(__file__).resolve().parents[1]/'check_task_document_links.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class LinkTests(unittest.TestCase):
    def test_source_missing(self):
        self.assertEqual(m.classify('a/README.md','missing.py',set())['kind'],'missing_source_or_document')
    def test_preserved_not_source(self):
        self.assertEqual(m.classify('a/README.md','runs/x/report.json',set())['kind'],'external_or_preserved_artifact')
    def test_relative_and_encoded(self):
        self.assertIsNone(m.classify('a/b/README.md','../my%20report.md#x',{'a/my report.md'}))
    def test_external_and_anchor(self):
        self.assertIsNone(m.classify('a.md','https://example.org',set()))
        self.assertIsNone(m.classify('a.md','#head',set()))
    def test_private(self):
        self.assertEqual(m.classify('LightGenV2/tasks/t/README.md','../../../handoffs/x.md',set())['kind'],'private_handoff')
if __name__=='__main__':unittest.main()
