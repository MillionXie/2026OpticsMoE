from types import SimpleNamespace
from pathlib import Path
import unittest
from LightGenV2.tasks.t06_video_quality_assessment.verify_formal_checkpoint import inspect_checkpoint


class FormalEntryTests(unittest.TestCase):
    def test_fixed_entry_requests_cpu_only(self):
        calls = []
        def loader(target, path, device):
            calls.append((target, path, device))
            return SimpleNamespace(parameters=lambda: []), SimpleNamespace(architecture_label='fixture')
        report = inspect_checkpoint('temporal', 'fixture.pt', loader)
        self.assertEqual(calls, [('temporal', Path('fixture.pt'), 'cpu')])
        self.assertEqual(report['checkpoint_sha256'], '5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c')
        self.assertTrue(report['read_only'])

    def test_legacy_profile_not_silently_substituted(self):
        with self.assertRaises(ValueError):
            inspect_checkpoint('temporal36_balanced', 'fixture.pt')


if __name__ == '__main__':
    unittest.main()
