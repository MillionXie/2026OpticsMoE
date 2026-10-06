import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("review_git", Path(__file__).with_name("review_git.py"))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ArtifactChecks(unittest.TestCase):
    def test_known_machine_configs_not_example_templates(self):
        for path in ('task/LAB.local.json', 'task/dual.local.json', 'task/paths.local.yaml'):
            self.assertIsNotNone(review.forbidden_artifact(path, 10))
        for path in ('task/LAB.example.json', 'task/dual.example.json', 'task/paths.example.yaml', 'task/config.yaml'):
            self.assertIsNone(review.forbidden_artifact(path, 10))
    def test_literal_connection_credentials_are_redacted(self):
        raw = b"client.connect('example.invalid', password='fixture-value')"
        issues = review.credential_issues('helper.py', raw)
        self.assertEqual(issues[0]['line'], 1)
        self.assertNotIn('fixture-value', str(issues))

    def test_private_configuration_connection_is_allowed(self):
        self.assertEqual(review.credential_issues('helper.py', b"client.connect(host, password=private_config.password)"), [])
        self.assertEqual(review.credential_issues('helper.py', b"client.connect(host, password=None)"), [])

    def test_private_key_header_is_redacted(self):
        raw = b'-----BEGIN ' + b'OPENSSH PRIVATE KEY-----\nfixture\n'
        issues = review.credential_issues('key.txt', raw)
        self.assertTrue(issues)
        self.assertNotIn('fixture', str(issues))

    def test_normal_source(self):
        self.assertIsNone(review.forbidden_artifact("LightGenV2/tasks/t04/models/model.py", 8000))

    def test_weights(self):
        self.assertIsNotNone(review.forbidden_artifact("assets/best.pt", 100))
        self.assertIsNotNone(review.forbidden_artifact("release.tar.gz", 100))

    def test_private_and_runtime(self):
        for path in [".codex_tmp/helper.py", "task/runs/a/metrics.json", "data/list.txt",
                     ".worktrees/a/model.py", "task\\rejected_dark\\frame.png"]:
            self.assertIsNotNone(review.forbidden_artifact(path, 100))

    def test_large_file(self):
        self.assertIsNotNone(review.forbidden_artifact("reports/output.bin", 10 * 1024 * 1024 + 1))


if __name__ == "__main__":
    unittest.main()
