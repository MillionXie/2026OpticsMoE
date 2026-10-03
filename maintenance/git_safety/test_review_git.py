import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("review_git", Path(__file__).with_name("review_git.py"))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ArtifactChecks(unittest.TestCase):
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
