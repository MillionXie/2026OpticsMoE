"""Ignore rules are visibility controls, never deletion or cleanup approval."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ArtifactIgnoreTests(unittest.TestCase):
    def ignored(self, paths):
        result = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '--no-index', '-z', '--stdin'],
                                input=('\0'.join(paths)+'\0').encode(), capture_output=True)
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return set(filter(None, result.stdout.decode().split('\0')))

    def test_private_arrays_and_generated_plates(self):
        paths = ['sample/cache.npy', 'sample/cache.npz', 'sample/model.safetensors',
                 'outputs/comparison/plate.png', 'LightGenV2/demo_check/package/ccd.bmp']
        self.assertEqual(self.ignored(paths), set(paths))

    def test_source_metrics_manifests_and_timing_not_hidden(self):
        paths = ['outputs/comparison/eval.py', 'outputs/comparison/README.md',
                 'outputs/comparison/per_image.csv', 'outputs/comparison/timing.json',
                 'LightGenV2/demo_check/package/config.yaml', 'maintenance/storage/receipt.json',
                 'LightGenV2/tasks/t12_text_to_image/perceptual/eval_regions.py']
        self.assertEqual(self.ignored(paths), set())


if __name__ == '__main__':
    unittest.main()
