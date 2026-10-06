"""Ignore rules are visibility controls, never deletion or cleanup approval."""
from pathlib import Path
import subprocess
import shutil
import tempfile
from unittest.mock import patch
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

    def test_audited_original_dataset_images_not_source(self):
        # Isolate the published rules from machine-local info/exclude settings.
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder)
            subprocess.run(['git', 'init', '-q', str(fixture)], check=True)
            shutil.copyfile(ROOT / '.gitignore', fixture / '.gitignore')
            with patch(__name__ + '.ROOT', fixture):
                self.check_audited_dataset_rules()

    def check_audited_dataset_rules(self):
        images = ['LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/train/input_00001.png',
                  'ABO_Lab_8um/original_a100/assets/test_dataset/images/product/image.jpg',
                  'ABO_Lab_8um/original_inference/assets/test_dataset/images/product/image.jpeg',
                  'ABO_Lab_8um/original_optics/reference_phases/vision_router.bmp']
        self.assertEqual(self.ignored(images), set(images))
        sources = ['LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/prepare.py',
                   'LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/manifest.json',
                   'LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/config.yaml',
                   'ABO_Lab_8um/original_a100/source/model.py',
                   'ABO_Lab_8um/original_a100/assets/test_dataset/manifest.json',
                   'ABO_Lab_8um/original_a100/reference_phases/manifest.json',
                   'ABO_Lab_8um/original_a100/reports/timing.csv']
        self.assertEqual(self.ignored(sources), set())


if __name__ == '__main__':
    unittest.main()
