"""Ignore rules are visibility controls, never deletion or cleanup approval."""
from pathlib import Path
import subprocess
import shutil
import tempfile
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ArtifactIgnoreTests(unittest.TestCase):
    def test_private_machine_inventory_not_public_identity(self):
        private = ['maintenance/storage/20261002_server_snapshots.json',
                   'maintenance/storage/20261003_server_status.json']
        public = ['maintenance/storage/HISTORICAL_BASELINE_PAYLOAD_VISIBILITY_20261006.json',
                  'maintenance/storage/T02_PROTOCOL_ENTRY_IMPORT_20261003.json',
                  'LightGenV2/tasks/t07_abo_image_retrieval/reports/report.json',
                  'maintenance/storage/20261006_future_source_review.json']
        self.assertEqual(self.ignored(private + public), set(private))

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
        images += ['LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/train/train_000001/scene.json',
                   'LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/test/test_000001/scene.json',
                   'LightGenV2/tasks/t07_abo_image_retrieval/reports/bringup/ccd.png',
                   'LightGenV2/reports/timing/latency_plot.png']
        images += ['handoffs/t12_small_baseline_handoff_20260928/stage/model_release/source/LightGenV2/tasks/t12_text_to_image/sealed_editor.py']
        images += ['ABO_Lab_8um/original_a100/backend/experiments/model.py',
                   'ABO_Lab_8um/original_optics/backend/experiments/model.py']
        images += ['LightGenV2/demo_check/EuroSAT_MoE_D2NN/code/experiments/vision_transfer/model.py']
        images += ['LightGenV2/reports/20260915_baseline_methods/code_packages/'+name+'/source/model.py'
                   for name in ('01_lgvq','02_abo_image_text','03_abo_image_image','04_lsp','05_salicon','06_openmoji')]
        self.assertEqual(self.ignored(images), set(images))
        sources = ['LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/prepare.py',
                   'LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/manifest.json',
                   'LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/config.yaml',
                   'ABO_Lab_8um/original_a100/source/model.py',
                   'ABO_Lab_8um/original_a100/assets/test_dataset/manifest.json',
                   'ABO_Lab_8um/original_a100/reference_phases/manifest.json',
                   'ABO_Lab_8um/original_a100/reports/timing.csv']
        sources += ['LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/dataset_summary.json',
                    'LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_grid_v2/train/train_000001/config.yaml',
                    'LightGenV2/reports/timing/timing_per_sample.csv',
                    'LightGenV2/reports/timing/source_snapshot/measure.py',
                    'LightGenV2/tasks/t07_abo_image_retrieval/reports/bringup/manifest.json']
        sources += ['handoffs/t12_small_baseline_handoff_20260928/stage/materialize_pairs.py',
                    'LightGenV2/reports/20260915_baseline_methods/code_packages/01_lgvq/run_baseline.py',
                    'LightGenV2/reports/20260915_baseline_methods/code_packages/01_lgvq/SOURCE_MANIFEST.json',
                    'LightGenV2/reports/20260915_baseline_methods/code_packages/01_lgvq/timing.csv',
                    'LightGenV2/common/baseline_measurement.py',
                    'LightGenV2/demo_check/EuroSAT_MoE_D2NN/code/train_eurosat.py',
                    'LightGenV2/demo_check/EuroSAT_MoE_D2NN/code/download_archives.py',
                    'LightGenV2/demo_check/EuroSAT_MoE_D2NN/FILES_SHA256.json',
                    'ABO_Lab_8um/original_a100/BACKEND_MANIFEST.json',
                    'ABO_Lab_8um/original_optics/SOFTWARE_MANIFEST.json',
                    'handoffs/t12_small_baseline_handoff_20260928/stage/model_release/manifest.json',
                    'LightGenV2/tasks/t12_text_to_image/sealed_editor.py']
        self.assertEqual(self.ignored(sources), set())


if __name__ == '__main__':
    unittest.main()
