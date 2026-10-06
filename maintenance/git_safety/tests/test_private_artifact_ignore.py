"""Ignore rules are visibility controls, never deletion or cleanup approval."""
from pathlib import Path
import subprocess
import shutil
import tempfile
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ArtifactIgnoreTests(unittest.TestCase):
    def test_review_package_copies_not_primary_source_or_changed_descriptors(self):
        import json
        d=json.loads((ROOT/'maintenance/storage/T06_REVIEW_SOURCE_VISIBILITY_20261007.json').read_text(encoding='utf8'))
        selected=[r['path'] for r in d['files']]
        self.assertEqual(len(selected),80)
        visible=d['excluded_unmatched']+[
            'LightGenPublic/tasks/t06_lgvq_temporal_consistency/train.py',
            'LightGenPublic/tasks/t06_lgvq_temporal_consistency/runtime/lgvq_temporal/fixed_weight.py',
            'LightGenPublic/tasks/t06_lgvq_temporal_consistency/teacher_release_final/lgvq_temporal_08044/new_fix.py',
            'LightGenPublic/tasks/t07_abo_image_retrieval/lightgen_abo/model.py']
        self.assertEqual(self.ignored(selected+visible),set(selected))

    def test_abo_handoff_records_keep_contracts_code_and_future_results_visible(self):
        import json
        d=json.loads((ROOT/'maintenance/storage/ABO_HANDOFF_RESULT_VISIBILITY_20261007.json').read_text(encoding='utf8'))
        selected=[r['path'] for r in d['files']]
        self.assertEqual(len(selected),74)
        visible=['handoffs/abo_latestfresh35_lab_20260930/RANK72_DEPLOYMENT_20260930.md',
                 'handoffs/abo_latestfresh35_lab_20260930/new_physical_report.json',
                 'handoffs/abo_i2i_sixhour_strong_20260928/capture_contract.json',
                 'handoffs/abo_i2i_sixhour_strong_20260928/selection.json',
                 'handoffs/abo_i2i_sixhour_strong_20260928/new_eval.py',
                 'handoffs/openmoji_robust_ablation_20260928/new_report.json']
        self.assertEqual(self.ignored(selected+visible),set(selected))

    def test_kather_transfer_evidence_not_tables_manifests_or_future_files(self):
        import json
        d=json.loads((ROOT/'maintenance/storage/KATHER_SCAN_EVIDENCE_VISIBILITY_20261007.json').read_text(encoding='utf8'))
        private=[row['path'] for row in d['files']]
        visible=[row['path'] for row in d['manifests']]
        visible += [d['root']+'/'+name for name in ('README.md','full_test_per_seed.csv',
                    'full_test_scan_lock.json','evidence/future_run/test_result.json','analyze.py')]
        self.assertEqual(len(private),20)
        self.assertEqual(self.ignored(private+visible),set(private))

    def test_large_result_records_not_source_new_results_or_manual_assets(self):
        import json
        descriptor=json.loads((ROOT/'maintenance/storage/LARGE_RESULT_PAYLOAD_VISIBILITY_20261007.json').read_text(encoding='utf8'))
        private=[row['path'] for row in descriptor['files']]
        visible=[row['path'] for row in descriptor['manual_review_files']]
        visible += ['handoffs/t12_lab_robust17m_20260927/future_metrics.json',
                    'LightGenV2/demo_check/EuroSAT_MoE_D2NN/code/train_eurosat.py',
                    'LightGenV2/reports/20260927_demo_energy_efficiency_a100/future_timing.csv']
        self.assertEqual(len(private),20)
        self.assertEqual(self.ignored(private+visible),set(private))

    def test_rank72_delivery_copy_not_unmatched_or_canonical_source(self):
        import json
        descriptor = json.loads((ROOT/'maintenance/storage/RANK72_DELIVERY_SOURCE_VISIBILITY_20261007.json').read_text())
        private = [row['path'] for row in descriptor['files']]
        visible = descriptor['excluded'] + [row['canonical'] for row in descriptor['files']]
        visible += ['handoffs/abo_latestfresh35_lab_20260930/rank72_source/standalone/future.py']
        self.assertEqual(len(private), 14)
        self.assertEqual(self.ignored(private + visible), set(private))

    def test_fixed478_records_keep_future_results_and_table_sources_visible(self):
        import json
        descriptor = json.loads((ROOT/'maintenance/storage/T10_FIXED478_REPORT_VISIBILITY_20261007.json').read_text())
        private = [row['path'] for row in descriptor['files']]
        self.assertEqual(len(private), 36)
        prefix = 'LightGenV2/tasks/t10_expert_scaling/reports/plotting/fixed478_topk_test_20260922/'
        visible = [prefix + name for name in ('README.md', 'test_per_seed.csv', 'test_summary.json',
                   'evidence/future_test_result.json', 'evidence/analyze.py')]
        self.assertEqual(self.ignored(private + visible), set(private))

    def test_lgvq_timing_payloads_keep_derivation_and_future_evidence_visible(self):
        import json
        descriptor = json.loads((ROOT/'maintenance/storage/LGVQ_TEMPORAL_TIMING_VISIBILITY_20261006.json').read_text())
        private = [row['path'] for row in descriptor['files']]
        self.assertEqual(len(private), 74)
        prefix = 'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/'
        visible = [prefix + name for name in ('README.md', 'build_analysis.py', 'protocol.json',
                   'SHA256SUMS.csv', 'calculated_summary.json', 'raw_remote/future/report.json',
                   'raw_remote/future/timing_per_call.csv', 'raw_remote/new_benchmark.py')]
        self.assertEqual(self.ignored(private + visible), set(private))

    def test_server_formal_timing_copy_is_exactly_named_not_folder_hidden(self):
        prefix = 'LightGenV2/reports/20260917_redbox_timing_energy_audit/server_2026OpticsMoE_a100_formal/'
        private = [prefix + '01_LGVQ_temporal/Ours/all_per_call_timings.csv',
                   prefix + '07_OpenMoji/Baseline/redbox_source_report.json']
        visible = [prefix + 'README_CN.md', prefix + 'RAW_DATA_STATUS.md',
                   prefix + 'SHA256SUMS.txt', prefix + 'future_run/report.json',
                   prefix + '07_OpenMoji/Baseline/future_measurement.csv',
                   prefix + 'measure.py']
        self.assertEqual(self.ignored(private + visible), set(private))

    def test_named_demo_source_export_not_main_or_future_source(self):
        prefix = 'LightGenV2/reports/20260927_demo_energy_efficiency_a100/ours/01_lgvq/source_snapshot/'
        private = [prefix + 'tasks/t06_video_quality_assessment/run.py']
        visible = [prefix + 'future_source.py', 'LightGenV2/tasks/t06_video_quality_assessment/run.py',
                   prefix + 'timing.csv', prefix + 'report.json']
        self.assertEqual(self.ignored(private + visible), set(private))

    def test_named_mnist_assets_keep_sources_timing_and_future_files_visible(self):
        prefix = 'MNIST_10cm_8um_Bench_Test_20260923/'
        private = [prefix + '02_mnist_10cm/phase/B_RECOMMENDED_native8_best.bmp']
        visible = [prefix + '02_mnist_10cm/phase/future.bmp',
                   prefix + 'measure.py', prefix + 'timing.csv', prefix + 'report.json']
        self.assertEqual(self.ignored(private + visible), set(private))

    def test_t12_share_only_reviewed_source_export_is_private(self):
        prefix = 'handoffs/t12_small_baseline_share_20260928/package/'
        private = [prefix + 'source/LightGenV2/common/__init__.py']
        visible = [prefix + 'export_pairs.py', prefix + 'infer_ours.py',
                   prefix + 'source/LightGenV2/tasks/t12_text_to_image/README.md',
                   prefix + 'source/LightGenV2/tasks/t12_text_to_image/reports/reproduction/README.md',
                   prefix + 'source/LightGenV2/tasks/t12_text_to_image/future_experiment.py',
                   'LightGenV2/tasks/t12_text_to_image/sealed_editor.py']
        self.assertEqual(self.ignored(private + visible), set(private))
    def test_demo_payload_ignore_keeps_source_and_future_evidence_visible(self):
        prefix = 'LightGenV2/reports/20260927_demo_energy_efficiency_a100/'
        private = [prefix + 'raw_remote/ours/narrow_200_no_warmup/all_per_call_timings.csv']
        visible = [prefix + 'build_summary.py', prefix + 'calculated_summary.json',
                   prefix + 'ours/01_lgvq/source_snapshot/tasks/t06_video_quality_assessment/configs/spatial_hardware_readout_tuning.json',
                   prefix + 'raw_remote/future_run/report.json']
        self.assertEqual(self.ignored(private + visible), set(private))
    def test_named_historical_timing_payload_not_code_or_future_evidence(self):
        private = ['LightGenV2/reports/20260917_redbox_timing_energy_audit/evidence/ours_narrow_clean_final/report.json']
        visible = ['LightGenV2/reports/20260917_redbox_timing_energy_audit/evidence/ours_narrow_clean_final/consolidate_narrow_optical_power.py',
                   'LightGenV2/reports/20260917_redbox_timing_energy_audit/evidence/future_run/report.json']
        self.assertEqual(self.ignored(private + visible), set(private))
    def test_known_machine_config_names_only(self):
        private = ['future_task/LAB.local.json', 'future_task/dual.local.json', 'future_task/paths.local.yaml']
        public = ['future_task/LAB.example.json', 'future_task/dual.example.json', 'future_task/config.yaml']
        self.assertEqual(self.ignored(private + public), set(private))
    def test_dataset_split_payload_not_timing_or_predictions(self):
        private = ['handoffs/t12_small_baseline_handoff_20260928/stage/datasets/abo_cleanrender_lamp_table_pillow_256_v1/' + split + '.jsonl' for split in ('train', 'val', 'test')]
        public = ['LightGenV2/reports/timing/sample_preprocessing.jsonl',
                  'LightGenV2/reports/baseline_plotting_20260922/_server_raw/openmoji/test_predictions.jsonl',
                  'handoffs/future_dataset/train.jsonl']
        self.assertEqual(self.ignored(private + public), set(private))

    def test_only_named_private_launchers_hidden(self):
        private = ['handoffs/abo_latestfresh35_lab_20260930/run_rank72_full.cmd',
                   'handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/run_rank64_queue_20261002.cmd']
        visible = ['handoffs/abo_latestfresh35_lab_20260930/future_launcher.cmd',
                   'handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/train_rank64_common_20261002.py',
                   'handoffs/t12_channel_robust_20260927/run_original_val96.cmd']
        self.assertEqual(self.ignored(private + visible), set(private))

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

    def test_sha_bound_component_timing_rules_are_exact(self):
        import json
        folder = ROOT/'maintenance/storage'
        qwen = json.loads((folder/'QWEN_FIRSTBLOCK_TIMING_VISIBILITY_20261006.json').read_text())
        optical = json.loads((folder/'OPTICAL_COMPONENT_TIMING_VISIBILITY_20261006.json').read_text())
        descriptors = [qwen] + optical['groups']
        payloads = [row['path'] for d in descriptors for row in d['files']]
        self.assertEqual(len(payloads), 92)
        self.assertEqual(self.ignored(payloads), set(payloads))
        visible = []
        for d in descriptors:
            base = Path(d['manifest']).parent.as_posix()
            visible += [base+'/README.md', base+'/SHA256SUMS.txt',
                        base+'/source_snapshot/new_benchmark.py', base+'/future_timing.json']
        self.assertEqual(self.ignored(visible), set())

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

    def test_vendor_assets_do_not_hide_source_or_new_files(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder)
            subprocess.run(['git', 'init', '-q', str(fixture)], check=True)
            shutil.copyfile(ROOT / '.gitignore', fixture / '.gitignore')
            with patch(__name__ + '.ROOT', fixture):
                self.check_vendor_assets()

    def check_vendor_assets(self):
        base = 'ABO_Lab_8um/original_a100/models/Qwen3-VL-Embedding-2B/'
        selected = [base+'config.json', base+'tokenizer.json', base+'ASSET_MANIFEST.json']
        self.assertEqual(self.ignored(selected), set(selected))
        visible = [base+'scripts/qwen3_vl_embedding.py', base+'scripts/new_fix.py',
                   base+'new_config.json', 'ABO_Lab_8um/original_a100/BACKEND_MANIFEST.json']
        self.assertEqual(self.ignored(visible), set())

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
