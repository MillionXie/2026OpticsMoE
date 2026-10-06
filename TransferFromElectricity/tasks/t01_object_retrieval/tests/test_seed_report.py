"""Prevent incomplete or mismatched experiments from becoming a seed average."""
import json
import tempfile
import unittest
from pathlib import Path
from TransferFromElectricity.tasks.t01_object_retrieval.report_seeds import summarize


class SeedReportTests(unittest.TestCase):
    def make_runs(self, root):
        runs = []
        for seed in (42, 43):
            for method in ('direct', 'qwen_lora'):
                path = root / f'{seed}_{method}'
                path.mkdir()
                result = {'method': method, 'git_sha': 'test-source', 'split_sha256': 'test-split',
                          'optimizer_updates': 1200, 'selected_epoch': 20,
                          'selected_live_test': {'top1_retrieval_accuracy': .8 if method == 'direct' else .82},
                          'selected_expert_phase': {'rms_change_rad': .9},
                          'ablations': {'initial_experts': {'top1_retrieval_accuracy': .7}}, 'export_max_error': 0}
                files = {'status.json': {'status': 'complete'}, 'final_report.json': result,
                         'protocol.json': {'seed': seed, 'method': method, 'data_seed': 42},
                         'environment.json': {'device': 'test RTX'}}
                for name, value in files.items():
                    (path / name).write_text(json.dumps(value))
                runs.append(path)
        return runs

    def test_paired_average_and_reject_mismatched_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = self.make_runs(root)
            report = summarize(runs, root / 'report')
            self.assertAlmostEqual(report['summary']['qwen_lora']['mean_top1'], .82)
            self.assertAlmostEqual(report['rows'][0]['qwen_minus_direct_pp'], 2.)
            path = runs[-1] / 'final_report.json'
            result = json.loads(path.read_text())
            result['split_sha256'] = 'different-test-samples'
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'sharing source, split'):
                summarize(runs, root / 'bad_report')

    def test_reject_missing_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = self.make_runs(root)
            with self.assertRaisesRegex(ValueError, 'complete seed pairs'):
                summarize(runs[:-1], root / 'bad_report')
