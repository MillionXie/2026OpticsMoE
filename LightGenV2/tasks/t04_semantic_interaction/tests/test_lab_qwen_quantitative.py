"""Portable synthetic fixtures: report identity, source hashes and plotting units."""
import json
from pathlib import Path
import tempfile
import unittest

from LightGenV2.tasks.t04_semantic_interaction.plot_lab_qwen_quantitative import (
    GROUPS, METRICS, OURS_SHA, QWEN_SHA, audit, sha, figure_tables,
)


class QuantitativeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        def metrics(value):
            return {group: {'samples': 1000 if group == 'overall' else 250,
                            **{key: value for key in METRICS}} for group in GROUPS}
        self.write('ours_lab_full_report.json', {
            'status': 'complete', 'checkpoint_sha256': OURS_SHA, 'test_count': 1000,
            'ccd_images': 6000, 'notes': 'no downstream retraining', 'metrics': metrics(.8655)})
        self.write('qwen_evaluation.json', {
            'test_samples': 1000, 'selected_epoch': 30, 'metrics': metrics(.812)})
        records = [{'sample_id': str(i), 'task': GROUPS[1+i//250],
                    **{key: .812 for key in METRICS}} for i in range(1000)]
        (self.root/'qwen_test_predictions.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in records), encoding='utf-8')
        self.manifest = {'qwen_checkpoint_sha256': QWEN_SHA, 'files': {
            name: {'sha256': sha(self.root/name)} for name in (
                'ours_lab_full_report.json', 'qwen_evaluation.json', 'qwen_test_predictions.jsonl')}}
        self.write('source_manifest.json', self.manifest)

    def write(self, name, content):
        (self.root/name).write_text(json.dumps(content), encoding='utf-8')

    def test_all_groups_metrics_and_percentage_point_units(self):
        result = audit(self.root)
        self.assertEqual(len(result['rows']), 40)
        self.assertAlmostEqual(result['rows'][0]['difference_pp'], 5.35)
        self.assertAlmostEqual(result['rows'][0]['relative_change_percent'], 6.5886699507389)
        self.assertIn('no paired per-case audit', result['protocol']['pairing_limit'])

    def test_source_tampering_is_rejected(self):
        with (self.root/'qwen_test_predictions.jsonl').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(AssertionError, 'Source SHA changed'):
            audit(self.root)

    def test_figure_tables_cover_all_metrics_and_both_operation_panels(self):
        result = audit(self.root)
        figures = figure_tables(result)
        self.assertEqual(len(figures['figure1_rows']), 8)
        self.assertEqual({r['metric'] for r in figures['figure1_rows']}, set(METRICS))
        self.assertEqual(sum(r['recommended_for_main'] for r in figures['figure1_rows']), 4)
        self.assertEqual([r['panel'] for r in figures['figure2_rows']], ['a']*4+['b']*4)
        for row in figures['figure1_rows']+figures['figure2_rows']:
            original = result['rows'][row['source_row']-6]
            self.assertEqual((row['group'], row['metric'], row['ours'], row['qwen']),
                             (original['group'], original['metric'], original['ours'], original['qwen']))

    def test_simulation_substitution_is_rejected_even_after_rehash(self):
        path = self.root/'ours_lab_full_report.json'
        report = json.loads(path.read_text())
        report['metrics']['overall']['changed_cell_accuracy'] = .8765
        self.write(path.name, report)
        self.manifest['files'][path.name]['sha256'] = sha(path)
        self.write('source_manifest.json', self.manifest)
        with self.assertRaises(AssertionError):
            audit(self.root)


if __name__ == '__main__':
    unittest.main()
