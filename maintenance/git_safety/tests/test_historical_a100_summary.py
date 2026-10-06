import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    path = ROOT / 'LightGenV2/scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HistoricalSummaryTests(unittest.TestCase):
    def test_original_byte_identity(self):
        record = json.loads((ROOT / 'maintenance/storage/HISTORICAL_A100_SUMMARY_SOURCE_20261006.json').read_text())
        for row in record['files']:
            self.assertEqual(hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest(),
                             row['original_server_and_local_sha256'])

    def test_electronics_pooling_and_occurrences(self):
        tool = load('consolidate_latest_a100_electronics')
        result = tool.summarize([1, 2, 8, 9])
        self.assertEqual(result['mean'], 5)
        self.assertEqual(result['median'], 5)
        self.assertEqual(result['p95'], 8.85)
        self.assertEqual(tool.occurrences('lgvq_temporal', True)['frame_parallel_residual'], 2)
        self.assertEqual(tool.occurrences('openmoji', True)['language_to_vision_bridge'], 1)

    def test_qwen_equivalent_workload_and_preserved_source(self):
        tool = load('consolidate_latest_qwen_baselines_a100')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / 'batch_02'
            batch.mkdir()
            (batch / 'report.json').write_text(json.dumps({'performance': {'srcc': .7}, 'test_videos': 4}))
            (batch / 'batch_timing.csv').write_text('batch_size_videos,legacy_vision_block0_to_score_synchronized_wall_ms\n2,10\n2,14\n1,1000\n')
            (batch / 'telemetry.csv').write_text('phase,watts,utilization_percent\nactive:test:size2,100,50\nactive:test:size2,120,60\nidle,999,0\n')
            row = tool.temporal_row(root, 2)
            self.assertEqual(row['timing_calls'], 2)
            self.assertEqual(row['wall_mean_ms_per_call'], 12)
            self.assertEqual(row['equivalent_16_video_calls'], 8)
            self.assertEqual(row['equivalent_16_video_wall_ms'], 96)
            self.assertAlmostEqual(row['equivalent_16_video_energy_j'], 10.56)

    def test_copy_does_not_overwrite_evidence(self):
        tool = load('consolidate_latest_qwen_baselines_a100')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source', root / 'target'
            source.write_bytes(b'original')
            target.write_bytes(b'preserved')
            with self.assertRaises(FileExistsError):
                tool.copy_evidence(source, target)
            self.assertEqual(source.read_bytes(), b'original')
            self.assertEqual(target.read_bytes(), b'preserved')


if __name__ == '__main__':
    unittest.main()
