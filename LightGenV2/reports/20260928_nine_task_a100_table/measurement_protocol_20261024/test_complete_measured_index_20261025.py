"""No-model tests of the missing-only final-index arithmetic/enrollment."""
import ast
import json
from pathlib import Path
import unittest
import complete_measured_index_20261025 as m


class CompletionTests(unittest.TestCase):
    def test_original_indices_remain_pinned(self):
        self.assertEqual(m.sha(m.MAIN),m.MAIN_SHA)
        self.assertEqual(m.sha(m.TEMP),m.TEMP_SHA)
        self.assertEqual(sum(r[s] is not None for r in m.read(m.MAIN)['nine_metric_rows'] for s in ('ours','qwen')),14)

    def test_aggregate_does_not_pick_fast_round(self):
        refs=[dict(repeat=i,seconds200=x,joules200=x*100) for i,x in enumerate((10.,20.,60.),1)]
        value=m.summed(refs)
        self.assertEqual(value['calls'],600)
        self.assertEqual(value['sample_ms'],150.)
        self.assertEqual(value['sample_j'],15.)
        self.assertEqual(value['raw_references'],refs)

    def test_incomplete_or_reordered_rounds_rejected(self):
        with self.assertRaises(ValueError):m.summed([dict(repeat=1)])
        with self.assertRaises(ValueError):m.summed([dict(repeat=i) for i in (1,3,2)])

    def test_same_numerator_time_inverse_ratio(self):
        value=m.common_pair(dict(sample_ms=40.),dict(sample_ms=120.),6000,'same input')
        self.assertEqual(value['time_ratio'],3.)
        self.assertEqual(value['ours_per_s']/value['qwen_per_s'],3.)
        self.assertTrue(value['not_generation_throughput'])

    def test_no_neural_runtime_import(self):
        tree=ast.parse(Path(m.__file__).read_text())
        names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        names += [x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
        self.assertFalse(any(n and n.split('.')[0] in ('torch','transformers','diffusers') for n in names))

    def test_t12_qualified_pair_and_existing_formula(self):
        pair,common=m.t12_pair()
        self.assertEqual(common['total'],19596)
        self.assertAlmostEqual(common['time_ratio'],2.734396742182599)
        self.assertEqual(pair['qwen']['validation_platform'],'same native A100 GPU equivalence, not CPU')
        m.attach_t12_student_op(pair['ours'])
        self.assertEqual(pair['ours']['derived_NN_dense_MAC_subtotal']['OP600'],24084996304896)
        self.assertFalse(pair['ours']['derived_NN_dense_MAC_subtotal']['independent_server_formula_replay'])

    def test_complete_costs_keep_old14_and_shared_anchor(self):
        main,temp=m.complete(m.LGVQ/'a100_lgvq_gpu_checked_remaining_20261025_02')
        original=m.read(m.MAIN);old={r['id']:r for r in original['nine_metric_rows']}
        self.assertEqual(sum(r[s] is not None for r in main['nine_metric_rows'] for s in ('ours','qwen')),18)
        for row in main['nine_metric_rows']:
            for system in ('ours','qwen'):
                if old[row['id']][system] is not None:self.assertEqual(row[system],old[row['id']][system])
        self.assertEqual(main['shared_qwen_temporal100'],original['shared_qwen_temporal100'])
        self.assertIsNone(main['physical_optical_dense_equivalent_OP'])
        self.assertEqual(temp['rows'],m.read(m.TEMP)['rows'])
        self.assertEqual(len(temp['relative_to_lightgen']),13)


if __name__=='__main__':unittest.main()
