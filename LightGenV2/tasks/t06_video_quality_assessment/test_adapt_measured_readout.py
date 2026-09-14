import unittest
import numpy as np
from .adapt_measured_readout import split_indices, metrics, replay_audit, trainable_readout_names, official_partitions


class AdaptationTests(unittest.TestCase):
    def test_original_train_test(self):
        shared=dict(contract='test',checkpoint_sha256='same',target_mean=50.,target_std=10.)
        tr=dict(shared,dataset_split='train',video_ids=[f'train{i}' for i in range(2250)])
        te=dict(shared,video_ids=[f'test{i}' for i in range(558)])
        a,b=official_partitions(tr,te)
        self.assertEqual((len(a),len(b)),(2250,558))
        self.assertFalse(set(a)&set(b))
        te['video_ids'][0]=tr['video_ids'][0]
        with self.assertRaises(ValueError):official_partitions(tr,te)

    def test_full(self):
        train, holdout = split_indices(558, 1., 123)
        self.assertEqual(len(train), 558)
        self.assertEqual(len(holdout), 0)

    def test_holdout_disjoint_complete_repeatable(self):
        train, holdout = split_indices(558, .8, 20260914)
        self.assertEqual((len(train),len(holdout)), (446,112))
        self.assertFalse(set(train)&set(holdout))
        self.assertEqual(set(train)|set(holdout),set(range(558)))
        np.testing.assert_array_equal(train,split_indices(558,.8,20260914)[0])

    def test_metrics(self):
        self.assertAlmostEqual(metrics([1,2,3],[1,2,3])['srcc'],1.)
        self.assertEqual(metrics([1,1,1],[1,2,3])['srcc'],0.)
        with self.assertRaises(ValueError): metrics([1,np.nan,3],[1,2,3])

    def test_replay_gate_bounds_values_and_order(self):
        self.assertTrue(replay_audit([1.001,2.001,3.001],[1,2,3],[1,2,3])['passed'])
        self.assertFalse(replay_audit([1.1,2.1,3.1],[1,2,3],[1,2,3])['passed'])
        self.assertFalse(replay_audit([1.002,1.001,3],[1.001,1.002,3],[1,2,3])['passed'])

    def test_terminal_scope(self):
        expected={'output.4.weight','output.4.bias','compact_output.4.weight','compact_output.4.bias'}
        self.assertEqual(trainable_readout_names(expected|{'token_norm.weight'},'terminal'),expected)
        with self.assertRaises(ValueError):trainable_readout_names({'unexpected.weight'},'terminal')


if __name__=='__main__':unittest.main()
