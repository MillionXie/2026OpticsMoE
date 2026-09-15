import unittest
import numpy as np
from types import SimpleNamespace
from .adapt_measured_readout import split_indices, metrics, replay_audit, trainable_readout_names, official_partitions, training_order, loss_weights, partial_test_indices, readout_data_loss


class AdaptationTests(unittest.TestCase):
    def test_weighted_loss_and_gradients(self):
        try:import torch
        except (ImportError,OSError) as exc:self.skipTest('Torch runtime unavailable: '+str(exc))
        p=torch.tensor([-.5,.2,.7,1.5],requires_grad=True);y=torch.tensor([-.8,.5,1.,1.2])
        plain=readout_data_loss(p,y);ones=readout_data_loss(p,y,sample_weights=torch.ones_like(p))
        torch.testing.assert_close(plain,ones)
        weighted=readout_data_loss(p,y,sample_weights=torch.tensor([1.,1.,1.,5.]))
        self.assertNotAlmostEqual(float(plain.detach()),float(weighted.detach()),places=6)
        weighted.backward();self.assertTrue(torch.isfinite(p.grad).all())
        with self.assertRaises(ValueError):readout_data_loss(p,y,sample_weights=torch.tensor([1.,0.,1.,1.]))

    def test_partial_test_adaptation_is_fixed_disjoint_and_explicit(self):
        original=np.arange(2250);test=np.arange(2250,2808)
        train,held,adapted=partial_test_indices(original,test,.2,20260914)
        self.assertEqual((len(train),len(held),len(adapted)),(2362,446,112))
        self.assertEqual(set(train),set(original)|set(adapted))
        self.assertFalse(set(train)&set(held))
        self.assertEqual(set(adapted)|set(held),set(test))
        self.assertEqual(len(np.unique(np.concatenate([train,held]))),2808)
        for x,y in zip((train,held,adapted),partial_test_indices(original,test,.2,20260914)):
            np.testing.assert_array_equal(x,y)
        # Identity selection does not advance the training/dropout random stream.
        np.random.seed(2);expected=np.random.random()
        np.random.seed(2);partial_test_indices(original,test,.2,20260914)
        self.assertEqual(np.random.random(),expected)

    def test_no_test_adaptation_default_and_invalid_fractions(self):
        original=np.arange(2250);test=np.arange(2250,2808)
        tr,te,adapted=partial_test_indices(original,test,0,42)
        np.testing.assert_array_equal(tr,original);np.testing.assert_array_equal(te,test)
        self.assertEqual(len(adapted),0)
        for fraction in [-.1,1,2,float('nan'),.999]:
            with self.assertRaises(ValueError):partial_test_indices(original,test,fraction,42)
        with self.assertRaises(ValueError):partial_test_indices(original,original,.2,42)

    def test_training_order_exact_train_only(self):
        idx=np.arange(2250);targets=np.arange(2808)
        np.random.seed(8);order=training_order(idx,targets,'mos_stratified')
        self.assertEqual(set(order),set(idx));self.assertEqual(len(order),len(idx))
        self.assertFalse(set(order)&set(range(2250,2808)))
        # Ten equally sized quality strata are represented once per group.
        for start in range(0,2250,10):self.assertEqual(set(order[start:start+10]//225),set(range(10)))
        np.random.seed(8);np.testing.assert_array_equal(order,training_order(idx,targets,'mos_stratified'))
        small=np.array([3,5,8,11,15,19,21,25,28,30,35])
        self.assertEqual(set(training_order(small,targets,'mos_stratified')),set(small))

    def test_random_order_backwards_compatible(self):
        np.random.seed(4);expected=np.random.permutation(2250)
        np.random.seed(4);np.testing.assert_array_equal(expected,training_order(np.arange(2250),np.arange(2250)))

    def test_loss_weights(self):
        self.assertEqual(loss_weights(SimpleNamespace()),(1.,.2,.1))
        self.assertEqual(loss_weights(SimpleNamespace(reg_weight=.5,rank_weight=1.,corr_weight=.5)),(.5,1.,.5))
        for value in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):loss_weights(SimpleNamespace(rank_weight=value))
        with self.assertRaises(ValueError):loss_weights(SimpleNamespace(reg_weight=0,rank_weight=0,corr_weight=0))

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
