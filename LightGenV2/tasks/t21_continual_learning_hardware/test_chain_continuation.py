import copy
import unittest
import numpy as np

from .continue_native_chain import validate_resume,phase_batches


class ChainContinuationTests(unittest.TestCase):
    def setUp(self):
        self.old=dict(stage_epochs=[12,4,6,4],seed=73,batch_size=8,memory_cycles_per_epoch=50,optical_lr=.0003)
        self.cfg=dict(self.old,stage_epochs=[12,8,6,4])
        self.protocol=dict(architecture='moe',TEST_gradients=False,config=self.old)
        self.history=[dict(epoch=i) for i in range(5)]
        self.completion=dict(phase='B',best_epoch=4)

    def test_only_fixed_extension(self):
        validate_resume(self.protocol,self.history,self.completion,'moe',self.cfg)
        cfg=dict(self.cfg,optical_lr=.0001)
        with self.assertRaises(ValueError):validate_resume(self.protocol,self.history,self.completion,'moe',cfg)
        cfg=dict(self.cfg,stage_epochs=[12,9,6,4])
        with self.assertRaises(ValueError):validate_resume(self.protocol,self.history,self.completion,'moe',cfg)

    def test_original_epoch_and_parent_required(self):
        with self.assertRaises(ValueError):validate_resume(self.protocol,self.history[:-1],self.completion,'moe',self.cfg)
        with self.assertRaises(ValueError):validate_resume(self.protocol,self.history,self.completion,'d2nn',self.cfg)
        with self.assertRaises(ValueError):validate_resume(self.protocol,self.history,dict(phase='B',best_epoch=3),'moe',self.cfg)

    def test_no_TEST_gradient_parent(self):
        bad=dict(self.protocol,TEST_gradients=True)
        with self.assertRaises(ValueError):validate_resume(bad,self.history,self.completion,'moe',self.cfg)

    def test_full_current_and_old300_50cycles(self):
        selection=dict(eurosat=np.arange(300),clevr=np.arange(1000))
        batches=phase_batches(selection,('eurosat','clevr'),self.cfg,5,2)
        rows={n:np.concatenate([indices for task,indices in batches if task==n]) for n in selection}
        self.assertEqual(len(rows['eurosat']),15000)
        self.assertTrue(np.array_equal(np.bincount(rows['eurosat']),np.full(300,50)))
        self.assertTrue(np.array_equal(np.sort(rows['clevr']),np.arange(1000)))
        for task,indices in batches:
            if task=='clevr':
                self.assertTrue(np.array_equal(indices[1::2],indices[::2]+1))
                self.assertTrue(np.all(indices[::2]%2==0))

    def test_deterministic_schedule(self):
        selection=dict(eurosat=np.arange(300),clevr=np.arange(1000))
        a=phase_batches(selection,tuple(selection),self.cfg,5,2)
        b=phase_batches(selection,tuple(selection),self.cfg,5,2)
        self.assertTrue(all(x[0]==y[0] and np.array_equal(x[1],y[1]) for x,y in zip(a,b)))


if __name__=='__main__':unittest.main()
