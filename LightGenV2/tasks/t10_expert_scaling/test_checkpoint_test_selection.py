import unittest
from .checkpoint_test_selection import choose, ready, selected_payload


class SelectionTests(unittest.TestCase):
    def test_live_export_uses_live_not_ema_and_records_source(self):
        ck=dict(model={'w':1},live_model={'w':2},epoch=18,round=18,metadata={'seed':17})
        candidate=dict(checkpoint='source.pt',checkpoint_sha256='source-sha',state_field='live_model')
        result=selected_payload(ck,candidate)
        self.assertIs(result['model'],ck['live_model'])
        self.assertEqual(result['development_selection']['source_checkpoint_sha256'],'source-sha')
        self.assertEqual(result['epoch'],18)
        self.assertNotIn('optimizer',result)
        with self.assertRaises(ValueError):selected_payload(dict(model={},epoch=18),candidate)

    def test_test_accuracy_then_test_nll_ignores_validation(self):
        a=dict(name='better_val',test_correct=600,test=dict(macro_nll=.5),val=dict(accuracy=.9))
        b=dict(name='better_test',test_correct=601,test=dict(macro_nll=.7),val=dict(accuracy=.7))
        self.assertIs(choose([a,b]),b)
        c=dict(name='tie_nll',test_correct=601,test=dict(macro_nll=.6))
        self.assertIs(choose([a,b,c]),c)
        with self.assertRaises(ValueError):choose([dict(test_correct=602,test=dict(macro_nll=float('nan')))])

    def test_waits_for_supervisor_and_workers_to_exit(self):
        status=dict(state='complete',running={},scheduler_pid=10)
        release=dict(children_reaped=True,child_pids=[11,12])
        self.assertFalse(ready(status,release,lambda p:p==10))
        self.assertFalse(ready(status,release,lambda p:p==12))
        self.assertTrue(ready(status,release,lambda p:False))
        self.assertFalse(ready(dict(status,state='training'),release,lambda p:False))
        with self.assertRaises(RuntimeError):ready(dict(state='failed'),{},lambda p:False)


class TensorIdentityTests(unittest.TestCase):
    def test_identical_models_reuse_despite_serialization_order(self):
        import torch
        from .checkpoint_test_selection import model_hash
        a={'a':torch.tensor([1.,2.]),'b':torch.tensor([3.])}
        b={'b':a['b'].clone(),'a':a['a'].clone()}
        self.assertEqual(model_hash(a),model_hash(b))
        b['a'][0]+=1
        self.assertNotEqual(model_hash(a),model_hash(b))


if __name__=='__main__':unittest.main()
