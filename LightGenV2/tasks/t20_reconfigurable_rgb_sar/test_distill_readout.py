import ast,inspect,unittest
import torch
from . import distill_fixed_readout as m


class ReadoutTeacherTests(unittest.TestCase):
    def test_wrong_teacher_does_not_replace_TRAIN_label(self):
        x=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64,requires_grad=True);target=torch.tensor([[8.,0.],[8.,0.]],dtype=torch.float64);y=torch.tensor([0,1]);v=m.supervised_KD(x,target,y,3.)
        expected=torch.nn.functional.cross_entropy(x,y,reduction='none',label_smoothing=.02)
        torch.testing.assert_close(v[1],expected[1]);self.assertGreater(float(v[0]),float(expected[0]));v.mean().backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_existing_inference_only_and_lock_before_TEST(self):
        code=inspect.getsource(m);tree=ast.parse(code)
        self.assertNotIn('nn.Linear(',code);self.assertIn('model.eval().requires_grad_(False)',code)
        self.assertLess(code.index('locked_before_TEST.json'),code.index("test=load_arrays(a.data,'test')"))
        self.assertIn('selection_uses_TEST=False',code);self.assertIn('legacy_parity_passed',code)


if __name__=='__main__':unittest.main()
