import ast,inspect,json,unittest
from pathlib import Path
import torch
from . import augmented_teacher as module


class TeacherTests(unittest.TestCase):
    def test_teacher_is_offline_and_does_not_open_TEST(self):
        tree=ast.parse(inspect.getsource(module));splits=[n.args[1].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_arrays' and isinstance(n.args[1],ast.Constant)]
        self.assertEqual(splits,['train','validation'])
        self.assertNotIn('build_static_four',inspect.getsource(module))

    def test_forward_gradient_and_augmentation_shape(self):
        torch.set_num_threads(2);torch.manual_seed(11);m=module.ImageTeacher();x=torch.randint(0,256,(4,3,56,56),dtype=torch.uint8)
        aug=module.augment(x,torch.Generator().manual_seed(11));self.assertEqual(aug.shape,x.shape)
        y=m(aug);self.assertEqual(y.shape,(4,10));y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_published_budget_and_no_TEST_access(self):
        c=json.loads(Path(module.__file__).with_name('config_augmented_teacher_20261025.json').read_text())
        self.assertFalse(c['TEST_access']);self.assertLessEqual(c['epochs'],100);self.assertEqual(c['per_process_GPU_memory_fraction'],.4)


if __name__=='__main__':unittest.main()
