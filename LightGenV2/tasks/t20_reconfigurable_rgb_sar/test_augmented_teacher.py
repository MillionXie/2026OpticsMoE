import ast,inspect,json,unittest
from pathlib import Path
import torch
from . import augmented_teacher as module
from .train_shared_frontend import d4_views,rgb_priority_selection,validate
import numpy as np


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

    def test_D4_keeps_labels_IDs_and_exact_flip(self):
        data=dict(images=np.arange(2*56*56*3).reshape(2,56,56,3),ids=np.array(['a','b']),labels=np.array([0,1]))
        views=d4_views(data);self.assertEqual(len(views),8)
        np.testing.assert_array_equal(views[1]['images'],data['images'][:,:,::-1,:])
        for v in views:
            np.testing.assert_array_equal(v['ids'],data['ids']);np.testing.assert_array_equal(v['labels'],data['labels'])
            self.assertTrue(v['images'].flags.c_contiguous)

    def test_VAL_only_RGB_priority_never_accepts_SAR_collapse(self):
        cfg=json.loads(Path(module.__file__).with_name('config_static_four_teacher_KD_20261025.json').read_text());validate(cfg)
        def report(r,s):return dict(scores=dict(RGB=dict(accuracy=r,loss=.1),SAR=dict(accuracy=s,loss=.2)))
        old=rgb_priority_selection(report(.809,.719),cfg,[0,1])
        self.assertLess(rgb_priority_selection(report(.99,.70),cfg,[0,1]),old)
        self.assertGreater(rgb_priority_selection(report(.85,.71),cfg,[0,1]),old)


if __name__=='__main__':unittest.main()
