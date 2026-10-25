import json,unittest,torch
from pathlib import Path
from .continue_LG_C_teacher_views import amplitude_view,teacher_views,validate


class ActualTeacherTests(unittest.TestCase):
    def test_view_keeps_source_shape_range_and_identity(self):
        c=json.loads(Path(__file__).with_name('config_LG_C_teacher_views_20261025.json').read_text());validate(c);x=torch.rand(4,8,8);before=x.clone();v=amplitude_view(x,c,0)
        torch.testing.assert_close(x,before);self.assertEqual(v.shape,x.shape);self.assertFalse(torch.equal(x,v));self.assertTrue(bool((v>=0).all() and (v<=1).all()));torch.testing.assert_close(v,amplitude_view(x,c,0))
        with self.assertRaises(ValueError):validate(dict(c,TEST_gradients=True))

    def test_cached_targets_are_actual_nonlinear_teacher_forwards(self):
        c=json.loads(Path(__file__).with_name('config_LG_C_teacher_views_20261025.json').read_text());x=torch.rand(4,3,3)
        class Teacher(torch.nn.Module):
            def forward(self,v):return {'logits':v.square().flatten(1)}
        teacher=Teacher();inputs,targets=teacher_views(teacher,x,c,torch.device('cpu'))
        self.assertFalse(targets.requires_grad);self.assertEqual(len(inputs),8)
        for view in range(8):torch.testing.assert_close(targets[view],teacher(inputs[view])['logits'])


if __name__=='__main__':unittest.main()
