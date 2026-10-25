import unittest,torch,json
from pathlib import Path
from .fit_B_balanced_residual_teacher import dkd_loss,whole_stage_selection,validate_profile,weak_pair_values
from .fit_augmented_residual_teacher import ResidualTeacher,residual_teacher_from_artifact


class BalancedTeacherTests(unittest.TestCase):
    def test_DKD_zero_at_equal_predictions_and_finite_gradient(self):
        torch.manual_seed(4);teacher=torch.randn(11,10);labels=torch.arange(11)%10;student=teacher.clone().requires_grad_();loss=dkd_loss(student,teacher,labels)
        self.assertLess(abs(float(loss)),1e-5);loss.backward();self.assertTrue(torch.isfinite(student.grad).all())
        other=torch.randn(11,10,requires_grad=True);loss=dkd_loss(other,teacher,labels);self.assertGreater(float(loss),0);loss.backward();self.assertTrue(torch.isfinite(other.grad).all())
    def test_sealed_teacher_width_and_legacy_reload(self):
        for width in (64,128):
            model=ResidualTeacher(torch.randn(10,12),width,0.).eval();artifact=dict(state_dict=model.state_dict(),teacher_dropout=0.)
            values=torch.randn(5,12);torch.testing.assert_close(model(values),residual_teacher_from_artifact(artifact)(values),atol=0,rtol=0)
    def test_exact_profile_rejects_more_sources_or_budget(self):
        cfg=json.loads(Path(__file__).with_name('config_B_balanced_residual_teacher_20261025.json').read_text());validate_profile(cfg)
        for key,value in [('CLEVR_derivative_sources',1000),('teacher_epochs',10000),('GPU',True)]:
            with self.assertRaises(ValueError):validate_profile(dict(cfg,**{key:value}))
    def test_whole_stage_not_per_cell_splice(self):
        floors={'a':.5,'b':.7};a={'a':{'balanced_accuracy':.6},'b':{'balanced_accuracy':.5}};b={'a':{'balanced_accuracy':.51},'b':{'balanced_accuracy':.69}}
        self.assertGreater(whole_stage_selection(b,floors),whole_stage_selection(a,floors))
    def test_paired_weak_noise_keeps_mean_one(self):
        x=torch.rand(20,13)+.5;x=x/x.mean(1,keepdim=True);a,b=weak_pair_values(x,x,torch.ones(13))
        torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(a.mean(1),torch.ones(20))


if __name__=='__main__':unittest.main()
