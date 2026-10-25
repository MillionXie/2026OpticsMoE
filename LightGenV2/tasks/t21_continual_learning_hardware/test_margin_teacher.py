import unittest,torch
from .distill_margin_teacher import within_bands,bounded_teacher_margin,margin_rows,source_average,smooth_tail


class MarginTeacherTests(unittest.TestCase):
    def test_inclusive_all_task_gate(self):
        scores={n:dict(n=1000,correct=k) for n,k in [('eurosat',512),('clevr',631),('speech_binary',522)]}
        self.assertTrue(within_bands(scores,3));scores['clevr']['correct']=630;self.assertFalse(within_bands(scores,3))
        scores['clevr']['correct']=681;scores['eurosat']['correct']=563;self.assertFalse(within_bands(scores,3))
        self.assertFalse(within_bands({'clevr':dict(n=1000,correct=710)},2))
    def test_wrong_teacher_not_inherited_and_margin_capped(self):
        logits=torch.tensor([[4.,0.,0.],[0.,4.,0.],[1.,1.2,0.]])
        torch.testing.assert_close(bounded_teacher_margin(logits,torch.tensor([0,0,1])),torch.tensor([2.,.5,.7]))
    def test_source_not_augmentation_count_weighted(self):
        x=torch.tensor([1.,1.,3.],requires_grad=True);means=source_average(x,torch.tensor([0,0,1]),2)
        torch.testing.assert_close(means,torch.tensor([1.,3.]));means.mean().backward();torch.testing.assert_close(x.grad,torch.tensor([.25,.25,.5]))
        with self.assertRaises(ValueError):source_average(x,torch.tensor([0,0,1]),3)
    def test_finite_tail_and_non_target_margin_gradient(self):
        logits=torch.zeros(5,10,dtype=torch.float64,requires_grad=True);labels=torch.arange(5);rows=margin_rows(logits,labels,torch.ones(5,dtype=torch.float64));threshold=torch.tensor(1.,requires_grad=True)
        loss=smooth_tail(rows,threshold)+rows.mean();loss.backward();self.assertTrue(torch.isfinite(logits.grad).all());self.assertTrue(torch.isfinite(threshold.grad));self.assertTrue(bool((logits.grad[range(5),labels]<0).all()))


if __name__=='__main__':unittest.main()
