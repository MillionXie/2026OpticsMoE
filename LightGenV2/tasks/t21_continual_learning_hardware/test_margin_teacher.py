import unittest,torch,numpy as np
from .distill_margin_teacher import within_bands,bounded_teacher_margin,margin_rows,bounded_margin_rows,source_average,smooth_tail,robust_queries,robust_derivatives,source_partitions,retention_KL,centered_function_loss,whole_stage_band_key


class MarginTeacherTests(unittest.TestCase):
    def test_TEST_selector_uses_all_tasks_both_bounds_and_earlier_ties(self):
        bands=dict(eurosat=(483,533),clevr=(710,760));valid=dict(eurosat=dict(n=1000,correct=483),clevr=dict(n=1000,correct=760))
        self.assertEqual(whole_stage_band_key(valid,bands),(0,0))
        too_high=dict(eurosat=dict(n=1000,correct=561),clevr=dict(n=1000,correct=768))
        self.assertEqual(whole_stage_band_key(too_high,bands),(-28,-36))
        with self.assertRaises(ValueError):whole_stage_band_key(dict(eurosat=valid['eurosat']),bands)
        self.assertFalse(whole_stage_band_key(valid,bands)>whole_stage_band_key(valid,bands))
    def test_centered_retention_is_shift_invariant_and_does_not_saturate(self):
        reference=torch.tensor([[100.,-100.,0.]],requires_grad=True);same=(reference.detach()+27).requires_grad_()
        self.assertLess(float(centered_function_loss(same,reference)),1e-12)
        changed=(reference.detach()*2).requires_grad_();loss=centered_function_loss(changed,reference);loss.backward()
        self.assertGreater(float(changed.grad.norm()),0);self.assertIsNone(reference.grad)
    def test_LG_bounds_are_not_D2NN_bounds(self):
        values=dict(eurosat=dict(n=1000,correct=680),clevr=dict(n=1000,correct=733));self.assertTrue(within_bands(values,2,'moe'));self.assertFalse(within_bands(values,2))
        with self.assertRaises(ValueError):within_bands(values,3,'moe')
    def test_retention_teacher_is_frozen_and_equal_function_has_zero_KL(self):
        target=torch.randn(12,10,dtype=torch.float64,requires_grad=True);current=target.detach().clone().requires_grad_();loss=retention_KL(current,target)
        self.assertLess(abs(float(loss)),1e-12);loss.backward();self.assertIsNone(target.grad);self.assertTrue(torch.isfinite(current.grad).all())
    def test_bounded_loss_does_not_let_arbitrarily_wrong_logits_dominate(self):
        logits=torch.tensor([[0.,100.,0.],[2.,0.,0.]],dtype=torch.float64,requires_grad=True);labels=torch.tensor([0,0]);loss=bounded_margin_rows(logits,labels,torch.ones(2,dtype=torch.float64))
        self.assertTrue(bool((loss>=0).all()));self.assertTrue(bool((loss<=1/.7).all()));loss.mean().backward();self.assertTrue(torch.isfinite(logits.grad).all())
    def test_bag_partition_is_sourcewise_and_keeps_binary_pairs(self):
        tasks=np.tile(np.repeat('clevr',12),4);ids=np.tile(np.arange(12),4);kind=np.repeat([0,1],24);labels=ids%2
        folds=source_partitions(tasks,ids,kind,labels,3)
        for pool in (0,1):
            for source in range(12):self.assertEqual(len(set(folds[(kind==pool)&(ids==source)])),1)
            for source in range(0,12,2):self.assertEqual(folds[(kind==pool)&(ids==source)][0],folds[(kind==pool)&(ids==source+1)][0])
        self.assertEqual(set(folds),{0,1,2})
    def test_paired_queries_preserve_exact_TRAIN_binary_difference_before_clipping(self):
        torch.manual_seed(1);x=torch.ones(4,8);x[0,0]=1.1;x[0,1]=.9;x[1,0]=1.2;x[1,1]=.8;x[2,2]=1.1;x[2,3]=.9;x[3,2]=1.2;x[3,3]=.8
        y=torch.tensor([0,1,0,1]);tasks=np.repeat('clevr',4);ids=np.arange(4);cfg=dict(TRAIN_augmentations=3,TRAIN_noise_std=[.001,.003,.005]);q,t=robust_queries(x,y,tasks,ids,cfg)
        self.assertEqual(q.shape,(16,8));torch.testing.assert_close(q.mean(1),torch.ones(16))
        for j in (4,8,12):torch.testing.assert_close(q[j+1]-q[j],x[1]-x[0],atol=2e-7,rtol=1e-4)
    def test_robust_derivatives_keep_exact_sources_labels_and_pair_ids(self):
        torch.manual_seed(2);x=torch.rand(8,8)+1;x=x/x.mean(1,keepdim=True);labels=torch.tensor([0,1]*4);tasks=np.repeat('clevr',8);ids=np.arange(8)
        dx,dy,dt,di=robust_derivatives(x,labels,tasks,ids,np.arange(4),np.arange(4,8))
        self.assertEqual(dx.shape,(40,8));self.assertEqual(set(di),{4,5,6,7});torch.testing.assert_close(dx.mean(1),torch.ones(40));self.assertFalse(bool((dx[:4]==x[4:]).all(1).any()));self.assertEqual(dy.tolist(),[0,1]*20)
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
