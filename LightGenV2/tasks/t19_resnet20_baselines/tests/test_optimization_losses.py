import unittest
from unittest.mock import patch
import torch
from torch import nn
from LightGenV2.tasks.t19_resnet20_baselines.optimization_losses import (
    pose_distribution_loss,balanced_edit_loss,forward_features,SmallSaliconHead)


class OptimizationTests(unittest.TestCase):
    def test_pose_target_scale_invariant_and_invisible_channels_ignored(self):
        x=torch.randn(2,3,4,4,requires_grad=True);y=torch.rand_like(x);v=torch.tensor([[1,0,1],[1,1,0]])
        a=pose_distribution_loss(x,y,v);b=pose_distribution_loss(x,y*19,v)
        torch.testing.assert_close(a,b);a.backward()
        self.assertEqual(float(x.grad[0,1].abs().sum()),0.)
        self.assertGreater(float(x.grad[0,0].abs().sum()),0.)

    def test_pose_different_joint_peaks_fit_without_amplitude_conflict(self):
        target=torch.zeros(1,3,4,4);target[0,0,0,0]=1;target[0,1,2,1]=1;target[0,2,3,3]=1
        logits=nn.Parameter(torch.zeros_like(target));optimizer=torch.optim.Adam([logits],lr=.3)
        for _ in range(40):
            optimizer.zero_grad();loss=pose_distribution_loss(logits,target,torch.ones(1,3));loss.backward();optimizer.step()
        self.assertEqual(logits.flatten(2).argmax(-1).tolist(),[[0,9,15]])
        self.assertLess(float(loss),.02)

    def test_pose_all_invisible_is_finite_zero(self):
        x=torch.randn(1,2,3,3,requires_grad=True)
        loss=pose_distribution_loss(x,torch.zeros_like(x),torch.zeros(1,2));loss.backward()
        self.assertEqual(float(loss),0.);self.assertTrue(torch.isfinite(x.grad).all())

    def test_balanced_edit_gives_changed_cells_gradient(self):
        logits=torch.zeros(1,3,6,6,requires_grad=True);edit=torch.zeros(1,6,6,requires_grad=True)
        target=torch.zeros(1,6,6,dtype=torch.long);target[0,2,3]=2
        mask=target.ne(0).float()
        batch=dict(target_grid=target,edit_grid=mask,task_index=torch.zeros(1,dtype=torch.long))
        loss=balanced_edit_loss(dict(category_logits=logits,edit_logits=edit,task_logits=torch.zeros(1,4,requires_grad=True)),batch)
        loss.backward()
        self.assertGreater(abs(float(logits.grad[0,2,2,3])),20*abs(float(logits.grad[0,2,0,0])))
        self.assertLess(float(edit.grad[0,2,3]),0.)
        self.assertGreater(float(edit.grad[0,0,0]),0.)

    def test_small_salicon_head_has513_trainable_parameters(self):
        class Encoder(nn.Module):
            def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(1),requires_grad=False)
            def features(self,x):return (torch.ones(len(x),512,7,7),)
        with patch('LightGenV2.tasks.t19_resnet20_baselines.optimization_losses.FrozenResNet18',Encoder):
            model=SmallSaliconHead();output=model(torch.zeros(2,3,224,224))
        self.assertEqual(output.shape,(2,1,224,224))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),513)

    def test_existing_dense_head_feature_path_matches(self):
        from LightGenV2.tasks.t19_resnet20_baselines.frozen_visual_models import FrozenDenseModel
        class Encoder(nn.Module):
            def features(self,x):return tuple(torch.ones(len(x),c,s,s)*.1 for c,s in ((64,8),(128,4),(256,2),(512,1)))
        with patch('LightGenV2.tasks.t19_resnet20_baselines.frozen_visual_models.FrozenResNet18',Encoder):model=FrozenDenseModel(14,56)
        x=torch.randn(2,3,32,32)
        torch.testing.assert_close(model(x),forward_features(model,'lsp',model.encoder.features(x)),rtol=0,atol=0)

    def test_existing_openmoji_head_feature_path_matches(self):
        from LightGenV2.tasks.t19_resnet20_baselines.frozen_multimodal_models import FrozenOpenMoji
        class Encoder(nn.Module):
            def features(self,x):return (torch.ones(len(x),512,7,7)*.1,)
        with patch('LightGenV2.tasks.t19_resnet20_baselines.frozen_multimodal_models.FrozenResNet18',Encoder):model=FrozenOpenMoji('byte_gru')
        x=torch.randn(2,3,32,32);tokens=torch.ones(2,8,dtype=torch.long)
        a=model(x,tokens);b=forward_features(model,'openmoji',model.encoder.features(x),tokens)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
