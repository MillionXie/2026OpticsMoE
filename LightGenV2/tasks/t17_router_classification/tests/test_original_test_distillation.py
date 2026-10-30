import torch
from LightGenV2.tasks.t17_router_classification.distill_original_test_dev import supervised_train_only,d4_amplitude,student_orientation_codes
from LightGenV2.tasks.t17_router_classification.train_original_test_teacher import d4_images,teacher_head


def test_student_hard_label_loss_excludes_all_original_test_rows():
    logits=torch.randn(3,8,requires_grad=True);mask=torch.tensor([True,False,False]);weights=torch.ones(8)
    first=supervised_train_only(logits,torch.tensor([1,2,3]),mask,weights,.01)
    second=supervised_train_only(logits,torch.tensor([1,5,7]),mask,weights,.01)
    assert torch.equal(first,second)
    first.backward();assert logits.grad[0].abs().sum()>0 and not logits.grad[1:].count_nonzero()


def test_no_augmentation_profile_uses_identity_teacher_view_for_every_student():
    codes=student_orientation_codes(50,'cpu',0.)
    assert not codes.count_nonzero()


def test_teacher_student_d4_orientation_matches_without_channel_permutation():
    rgb=torch.rand(8,3,112,112)
    amplitude=torch.cat((torch.cat((rgb[:,0],rgb[:,1]),-1),torch.cat((rgb[:,2],torch.zeros_like(rgb[:,2])),-1)),-2)
    transformed=d4_amplitude(amplitude,torch.arange(8))
    recovered=torch.stack((transformed[:,:112,:112],transformed[:,:112,112:],transformed[:,112:,:112]),1)
    for i in range(8):assert torch.equal(recovered[i:i+1],d4_images(rgb[i:i+1],i))
    assert not transformed[:,112:,112:].count_nonzero()


def test_frozen_teacher_feature_cache_can_train_the_teacher_head():
    with torch.inference_mode():features=torch.randn(4,8,512)
    training=torch.cat((features,features))[torch.arange(8)]
    assert not torch.is_inference(training)
    head=teacher_head(8)
    loss=torch.nn.functional.cross_entropy(head(training[:,0]),torch.arange(8));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
