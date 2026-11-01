import unittest
try:
    import torch
except (ImportError,OSError):
    torch=None
from ..scene_expanded_textdelta_recovery import configure_trainable,joint_loss


@unittest.skipIf(torch is None,'PyTorch runtime unavailable')
class JointConditions(unittest.TestCase):
    def test_all_existing_branches_receive_gradients_teacher_is_detached(self):
        from ..qwen3_scene_conditioning import QwenSDXLConditions
        torch.manual_seed(1042)
        head=QwenSDXLConditions(width=32,text_width=24,image_width=16,pooled_width=16)
        configure_trainable(head,'joint_conditions')
        features=[dict(hidden=torch.randn(n,2048),visual=torch.randn(4,2048))for n in [5,7]]
        features[1]['visual']=features[0]['visual'].clone()
        targets={k:torch.randn(*shape,requires_grad=True)for k,shape in
                 [('text',(2,77,24)),('pooled',(2,16)),('image',(2,257,16))]}
        loss,_=joint_loss(head,features,targets);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for prefix in ['project.','text.','pooled.','image.']:
            grads=[p.grad for n,p in head.named_parameters()if n.startswith(prefix)and p.grad is not None]
            self.assertTrue(grads,prefix)
            self.assertGreater(sum(float(g.abs().sum())for g in grads),0,prefix)
            self.assertTrue(all(torch.isfinite(g).all()for g in grads))
        self.assertTrue(all(t.grad is None for t in targets.values()))

    def test_repeated_prompt_image_target_does_not_reweight_joint_loss(self):
        from ..qwen3_scene_conditioning import QwenSDXLConditions
        torch.manual_seed(1042)
        head=QwenSDXLConditions(width=32,text_width=24,image_width=16,pooled_width=16)
        features=[dict(hidden=torch.randn(5,2048),visual=torch.randn(4,2048))for _ in range(2)]
        targets={k:torch.randn(*shape)for k,shape in
                 [('text',(2,77,24)),('pooled',(2,16)),('image',(2,257,16))]}
        first,_=joint_loss(head,features,targets)
        targets['image'][1].fill_(1000)
        features[1]['visual'].fill_(1000)
        second,_=joint_loss(head,features,targets)
        self.assertTrue(torch.equal(first,second))

    def test_existing_modes_keep_shared_projection_and_other_branches_frozen(self):
        from ..qwen3_scene_conditioning import QwenSDXLConditions
        head=QwenSDXLConditions(width=32)
        for mode,allowed in [('text_delta',('text.','pooled.')),('image_prefix',('image.',))]:
            configure_trainable(head,mode)
            self.assertTrue(all(p.requires_grad==n.startswith(allowed)for n,p in head.named_parameters()))


if __name__=='__main__':unittest.main()
