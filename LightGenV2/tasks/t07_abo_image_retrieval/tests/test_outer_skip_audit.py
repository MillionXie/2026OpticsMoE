import unittest
from types import SimpleNamespace
import torch
from torch import nn

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.outer_skip_audit import intervention, skip_statistics
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import refresh_bank_before_step


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_adapter=nn.Linear(4,4,bias=False)
        self.residual_logit=nn.Parameter(torch.zeros(()))
        self.remove_optical=False
    def forward(self,x):
        return x+self.residual_logit.sigmoid()*self.output_adapter(x)


class SkipAuditTests(unittest.TestCase):
    def setUp(self):
        self.model=SimpleNamespace(vision=TinyVision(),language=SimpleNamespace(remove_optical=False))
        self.x=torch.randn(3,2,4)
    def test_normal_exact_and_no_weight_change(self):
        before={n:v.clone() for n,v in self.model.vision.state_dict().items()}
        expected=self.model.vision(self.x)
        obs=[]
        with intervention(self.model,'normal',obs):
            self.assertTrue(torch.equal(expected,self.model.vision(self.x)))
        self.assertEqual(len(obs),3)
        for n,v in self.model.vision.state_dict().items():self.assertTrue(torch.equal(v,before[n]))
        self.assertFalse(self.model.vision._forward_hooks)
    def test_no_outer_skip_keeps_gated_update(self):
        expected=.5*self.model.vision.output_adapter(self.x)
        with intervention(self.model,'no_outer_skip',[]):
            self.assertTrue(torch.equal(expected,self.model.vision(self.x)))
    def test_skip_only_is_original_patch_not_rgb(self):
        with intervention(self.model,'skip_only',[]):
            self.assertIs(self.model.vision(self.x),self.x)
            self.assertFalse(self.model.language.remove_optical)
    def test_modal_removals_and_restore_on_error(self):
        for mode,expected in [('remove_vision_optics',(True,False)),('remove_language_optics',(False,True)),('remove_all_optics',(True,True))]:
            with self.assertRaisesRegex(RuntimeError,'test'):
                with intervention(self.model,mode,[]):
                    self.assertEqual((self.model.vision.remove_optical,self.model.language.remove_optical),expected)
                    raise RuntimeError('test')
            self.assertFalse(self.model.vision.remove_optical)
            self.assertFalse(self.model.language.remove_optical)
            self.assertFalse(self.model.vision.output_adapter._forward_hooks)
    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            with intervention(self.model,'unknown',[]):pass
    def test_scales_are_feature_statistics(self):
        s=skip_statistics(torch.ones(2,3,4),torch.ones(2,3,4)*4,.5)
        torch.testing.assert_close(s['update_to_input_rms'],torch.full((2,),2.))
        torch.testing.assert_close(s['cosine'],torch.ones(2))
    def test_bank_refresh_schedule(self):
        self.assertEqual([s for s in range(100) if refresh_bank_before_step(s,25)],[25,50,75])
        self.assertFalse(any(refresh_bank_before_step(s,0) for s in range(100)))
        self.assertFalse(refresh_bank_before_step(0,1))
    def test_bank_refresh_invalid(self):
        for s,n in [(-1,25),(0,-1),(1,True),(1,2.5)]:
            with self.assertRaises(ValueError):refresh_bank_before_step(s,n)


if __name__=='__main__':unittest.main()
