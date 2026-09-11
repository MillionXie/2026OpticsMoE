import unittest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import RetrievalHead
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import expand_retrieval_head, overlay_config, apply_contract


class ReadoutExpansionTests(unittest.TestCase):
    def test_signed_relu_preserves_outputs_and_all_other_weights(self):
        torch.manual_seed(42)
        old=RetrievalHead().eval()
        state={'readout.'+k:v.clone() for k,v in old.state_dict().items()}
        state['vision.optics.phase_raw']=torch.randn(4,224,224)
        state['language.optics.router.phase_raw']=torch.randn(478,478)
        state['vision.blocks.0.mlp.0.weight']=torch.randn(384,192)
        original={'metadata':{'fusion_alpha_min':.4001},'state_dict':state}
        converted=expand_retrieval_head(original,'relu128')
        self.assertNotIn('retrieval_head',original['metadata'])
        for k,v in state.items():
            if not k.startswith('readout.projection.'):
                self.assertIs(converted['state_dict'][k],v)
        new=RetrievalHead('relu128').eval()
        new.load_state_dict({k[len('readout.'):]:v for k,v in converted['state_dict'].items() if k.startswith('readout.')})
        latent=[torch.randn(n,192) for n in (7,49,77)]
        torch.testing.assert_close(new(latent),old(latent),rtol=2e-6,atol=2e-7)
        self.assertEqual(sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters()),32896)
        # Learning must escape the signed-pair linear initialization.
        before=new.projection[2].weight.detach().clone()
        optimizer=torch.optim.SGD(new.parameters(),lr=.1)
        loss=(new(latent)-torch.randn(3,64)).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters()))
        optimizer.step()
        self.assertGreater(float((new.projection[2].weight-before).abs().sum()),0.)
        # Loading a trained variant must not repeat initialization.
        self.assertIs(expand_retrieval_head(converted,'relu128'),converted)
        with self.assertRaises(ValueError):expand_retrieval_head(converted,'linear64')

    def test_profile_only_changes_serial_readout(self):
        original=overlay_config({'adapt':{},'augmentation':{}},'domain_distill_aligned_feature')
        variant=overlay_config({'adapt':{},'augmentation':{}},'domain_distill_feature_mlp')
        self.assertEqual(variant.pop('retrieval_head'),'relu128')
        original.pop('protocol');variant.pop('protocol')
        self.assertEqual(original,variant)
        head=RetrievalHead()
        payload={'metadata':{'fusion_alpha_min':.4001},
                 'state_dict':{'readout.'+k:v for k,v in head.state_dict().items()}}
        cfg=overlay_config({'adapt':{},'augmentation':{}},'domain_distill_feature_mlp')
        changed=apply_contract(payload,cfg)
        self.assertEqual(changed['metadata']['retrieval_head'],'relu128')
        repeated=apply_contract(changed,cfg)
        self.assertTrue(all(repeated['state_dict'][k] is v for k,v in changed['state_dict'].items()))

    def test_invalid_head_rejected_and_default_unchanged(self):
        for kind in ('attention','unknown',128):
            with self.assertRaises(ValueError):RetrievalHead(kind)
        head=RetrievalHead()
        self.assertIsInstance(head.projection,torch.nn.Linear)
        bad={'metadata':{},'state_dict':{'readout.projection.weight':torch.randn(63,384),
                                        'readout.projection.bias':torch.randn(63)}}
        with self.assertRaises(ValueError):expand_retrieval_head(bad,'relu128')


if __name__=='__main__':unittest.main()
