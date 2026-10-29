import unittest
import torch
from torch import nn
from LightGenV2.tasks.t12_text_to_image.scene_native_readouts import ReadoutMasters,modules,snapshot,restore
from LightGenV2.tasks.t12_text_to_image.scene_frontend_rebind import binding_metadata


def model():
    q=nn.Module();q.model=nn.Module();q.model.visual=nn.Module();v=q.model.visual
    v.merger=nn.Linear(4,4).bfloat16();v.deepstack_merger_list=nn.ModuleList([nn.Linear(4,4).bfloat16() for _ in range(3)])
    v.frozen_block=nn.Linear(4,4).bfloat16().requires_grad_(False)
    return q


class NativeReadouts(unittest.TestCase):
    def test_real_gradients_update_existing_BF16_modules_without_deployment_growth(self):
        torch.manual_seed(1042);q=model();count=sum(p.numel() for p in q.parameters());old=snapshot(q)
        masters=ReadoutMasters(q);opt=torch.optim.AdamW(masters.parameters(),lr=.01);v=q.model.visual
        x=v.frozen_block(torch.randn(2,4).bfloat16()).detach();out=v.merger(x)+sum(m(x) for m in v.deepstack_merger_list)
        out.float().square().mean().backward();masters.collect()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in masters.master))
        self.assertIsNone(v.frozen_block.weight.grad);opt.step();masters.copy_to_native()
        self.assertEqual(sum(p.numel() for p in q.parameters()),count)
        self.assertTrue(all(p.dtype==torch.bfloat16 for p in modules(q).parameters()))
        self.assertTrue(any(not torch.equal(snapshot(q)[k],value) for k,value in old.items()))
        new=snapshot(q);other=model();restore(other,new)
        self.assertTrue(all(torch.equal(snapshot(other)[k],value) for k,value in new.items()))
        restore(q,old);masters.reset_from_native()
        self.assertTrue(all(torch.equal(p.float(),m) for p,m in zip(masters.actual,masters.master)))

    def test_invalid_keys_shapes_and_nonfinite_readouts_are_rejected(self):
        q=model();state=snapshot(q)
        with self.assertRaises(ValueError):restore(q,dict(state,unrequested_layer=torch.ones(1)))
        key=next(iter(state));bad=dict(state);bad[key]=torch.zeros(1)
        with self.assertRaises(ValueError):restore(q,bad)
        bad=dict(state);bad[key]=state[key].clone().fill_(float('nan'))
        with self.assertRaises(ValueError):restore(q,bad)

    def test_TRAIN_native_recovery_needs_explicit_frozen_TF_and_no_new_structure(self):
        state=dict(optical_frontend=dict(checkpoint_sha256='old'))
        report=dict(complete=True,existing_native_readouts_also_trainable=True,head_UNet_VAE_retained_Qwen_TF_frozen=True,
            native_readout_parameters_trained=80,extra_deployment_parameters=0,parent_frontend_sha256='old',TEST_opened=False,DEV_or_TEST_forward_performed=False)
        self.assertTrue(binding_metadata(state,report,'new')['native_visual_readouts_adapted'])
        for change in [dict(extra_deployment_parameters=1),dict(head_UNet_VAE_retained_Qwen_TF_frozen=False),dict(DEV_or_TEST_forward_performed=True)]:
            with self.assertRaises(ValueError):binding_metadata(state,dict(report,**change),'new')


if __name__=='__main__':unittest.main()
